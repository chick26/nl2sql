"""
BERT-CRF NER 模型训练脚本
使用 bert-base-chinese 进行序列标注
"""
import json
import os
import sys

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    TrainingArguments,
    Trainer,
    DataCollatorForTokenClassification,
)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
import config

# ─── 标签体系 ───
LABELS = config.NER_LABELS  # ["O", "B-CABLE", "I-CABLE", "B-SEGMENT", "I-SEGMENT", "B-CITY", "I-CITY"]
LABEL2ID = {label: i for i, label in enumerate(LABELS)}
ID2LABEL = {i: label for i, label in enumerate(LABELS)}


class NERDataset(Dataset):
    """NER 数据集"""

    def __init__(self, file_path: str, tokenizer, max_length: int = 128):
        self.samples = []
        self.tokenizer = tokenizer
        self.max_length = max_length

        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                sample = json.loads(line.strip())
                self.samples.append(sample)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        text = sample["text"]
        tokens = sample["tokens"]
        labels = sample["labels"]

        # 使用 tokenizer 编码
        encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_offsets_mapping=True,
        )

        # 对齐标签到 subword tokens
        offset_mapping = encoding.pop("offset_mapping")
        label_ids = []

        # 构建字符到原始 token 标签的映射
        char_labels = ["O"] * len(text)
        pos = 0
        for token, label in zip(tokens, labels):
            token_start = text.find(token, pos)
            if token_start >= 0:
                for i in range(token_start, min(token_start + len(token), len(text))):
                    char_labels[i] = label
                pos = token_start + len(token)

        for offset in offset_mapping:
            start, end = offset
            if start == 0 and end == 0:
                label_ids.append(-100)  # special tokens
            else:
                # 取该 span 第一个字符的标签
                label_ids.append(LABEL2ID.get(char_labels[start], 0))

        encoding["labels"] = label_ids
        return {k: torch.tensor(v) for k, v in encoding.items()}


def compute_metrics(eval_pred):
    """计算评估指标"""
    import numpy as np

    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=-1)

    # 只计算非 -100 位置
    true_labels = []
    pred_labels = []
    for pred_seq, label_seq in zip(predictions, labels):
        for p, l in zip(pred_seq, label_seq):
            if l != -100:
                true_labels.append(l)
                pred_labels.append(p)

    correct = sum(1 for t, p in zip(true_labels, pred_labels) if t == p)
    total = len(true_labels)
    accuracy = correct / total if total > 0 else 0

    # Entity-level F1 (简化版)
    entity_correct = sum(1 for t, p in zip(true_labels, pred_labels) if t == p and t != 0)
    entity_total_true = sum(1 for t in true_labels if t != 0)
    entity_total_pred = sum(1 for p in pred_labels if p != 0)

    precision = entity_correct / entity_total_pred if entity_total_pred > 0 else 0
    recall = entity_correct / entity_total_true if entity_total_true > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def train(
    data_dir: str = None,
    model_name: str = "bert-base-chinese",
    output_dir: str = None,
    num_epochs: int = 5,
    batch_size: int = 32,
    learning_rate: float = 3e-5,
):
    """训练 NER 模型"""
    if data_dir is None:
        data_dir = os.path.join(os.path.dirname(__file__), "training_data")
    if output_dir is None:
        output_dir = config.NER_MODEL_DIR

    print(f"模型: {model_name}")
    print(f"训练数据: {data_dir}")
    print(f"输出目录: {output_dir}")

    # 加载 tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # 加载数据集
    train_dataset = NERDataset(os.path.join(data_dir, "train.jsonl"), tokenizer)
    val_dataset = NERDataset(os.path.join(data_dir, "val.jsonl"), tokenizer)

    print(f"训练集: {len(train_dataset)} 条")
    print(f"验证集: {len(val_dataset)} 条")

    # 加载模型
    model = AutoModelForTokenClassification.from_pretrained(
        model_name,
        num_labels=len(LABELS),
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )

    # 训练参数
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=learning_rate,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        logging_steps=50,
        save_total_limit=2,
        fp16=torch.cuda.is_available(),
    )

    # Data collator
    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    # 训练
    print("\n开始训练...")
    trainer.train()

    # 保存最佳模型
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"\n✅ 模型已保存至: {output_dir}")

    # 评估
    print("\n评估结果:")
    eval_results = trainer.evaluate()
    for key, value in eval_results.items():
        print(f"  {key}: {value:.4f}")


if __name__ == "__main__":
    train()
