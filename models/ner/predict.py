"""
BERT NER 模型推理
"""
import os
import sys

import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
import config


class NERPredictor:
    """NER 模型推理器"""

    def __init__(self, model_dir: str = None):
        if model_dir is None:
            model_dir = config.NER_MODEL_DIR
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForTokenClassification.from_pretrained(model_dir)
        self.model.eval()
        self.id2label = {i: label for i, label in enumerate(config.NER_LABELS)}

    def predict(self, text: str) -> list[dict]:
        """
        对文本进行 NER 预测
        返回: [{"text": "AAE-1", "type": "CABLE", "start": 0, "end": 5}, ...]
        """
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=128,
            return_offsets_mapping=True,
        )
        offset_mapping = inputs.pop("offset_mapping")[0].tolist()

        with torch.no_grad():
            outputs = self.model(**inputs)
        predictions = torch.argmax(outputs.logits, dim=-1)[0].tolist()

        # 合并 BIO 为实体
        entities = []
        current = None

        for idx, (pred_id, offset) in enumerate(zip(predictions, offset_mapping)):
            start, end = offset
            if start == 0 and end == 0:
                continue  # special token

            label = self.id2label.get(pred_id, "O")

            if label.startswith("B-"):
                if current:
                    current["text"] = text[current["start"]:current["end"]]
                    entities.append(current)
                current = {"type": label[2:], "start": start, "end": end}
            elif label.startswith("I-") and current and current["type"] == label[2:]:
                current["end"] = end
            else:
                if current:
                    current["text"] = text[current["start"]:current["end"]]
                    entities.append(current)
                    current = None

        if current:
            current["text"] = text[current["start"]:current["end"]]
            entities.append(current)

        return entities


if __name__ == "__main__":
    if not os.path.exists(config.NER_MODEL_DIR):
        print(f"模型目录不存在: {config.NER_MODEL_DIR}")
        print("请先运行 train.py 训练模型")
        sys.exit(1)

    predictor = NERPredictor()

    test_texts = [
        "AAE1在S1-8段有故障吗？",
        "APG海缆香港到釜山方向有影响吗？",
        "SMW5最近的故障是什么？",
    ]
    for text in test_texts:
        entities = predictor.predict(text)
        print(f"文本: {text}")
        print(f"实体: {entities}\n")
