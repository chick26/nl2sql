"""
方案B: Qwen3-0.6B LoRA 微调 - 端到端实体纠错
输入: 用户自然语言文本
输出: JSON 格式的标准化实体

使用方式:
  python finetune.py --data_dir training_data --output_dir saved_model
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
import config


def generate_correction_data(output_dir: str, num_samples: int = 3000):
    """
    生成端到端纠错训练数据
    格式: {"input": "用户问题", "output": JSON字符串}
    """
    import random

    cable_dict = json.load(open(config.CABLE_DICT_PATH, encoding="utf-8"))
    segment_dict = json.load(open(config.SEGMENT_DICT_PATH, encoding="utf-8"))
    city_dict = json.load(open(config.CITY_DICT_PATH, encoding="utf-8"))

    templates = [
        "{cable}海缆现在有故障吗？",
        "{cable}的{segment}段有故障吗？",
        "{cable}在{segment}段现在有故障吗？",
        "{cable}海缆{city}方向有影响吗？",
        "{cable}故障影响了{city}吗？",
        "{cable}的{segment}段修复进展如何？",
        "目前{cable}有哪些未修复的故障？",
        "{city}到{city2}的海缆有故障吗？",
        "{cable}在{city}方向的故障修了吗？",
        "当前影响业务的{cable}故障有哪些？",
        "{cable}的{segment}段维修船是哪艘？",
        "{cable}故障从{city}到{city2}的方向中断了吗？",
        "当前未修复的故障有哪些？",
        "目前有多少条海缆在故障中？",
        "{cable}今年有几次故障？",
    ]

    random.seed(42)
    samples = []

    cable_standards = list(cable_dict.keys())
    segment_standards = list(segment_dict.keys())
    city_standards = [c for c in city_dict.keys() if len(c) >= 2 and city_dict[c]]

    for _ in range(num_samples):
        template = random.choice(templates)
        output = {"cable_name": None, "segment": None, "city": None}

        filled = template

        if "{cable}" in filled:
            std = random.choice(cable_standards)
            variants = [std] + cable_dict.get(std, [])
            variant = random.choice(variants)
            filled = filled.replace("{cable}", variant, 1)
            output["cable_name"] = std

        if "{segment}" in filled:
            std = random.choice(segment_standards)
            variants = [std] + segment_dict.get(std, [])
            variant = random.choice(variants)
            filled = filled.replace("{segment}", variant, 1)
            output["segment"] = std

        if "{city}" in filled:
            std = random.choice(city_standards)
            variants = [std] + city_dict.get(std, [])
            variant = random.choice([v for v in variants if v]) if variants else std
            filled = filled.replace("{city}", variant, 1)
            output["city"] = std

        if "{city2}" in filled:
            std = random.choice(city_standards)
            variants = [std] + city_dict.get(std, [])
            variant = random.choice([v for v in variants if v]) if variants else std
            filled = filled.replace("{city2}", variant, 1)
            # city2 也记录在 city 字段（用逗号分隔）
            if output["city"]:
                output["city"] = output["city"] + "," + std
            else:
                output["city"] = std

        samples.append({
            "input": filled,
            "output": json.dumps(output, ensure_ascii=False),
        })

    # 保存
    os.makedirs(output_dir, exist_ok=True)
    random.shuffle(samples)
    n = len(samples)

    for split_name, split_data in [
        ("train", samples[:int(n*0.8)]),
        ("val", samples[int(n*0.8):int(n*0.9)]),
        ("test", samples[int(n*0.9):]),
    ]:
        path = os.path.join(output_dir, f"{split_name}.jsonl")
        with open(path, "w", encoding="utf-8") as f:
            for s in split_data:
                f.write(json.dumps(s, ensure_ascii=False) + "\n")
        print(f"  {split_name}: {len(split_data)} 条 → {path}")


def train_lora(data_dir: str, output_dir: str, base_model: str = "Qwen/Qwen3-0.6B"):
    """
    使用 LoRA 微调 Qwen3-0.6B

    注意: 此脚本需要在有 GPU 的环境中运行
    依赖: pip install peft transformers trl datasets
    """
    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
        from peft import LoraConfig, get_peft_model, TaskType
        from trl import SFTTrainer
        from datasets import load_dataset
    except ImportError as e:
        print(f"缺少依赖: {e}")
        print("请安装: pip install peft trl datasets")
        return

    print(f"基座模型: {base_model}")
    print(f"训练数据: {data_dir}")

    # 加载 tokenizer 和模型
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        trust_remote_code=True,
        device_map="auto",
    )

    # LoRA 配置
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # 加载数据
    dataset = load_dataset("json", data_files={
        "train": os.path.join(data_dir, "train.jsonl"),
        "validation": os.path.join(data_dir, "val.jsonl"),
    })

    def format_prompt(example):
        return {
            "text": f"<|im_start|>system\n你是海缆领域实体纠错助手。从用户输入中提取海缆名称、缆段、城市实体，并纠正为标准名称。输出JSON格式。<|im_end|>\n<|im_start|>user\n{example['input']}<|im_end|>\n<|im_start|>assistant\n{example['output']}<|im_end|>",
        }

    dataset = dataset.map(format_prompt)

    # 训练参数
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=3,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        learning_rate=2e-4,
        warmup_steps=100,
        logging_steps=50,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        fp16=True,
    )

    # Trainer
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        tokenizer=tokenizer,
        dataset_text_field="text",
        max_seq_length=256,
    )

    trainer.train()
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"\n✅ 模型已保存至: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Qwen3-0.6B LoRA 微调")
    parser.add_argument("--action", choices=["generate", "train"], default="generate")
    parser.add_argument("--data_dir", default=os.path.join(os.path.dirname(__file__), "training_data"))
    parser.add_argument("--output_dir", default=os.path.join(os.path.dirname(__file__), "saved_model"))
    parser.add_argument("--base_model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--num_samples", type=int, default=3000)
    args = parser.parse_args()

    if args.action == "generate":
        print("生成纠错训练数据...")
        generate_correction_data(args.data_dir, args.num_samples)
    elif args.action == "train":
        train_lora(args.data_dir, args.output_dir, args.base_model)
