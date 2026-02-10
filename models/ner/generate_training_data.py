"""
NER 训练数据自动生成
从标准词典 + 问题模板批量生成 BIO 标注数据
可选: 调用 Qwen3-32B 生成更多样化的问法
"""
import json
import os
import random
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
import config

# ─── 加载词典 ───

def _load_dict(path):
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

cable_dict = _load_dict(config.CABLE_DICT_PATH)
segment_dict = _load_dict(config.SEGMENT_DICT_PATH)
city_dict = _load_dict(config.CITY_DICT_PATH)

# ─── 问题模板 ───

TEMPLATES = [
    "{cable}海缆现在有故障吗？",
    "{cable}目前有什么故障？",
    "{cable}的{segment}段有故障吗？",
    "{cable}的{segment}段落故障修了吗？",
    "{cable}在{segment}段现在有故障吗？",
    "{cable}海缆{city}方向有影响吗？",
    "{cable}故障影响了{city}吗？",
    "{cable}的{segment}段修复进展如何？",
    "{cable}最近的故障是什么时候？",
    "{cable}今年有几次故障？",
    "目前{cable}有哪些未修复的故障？",
    "{city}到{city2}的海缆有故障吗？",
    "请问{cable}的故障情况",
    "{segment}段的故障什么时候能修好？",
    "{cable}海缆{segment}段影响业务吗？",
    "当前影响业务的{cable}故障有哪些？",
    "{cable}在{city}方向的故障修了吗？",
    "{cable}的{segment}段维修船是哪艘？",
    "最近{city}方向的海缆故障有几起？",
    "{cable}故障从{city}到{city2}的方向中断了吗？",
    "上周{cable}有新故障吗？",
    "今天{cable}的{segment}还在故障吗？",
    "{cable}的故障预计什么时候修复？",
    "目前有多少条海缆在故障中？",
    "当前未修复的故障有哪些？",
    "最近一周有几起海缆故障？",
    "影响业务的海缆故障清单",
    "{cable}故障持续多久了？",
    "{segment}段修复了吗？",
    "{cable}的{segment}段故障影响了多少中继？",
    "{cable}系统{segment}段落现在是什么状态？",
    "请帮我查一下{cable}的最新故障进展",
    "{city}相关的海缆有没有出故障？",
    "{cable}和{cable2}哪个故障多？",
    "今年故障次数最多的海缆是哪个？",
    "给我生成一份{cable}的故障报告",
    "{cable}的{segment}故障负责人是谁？",
    "{cable}海缆{segment}段落目前维修进展",
    "查询{cable}系统所有历史故障",
    "从{city}到{city2}经过哪些海缆段落？",
    "{cable}的故障有没有影响到{city}？",
    "目前维修中的故障有几个？",
    "{cable}的最后一次故障是什么时候修好的？",
    "所有待修复的故障列表",
    "今年{cable}总共故障了几次？",
    "哪些海缆故障影响了业务？",
    "查一下{cable}海缆{segment}的故障详情",
    "{cable}近期有故障吗",
    "{cable}的{segment}段什么原因故障的",
    "帮我统计一下今年各海缆的故障次数",
]


def _get_all_variants(std_name: str, alias_dict: dict) -> list[str]:
    """获取标准名 + 所有别名"""
    variants = [std_name]
    if std_name in alias_dict:
        variants.extend(alias_dict[std_name])
    return variants


def _char_tokenize(text: str) -> list[str]:
    """字符级分词(BERT 中文通常用字符级)"""
    tokens = []
    i = 0
    while i < len(text):
        ch = text[i]
        if ch.isascii() and (ch.isalnum() or ch in "-_."):
            # 英文/数字连续字符作为一个 token
            j = i + 1
            while j < len(text) and text[j].isascii() and (text[j].isalnum() or text[j] in "-_."):
                j += 1
            tokens.append(text[i:j])
            i = j
        else:
            tokens.append(ch)
            i += 1
    return tokens


def _annotate_bio(text: str, entities: list[tuple[str, str, str]]) -> list[tuple[str, str]]:
    """
    对文本进行 BIO 标注
    entities: [(entity_text, entity_type, standard_name), ...]
    entity_type: CABLE / SEGMENT / CITY
    """
    # 找到每个实体在文本中的位置
    annotations = []  # (start, end, type)
    for ent_text, ent_type, _ in entities:
        start = text.find(ent_text)
        if start >= 0:
            annotations.append((start, start + len(ent_text), ent_type))

    # 按位置排序
    annotations.sort(key=lambda x: x[0])

    # 字符级标注
    char_labels = ["O"] * len(text)
    for start, end, etype in annotations:
        if all(char_labels[i] == "O" for i in range(start, end)):
            char_labels[start] = f"B-{etype}"
            for i in range(start + 1, end):
                char_labels[i] = f"I-{etype}"

    # token 化并对齐标签
    tokens = _char_tokenize(text)
    token_labels = []
    pos = 0
    for token in tokens:
        token_start = text.find(token, pos)
        if token_start < 0:
            token_labels.append("O")
            pos += len(token)
            continue
        # 取 token 中第一个字符的标签
        label = char_labels[token_start]
        token_labels.append(label)
        pos = token_start + len(token)

    return list(zip(tokens, token_labels))


def generate_samples(num_samples: int = 5000, seed: int = 42) -> list[dict]:
    """
    生成训练样本
    返回: [{"tokens": [...], "labels": [...], "text": "..."}, ...]
    """
    random.seed(seed)
    samples = []

    cable_standards = list(cable_dict.keys())
    segment_standards = list(segment_dict.keys())
    city_standards = [c for c in city_dict.keys() if len(c) >= 2]

    for _ in range(num_samples):
        template = random.choice(TEMPLATES)

        entities = []  # (text_in_query, type, standard)

        # 填充模板
        filled = template

        if "{cable}" in filled:
            std = random.choice(cable_standards)
            variant = random.choice(_get_all_variants(std, cable_dict))
            filled = filled.replace("{cable}", variant, 1)
            entities.append((variant, "CABLE", std))

        if "{cable2}" in filled:
            std = random.choice(cable_standards)
            variant = random.choice(_get_all_variants(std, cable_dict))
            filled = filled.replace("{cable2}", variant, 1)
            entities.append((variant, "CABLE", std))

        if "{segment}" in filled:
            std = random.choice(segment_standards)
            variant = random.choice(_get_all_variants(std, segment_dict))
            filled = filled.replace("{segment}", variant, 1)
            entities.append((variant, "SEGMENT", std))

        if "{city}" in filled:
            std = random.choice(city_standards)
            variants = _get_all_variants(std, city_dict)
            variant = random.choice(variants) if variants else std
            filled = filled.replace("{city}", variant, 1)
            entities.append((variant, "CITY", std))

        if "{city2}" in filled:
            std = random.choice(city_standards)
            variants = _get_all_variants(std, city_dict)
            variant = random.choice(variants) if variants else std
            filled = filled.replace("{city2}", variant, 1)
            entities.append((variant, "CITY", std))

        # 标注
        annotated = _annotate_bio(filled, entities)
        tokens = [t for t, _ in annotated]
        labels = [l for _, l in annotated]

        samples.append({
            "tokens": tokens,
            "labels": labels,
            "text": filled,
        })

    return samples


def save_samples(samples: list[dict], output_dir: str):
    """保存为 JSON Lines 格式"""
    os.makedirs(output_dir, exist_ok=True)

    # 划分数据集
    random.shuffle(samples)
    n = len(samples)
    train_end = int(n * 0.8)
    val_end = int(n * 0.9)

    splits = {
        "train": samples[:train_end],
        "val": samples[train_end:val_end],
        "test": samples[val_end:],
    }

    for split_name, split_data in splits.items():
        path = os.path.join(output_dir, f"{split_name}.jsonl")
        with open(path, "w", encoding="utf-8") as f:
            for sample in split_data:
                f.write(json.dumps(sample, ensure_ascii=False) + "\n")
        print(f"  {split_name}: {len(split_data)} 条 → {path}")


if __name__ == "__main__":
    output_dir = os.path.join(os.path.dirname(__file__), "training_data")
    print("正在生成 NER 训练数据...")
    samples = generate_samples(num_samples=5000)
    print(f"生成 {len(samples)} 条样本")
    save_samples(samples, output_dir)

    # 展示几个样本
    print("\n示例样本:")
    for s in samples[:3]:
        print(f"  文本: {s['text']}")
        entities = []
        current = None
        for t, l in zip(s["tokens"], s["labels"]):
            if l.startswith("B-"):
                if current:
                    entities.append(current)
                current = {"type": l[2:], "text": t}
            elif l.startswith("I-") and current:
                current["text"] += t
            else:
                if current:
                    entities.append(current)
                    current = None
        if current:
            entities.append(current)
        print(f"  实体: {entities}")
        print()
