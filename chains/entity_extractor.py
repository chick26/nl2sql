"""
三级实体抽取与纠错 Chain
Level 1: 规则引擎 (正则 + 词典模糊匹配)
Level 2: 小模型 NER (BERT-CRF, 可选)
Level 3: LLM 兜底 (Qwen3-32B thinking)
"""
import json
import os
import re
import sys
from dataclasses import dataclass, field, asdict

from rapidfuzz import fuzz, process

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import config


# ──────────────────────── 数据结构 ────────────────────────

@dataclass
class EntityMatch:
    raw: str
    standard: str
    entity_type: str  # cable / segment / city / direction
    confidence: float
    source: str  # rule / ner / llm


@dataclass
class DirectionMatch:
    """方向实体: '中美' → site_a 国家前缀 + site_b 国家前缀"""
    raw: str                # 用户原始输入, 如 "中美"
    country_a: str          # 站点A国家前缀, 如 "中国"
    country_b: str          # 站点B国家前缀, 如 "美国"
    confidence: float
    source: str


@dataclass
class ExtractionResult:
    cable_names: list[EntityMatch] = field(default_factory=list)
    segments: list[EntityMatch] = field(default_factory=list)
    cities: list[EntityMatch] = field(default_factory=list)
    directions: list[DirectionMatch] = field(default_factory=list)
    normalized_query: str = ""

    def to_dict(self) -> dict:
        return {
            "cable_names": [asdict(e) for e in self.cable_names],
            "segments": [asdict(e) for e in self.segments],
            "cities": [asdict(e) for e in self.cities],
            "directions": [asdict(e) for e in self.directions],
            "normalized_query": self.normalized_query,
        }

    def entities_summary(self) -> str:
        """生成供 Text2SQL 使用的实体摘要"""
        parts = []
        for e in self.cable_names:
            parts.append(f"海缆名称: {e.standard} (原始输入: {e.raw})")
        for e in self.segments:
            parts.append(f"缆段: {e.standard} (原始输入: {e.raw})")
        for e in self.cities:
            parts.append(f"城市/站点: {e.standard} (原始输入: {e.raw})")
        for d in self.directions:
            parts.append(f"方向: {d.raw} → 站点A国家含\"{d.country_a}\" AND 站点B国家含\"{d.country_b}\" (需 JOIN sea_cable_segment_direction 表)")
        return "\n".join(parts) if parts else "无特定实体"


# ──────────────────────── 词典加载 ────────────────────────

def _load_dict(path: str) -> dict[str, list[str]]:
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


_cable_aliases: dict[str, list[str]] = {}
_segment_aliases: dict[str, list[str]] = {}
_city_aliases: dict[str, list[str]] = {}

# 反向索引: alias → standard_name
_cable_reverse: dict[str, str] = {}
_segment_reverse: dict[str, str] = {}
_city_reverse: dict[str, str] = {}


def _init_dictionaries():
    global _cable_aliases, _segment_aliases, _city_aliases
    global _cable_reverse, _segment_reverse, _city_reverse

    _cable_aliases = _load_dict(config.CABLE_DICT_PATH)
    _segment_aliases = _load_dict(config.SEGMENT_DICT_PATH)
    _city_aliases = _load_dict(config.CITY_DICT_PATH)

    # 构建反向索引
    for std, aliases in _cable_aliases.items():
        _cable_reverse[std.lower()] = std
        for a in aliases:
            _cable_reverse[a.lower()] = std

    for std, aliases in _segment_aliases.items():
        _segment_reverse[std.lower()] = std
        for a in aliases:
            _segment_reverse[a.lower()] = std

    for std, aliases in _city_aliases.items():
        _city_reverse[std] = std  # 中文不 lower
        for a in aliases:
            _city_reverse[a] = std
            _city_reverse[a.lower()] = std


_init_dictionaries()


# ──────────────────────── 方向词典 ────────────────────────
# 基于海缆段落方向静态关系表中 site_a / site_b 的国家前缀
# "中美" → 一端为"中国%"站点，另一端为"美国%"站点

# 方向简称 → (country_a_prefix, country_b_prefix)
# country 前缀必须与 sea_cable_segment_direction 表 site_a/site_b 的开头一致
DIRECTION_ALIASES: dict[str, tuple[str, str]] = {
    # 中国 ↔ X
    "中美":   ("中国", "美国"),
    "中日":   ("中国", "日本"),
    "中韩":   ("中国", "韩国"),
    "中新":   ("中国", "新加坡"),
    "中马":   ("中国", "马来西亚"),
    "中越":   ("中国", "越南"),
    "中泰":   ("中国", "泰国"),
    "中菲":   ("中国", "菲律宾"),
    "中柬":   ("中国", "柬埔寨"),
    "中欧":   ("中国", "法国"),      # 中欧方向终点为法国/意大利
    "中印":   ("中国", "印度"),
    "中缅":   ("中国", "缅甸"),
    "中台":   ("中国", "中国台湾"),   # 特殊: 中国大陆 ↔ 台湾
    # 新加坡 ↔ X
    "新欧":   ("新加坡", "法国"),
    "新日":   ("新加坡", "日本"),
    "新马":   ("新加坡", "马来西亚"),
    # 香港 ↔ X (香港站点可能是 "香港鹤咀" 或 "中国香港将军澳")
    "港美":   ("香港", "美国"),
    "港日":   ("香港", "日本"),
    "港韩":   ("香港", "韩国"),
    "港新":   ("香港", "新加坡"),
}

# 双字方向正则: 匹配 "中美"、"中日" 等
_DIRECTION_PATTERN = re.compile(
    r"(" + "|".join(re.escape(k) for k in sorted(DIRECTION_ALIASES.keys(), key=len, reverse=True)) + r")"
)

# 也支持全称: "中国到美国"、"中国-美国"、"中国至美国"
_COUNTRY_PREFIXES = [
    "中国", "美国", "日本", "韩国", "新加坡", "马来西亚", "越南", "泰国",
    "菲律宾", "印度", "柬埔寨", "法国", "意大利", "土耳其", "埃及",
    "巴基斯坦", "斯里兰卡", "阿联酋", "阿曼", "沙特", "缅甸",
    "孟加拉", "也门", "香港",
]
_FULL_DIRECTION_PATTERN = re.compile(
    r"(" + "|".join(re.escape(c) for c in _COUNTRY_PREFIXES) + r")"
    r"\s*[到至\-—～~↔]\s*"
    r"(" + "|".join(re.escape(c) for c in _COUNTRY_PREFIXES) + r")"
)


# ──────────────────────── Level 1: 规则引擎 ────────────────────────

# 缆段正则模式
_SEGMENT_PATTERNS = [
    re.compile(r"\b(S\d+[\.\-]?\d*[\.\-]?\d*[a-zA-Z]?)\b", re.IGNORECASE),
    re.compile(r"\b(Segment\s*\d+[\.\-]?\d*)\b", re.IGNORECASE),
]

# 海缆名称正则
_CABLE_PATTERNS = [
    re.compile(r"\b(AAE[\-_]?1)\b", re.IGNORECASE),
    re.compile(r"\b(APCN[\-_]?2)\b", re.IGNORECASE),
    re.compile(r"\b(APG)\b", re.IGNORECASE),
    re.compile(r"\b(SMW[\-_]?[345])\b", re.IGNORECASE),
    re.compile(r"\b(TPE)\b", re.IGNORECASE),
    re.compile(r"\b(NCP)\b", re.IGNORECASE),
    re.compile(r"\b(CSCN)\b", re.IGNORECASE),
    re.compile(r"\b(TSE[\-_]?1)\b", re.IGNORECASE),
    re.compile(r"\b(SEA[\-\s]?ME[\-\s]?WE[\-\s]?[345])\b", re.IGNORECASE),
]


def _rule_extract_cables(text: str) -> list[EntityMatch]:
    """用正则 + 词典匹配海缆名称"""
    matches = []
    seen = set()

    # 正则匹配
    for pat in _CABLE_PATTERNS:
        for m in pat.finditer(text):
            raw = m.group(1)
            std = _cable_reverse.get(raw.lower())
            if std and std not in seen:
                matches.append(EntityMatch(raw=raw, standard=std, entity_type="cable", confidence=0.95, source="rule"))
                seen.add(std)

    # 词典精确匹配(用于正则未覆盖的情况)
    text_lower = text.lower()
    for std_name in _cable_aliases:
        if std_name not in seen and std_name.lower() in text_lower:
            matches.append(EntityMatch(raw=std_name, standard=std_name, entity_type="cable", confidence=0.9, source="rule"))
            seen.add(std_name)

    return matches


def _rule_extract_segments(text: str) -> list[EntityMatch]:
    """用正则 + 词典匹配缆段"""
    matches = []
    seen = set()

    for pat in _SEGMENT_PATTERNS:
        for m in pat.finditer(text):
            raw = m.group(1).strip()
            # 先尝试精确匹配
            std = _segment_reverse.get(raw.lower())
            if not std:
                # 模糊匹配
                all_keys = list(_segment_reverse.keys())
                result = process.extractOne(raw.lower(), all_keys, scorer=fuzz.ratio)
                if result and result[1] >= config.FUZZY_MATCH_THRESHOLD:
                    std = _segment_reverse[result[0]]

            if std and std not in seen:
                matches.append(EntityMatch(raw=raw, standard=std, entity_type="segment", confidence=0.92, source="rule"))
                seen.add(std)

    return matches


def _rule_extract_cities(text: str) -> list[EntityMatch]:
    """用词典匹配城市/站点"""
    matches = []
    seen = set()

    # 按长度降序匹配(优先匹配更长的名称)
    all_aliases = sorted(_city_reverse.keys(), key=len, reverse=True)
    text_for_search = text

    for alias in all_aliases:
        if alias in text_for_search:
            std = _city_reverse[alias]
            if std not in seen:
                matches.append(EntityMatch(raw=alias, standard=std, entity_type="city", confidence=0.88, source="rule"))
                seen.add(std)

    return matches


def _rule_extract_directions(text: str) -> list[DirectionMatch]:
    """用词典匹配方向实体 (中美/中日/...) """
    matches = []
    seen = set()

    # 优先匹配全称: "中国到美国"
    for m in _FULL_DIRECTION_PATTERN.finditer(text):
        country_a, country_b = m.group(1), m.group(2)
        key = f"{country_a}-{country_b}"
        if key not in seen:
            matches.append(DirectionMatch(
                raw=m.group(0), country_a=country_a, country_b=country_b,
                confidence=0.95, source="rule",
            ))
            seen.add(key)

    # 再匹配简称: "中美"、"中日"
    for m in _DIRECTION_PATTERN.finditer(text):
        abbr = m.group(1)
        if abbr in DIRECTION_ALIASES:
            country_a, country_b = DIRECTION_ALIASES[abbr]
            key = f"{country_a}-{country_b}"
            if key not in seen:
                matches.append(DirectionMatch(
                    raw=abbr, country_a=country_a, country_b=country_b,
                    confidence=0.93, source="rule",
                ))
                seen.add(key)

    return matches


def rule_based_extract(query: str) -> ExtractionResult:
    """Level 1: 纯规则引擎抽取"""
    cables = _rule_extract_cables(query)
    segments = _rule_extract_segments(query)
    cities = _rule_extract_cities(query)
    directions = _rule_extract_directions(query)

    # 构建标准化查询: 将原始实体替换为标准名
    normalized = query
    for entity in cables + segments + cities:
        if entity.raw != entity.standard and entity.raw in normalized:
            normalized = normalized.replace(entity.raw, entity.standard)

    return ExtractionResult(
        cable_names=cables,
        segments=segments,
        cities=cities,
        directions=directions,
        normalized_query=normalized,
    )


# ──────────────────────── Level 2: 小模型 NER ────────────────────────

_ner_model = None
_ner_tokenizer = None


def _load_ner_model():
    """按需加载 BERT NER 模型"""
    global _ner_model, _ner_tokenizer
    if _ner_model is not None:
        return True

    if not os.path.exists(config.NER_MODEL_DIR):
        return False

    try:
        from transformers import AutoTokenizer, AutoModelForTokenClassification
        import torch

        _ner_tokenizer = AutoTokenizer.from_pretrained(config.NER_MODEL_DIR)
        _ner_model = AutoModelForTokenClassification.from_pretrained(config.NER_MODEL_DIR)
        _ner_model.eval()
        return True
    except Exception:
        return False


def ner_based_extract(query: str) -> ExtractionResult | None:
    """Level 2: 使用 BERT NER 模型提取实体"""
    if not _load_ner_model():
        return None  # 模型不可用，跳过

    try:
        import torch

        inputs = _ner_tokenizer(
            query,
            return_tensors="pt",
            truncation=True,
            max_length=128,
            return_offsets_mapping=True,
        )
        offset_mapping = inputs.pop("offset_mapping")[0].tolist()
        with torch.no_grad():
            outputs = _ner_model(**inputs)
        predictions = torch.argmax(outputs.logits, dim=-1)[0].tolist()
        labels = [config.NER_LABELS[p] for p in predictions]

        # 合并 BIO 标签为实体（基于 offset mapping，避免 [UNK]）
        entities = _merge_bio_entities_with_offsets(query, labels, offset_mapping)

        cables, segments, cities = [], [], []
        for etype, raw_text in entities:
            if etype == "CABLE":
                std = _cable_reverse.get(raw_text.lower(), raw_text)
                cables.append(EntityMatch(raw=raw_text, standard=std, entity_type="cable", confidence=0.85, source="ner"))
            elif etype == "SEGMENT":
                std = _segment_reverse.get(raw_text.lower(), raw_text)
                segments.append(EntityMatch(raw=raw_text, standard=std, entity_type="segment", confidence=0.85, source="ner"))
            elif etype == "CITY":
                std = _city_reverse.get(raw_text, _city_reverse.get(raw_text.lower(), raw_text))
                cities.append(EntityMatch(raw=raw_text, standard=std, entity_type="city", confidence=0.85, source="ner"))

        return ExtractionResult(cable_names=cables, segments=segments, cities=cities, normalized_query=query)
    except Exception:
        return None


def _merge_bio_entities(tokens: list[str], labels: list[str]) -> list[tuple[str, str]]:
    """将 BIO 标签序列合并为实体列表"""
    entities = []
    current_type = None
    current_tokens = []

    for token, label in zip(tokens, labels):
        if token in ("[CLS]", "[SEP]", "[PAD]"):
            continue

        if label.startswith("B-"):
            if current_type:
                entities.append((current_type, "".join(current_tokens).replace("##", "")))
            current_type = label[2:]
            current_tokens = [token]
        elif label.startswith("I-") and current_type == label[2:]:
            current_tokens.append(token)
        else:
            if current_type:
                entities.append((current_type, "".join(current_tokens).replace("##", "")))
                current_type = None
                current_tokens = []

    if current_type:
        entities.append((current_type, "".join(current_tokens).replace("##", "")))

    return entities


def _merge_bio_entities_with_offsets(
    text: str,
    labels: list[str],
    offsets: list[tuple[int, int]],
) -> list[tuple[str, str]]:
    """将 BIO 标签序列合并为实体列表（基于 offset mapping 还原原文子串）"""
    entities = []
    current = None  # {"type": str, "start": int, "end": int}

    for label, (start, end) in zip(labels, offsets):
        if start == 0 and end == 0:
            continue  # special tokens

        if label.startswith("B-"):
            if current:
                raw = text[current["start"]:current["end"]]
                if raw:
                    entities.append((current["type"], raw))
            current = {"type": label[2:], "start": start, "end": end}
        elif label.startswith("I-") and current and current["type"] == label[2:]:
            current["end"] = end
        else:
            if current:
                raw = text[current["start"]:current["end"]]
                if raw:
                    entities.append((current["type"], raw))
                current = None

    if current:
        raw = text[current["start"]:current["end"]]
        if raw:
            entities.append((current["type"], raw))

    return entities


# ──────────────────────── Level 3: LLM 兜底 ────────────────────────

def llm_fallback_extract(query: str) -> ExtractionResult:
    """Level 3: 使用 Qwen3-32B thinking 模式兜底"""
    from langchain_core.prompts import ChatPromptTemplate
    from llm_factory import create_chat_llm

    llm = create_chat_llm(thinking=True, temperature=0)

    cable_list = ", ".join(_cable_aliases.keys())
    segment_list = ", ".join(list(_segment_aliases.keys())[:30]) + " ..."
    city_list = ", ".join(list(_city_aliases.keys())[:30]) + " ..."

    prompt = ChatPromptTemplate.from_messages([
        ("system", f"""你是海缆领域实体抽取和纠错专家。
从用户输入中提取海缆名称、缆段名称、城市/站点，并纠正为标准名称。

标准海缆名称: {cable_list}
标准缆段（部分）: {segment_list}
标准站点（部分）: {city_list}

严格输出 JSON，不要输出其他内容:
{{"cable_names": [{{"raw": "原始", "standard": "标准名"}}], "segments": [...], "cities": [...]}}
如果没有对应实体，返回空数组。"""),
        ("human", "{query}"),
    ])

    chain = prompt | llm
    response = chain.invoke({"query": query})
    text = response.content.strip()

    # 解析 JSON
    if "```json" in text:
        text = text.split("```json")[-1].split("```")[0].strip()
    elif "```" in text:
        text = text.split("```")[1].split("```")[0].strip()

    try:
        data = json.loads(text)
        cables = [EntityMatch(raw=e["raw"], standard=e["standard"], entity_type="cable", confidence=0.75, source="llm") for e in data.get("cable_names", [])]
        segments = [EntityMatch(raw=e["raw"], standard=e["standard"], entity_type="segment", confidence=0.75, source="llm") for e in data.get("segments", [])]
        cities = [EntityMatch(raw=e["raw"], standard=e["standard"], entity_type="city", confidence=0.75, source="llm") for e in data.get("cities", [])]
        return ExtractionResult(cable_names=cables, segments=segments, cities=cities, normalized_query=query)
    except (json.JSONDecodeError, TypeError, KeyError):
        return ExtractionResult(normalized_query=query)


# ──────────────────────── 三级融合 ────────────────────────

def extract_entities(query: str) -> ExtractionResult:
    """
    三级实体抽取融合:
    1. 先用规则引擎
    2. 如果规则引擎置信度不足或未提取到，尝试 NER 模型
    3. 最后 LLM 兜底
    """
    # Level 1: 规则引擎
    result = rule_based_extract(query)

    all_entities = result.cable_names + result.segments + result.cities
    # 方向实体也算有效实体
    if result.directions:
        return result
    if all_entities:
        avg_conf = sum(e.confidence for e in all_entities) / len(all_entities)
        if avg_conf >= 0.85:
            return result  # 规则引擎已足够可靠

    # Level 2: NER 模型 (如果可用)
    ner_result = ner_based_extract(query)
    if ner_result:
        # 合并: NER 结果补充规则引擎未发现的实体
        existing_standards = {e.standard for e in result.cable_names}
        for e in ner_result.cable_names:
            if e.standard not in existing_standards:
                result.cable_names.append(e)

        existing_standards = {e.standard for e in result.segments}
        for e in ner_result.segments:
            if e.standard not in existing_standards:
                result.segments.append(e)

        existing_standards = {e.standard for e in result.cities}
        for e in ner_result.cities:
            if e.standard not in existing_standards:
                result.cities.append(e)

    all_entities = result.cable_names + result.segments + result.cities
    if all_entities:
        return result

    # Level 3: LLM 兜底 (仅在前两级都没有找到实体时)
    try:
        llm_result = llm_fallback_extract(query)
        if llm_result.cable_names or llm_result.segments or llm_result.cities:
            llm_result.normalized_query = result.normalized_query or query
            return llm_result
    except Exception:
        pass

    # 没有识别到任何实体也是合法的（如 "当前有多少故障？"）
    return result


if __name__ == "__main__":
    # 测试规则引擎
    test_queries = [
        "AAE1 在 S1-8 段现在有故障吗？",
        "APG 海缆香港到釜山的方向有影响吗？",
        "SMW5 最近的故障是什么？",
        "当前未修复的故障有哪些？",
        "中美方向当前有什么海缆故障？",
        "中日方向有多少故障？",
        "中国到美国的海缆有故障吗？",
        "TPE 海缆中美方向修了吗？",
    ]
    for q in test_queries:
        result = rule_based_extract(q)
        print(f"Q: {q}")
        print(f"  Cables:     {[(e.raw, e.standard) for e in result.cable_names]}")
        print(f"  Segments:   {[(e.raw, e.standard) for e in result.segments]}")
        print(f"  Cities:     {[(e.raw, e.standard) for e in result.cities]}")
        print(f"  Directions: {[(d.raw, d.country_a, d.country_b) for d in result.directions]}")
        print(f"  Summary:    {result.entities_summary()}")
        print()
