"""
海缆 NL2SQL 系统全局配置
"""
import os

# ─── Qwen3-32B API 配置 ───
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "http://10.120.84.7:8001/v1")
LLM_MODEL_NAME = os.getenv("LLM_MODEL_NAME", "Qwen/Qwen3-32B")
LLM_API_KEY = os.getenv("LLM_API_KEY", "test")

# ─── 数据库配置 ───
DB_PATH = os.path.join(os.path.dirname(__file__), "data", "sea_cable.db")
DB_URI = f"sqlite:///{DB_PATH}"

# ─── 数据文件路径 ───
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
RAW_DIR = os.path.join(DATA_DIR, "raw")
DICT_DIR = os.path.join(DATA_DIR, "dictionaries")

FAULT_CSV = os.path.join(os.path.dirname(__file__), "hk_sea_cable_fault_info.csv")
SEGMENT_CSV = os.path.join(os.path.dirname(__file__), "海缆段落方向静态关系表.csv")

# ─── 词典文件路径 ───
CABLE_DICT_PATH = os.path.join(DICT_DIR, "cable_aliases.json")
SEGMENT_DICT_PATH = os.path.join(DICT_DIR, "segment_aliases.json")
CITY_DICT_PATH = os.path.join(DICT_DIR, "city_aliases.json")

# ─── 表名 ───
FAULT_TABLE = "sea_cable_fault"
SEGMENT_TABLE = "sea_cable_segment_direction"

# ─── SQL 安全配置 ───
ALLOWED_TABLES = [FAULT_TABLE, SEGMENT_TABLE]
FORBIDDEN_SQL_KEYWORDS = ["DROP", "DELETE", "UPDATE", "INSERT", "ALTER", "TRUNCATE", "CREATE"]

# ─── NER 模型配置 ───
NER_MODEL_DIR = os.path.join(os.path.dirname(__file__), "models", "ner", "saved_model")
NER_LABELS = ["O", "B-CABLE", "I-CABLE", "B-SEGMENT", "I-SEGMENT", "B-CITY", "I-CITY"]

# ─── rapidfuzz 模糊匹配阈值 ───
FUZZY_MATCH_THRESHOLD = 75
