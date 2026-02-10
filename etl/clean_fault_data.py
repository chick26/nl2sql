"""
海缆故障数据清洗与入库脚本
- 读取原始 CSV
- 清洗 / 字段拆解 / 类型统一
- 写入 SQLite
"""
import json
import re
import sys
import os

import pandas as pd
from sqlalchemy import create_engine, text

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import config


# ─────────────────────── 辅助函数 ───────────────────────

def _parse_affect_info(raw: str) -> dict:
    """拆解 affect_info JSON 字段为 relay_num / affect_num / affect_rate"""
    default = {"relay_num": 0, "affect_num": 0, "affect_rate": ""}
    if pd.isna(raw) or not raw:
        return default
    try:
        d = json.loads(raw)
        return {
            "relay_num": int(d.get("relayNum", 0)),
            "affect_num": int(d.get("affectNum", 0)),
            "affect_rate": str(d.get("rate", "")),
        }
    except (json.JSONDecodeError, TypeError):
        return default


def _parse_fault_duration(raw: str) -> int | None:
    """从 other_info 中提取故障历时, 转换为分钟"""
    if pd.isna(raw) or not raw:
        return None
    try:
        d = json.loads(raw)
        duration_str = d.get("faultDuration", "")
    except (json.JSONDecodeError, TypeError):
        return None

    if not duration_str:
        return None

    total_minutes = 0
    day_match = re.search(r"(\d+)\s*天", duration_str)
    hour_match = re.search(r"(\d+)\s*小时", duration_str)
    minute_match = re.search(r"(\d+)\s*分", duration_str)
    if day_match:
        total_minutes += int(day_match.group(1)) * 1440
    if hour_match:
        total_minutes += int(hour_match.group(1)) * 60
    if minute_match:
        total_minutes += int(minute_match.group(1))
    return total_minutes if total_minutes > 0 else None


def _normalize_segment(seg: str) -> str:
    """统一缆段名称格式: 去掉多余空格"""
    if pd.isna(seg):
        return ""
    return str(seg).strip()


# ─────────────────────── 主清洗逻辑 ───────────────────────

def clean_fault_data(csv_path: str) -> pd.DataFrame:
    """读取并清洗故障 CSV"""
    df = pd.read_csv(csv_path, encoding="utf-8")

    # 过滤已删除记录
    if "is_delete" in df.columns:
        df = df[df["is_delete"] == 0].copy()

    # ── 保留 MVP 字段 ──
    keep_cols = [
        "affect_business", "affect_info", "create_time", "other_info",
        "pop_fault_seg", "pop_fault_seg_detail", "pop_fault_time",
        "pop_repair_charge_man", "pop_sys", "pop_type",
        "repair_status", "repair_status_name", "repair_progress",
        "affect_direction", "pop_repair_boat", "pop_repair_remark",
        "repair_boat", "repair_done_time",
    ]
    existing_cols = [c for c in keep_cols if c in df.columns]
    df = df[existing_cols].copy()

    # ── 拆解 affect_info ──
    if "affect_info" in df.columns:
        affect_parsed = df["affect_info"].apply(_parse_affect_info).apply(pd.Series)
        df = pd.concat([df, affect_parsed], axis=1)

    # ── 拆解 fault_duration ──
    if "other_info" in df.columns:
        df["fault_duration_minutes"] = df["other_info"].apply(_parse_fault_duration)

    # ── 标准化缆段 ──
    if "pop_fault_seg" in df.columns:
        df["pop_fault_seg"] = df["pop_fault_seg"].apply(_normalize_segment)

    # ── 时间字段统一 ──
    for col in ["pop_fault_time", "create_time", "repair_done_time"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    # ── 删除原始 JSON 字段(已拆解) ──
    df.drop(columns=["affect_info", "other_info"], errors="ignore", inplace=True)

    # ── 重置索引 ──
    df.reset_index(drop=True, inplace=True)
    df.index.name = "id"
    df = df.reset_index()

    return df


def clean_segment_data(csv_path: str) -> pd.DataFrame:
    """读取并清洗段落方向关系表"""
    df = pd.read_csv(csv_path, encoding="utf-8")

    # 标准化列名 → 英文
    column_map = {
        "海缆类别": "cable_category",
        "系统": "system_name",
        "段落": "segment",
        "段落标准": "segment_standard",
        "段落详情": "segment_detail",
        "段落描述": "segment_desc",
        "主干/分支": "trunk_or_branch",
        "主干段落": "trunk_segment",
        "主干段落详情": "trunk_segment_detail",
        "主干段落描述": "trunk_segment_desc",
        "方向序号": "direction_seq",
        "站点段": "site_section",
        "站点A": "site_a",
        "站点B": "site_b",
        "站点对_无向": "direction_undirected",
    }
    df.rename(columns=column_map, inplace=True)

    # 去首尾空格
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].str.strip()

    df.reset_index(drop=True, inplace=True)
    df.index.name = "id"
    df = df.reset_index()

    return df


# ─────────────────────── 入库 ───────────────────────

def load_to_db():
    """清洗并写入 SQLite"""
    os.makedirs(os.path.dirname(config.DB_PATH), exist_ok=True)

    engine = create_engine(config.DB_URI, echo=False)

    print("[1/2] 清洗故障数据...")
    df_fault = clean_fault_data(config.FAULT_CSV)
    print(f"  → 清洗后 {len(df_fault)} 条记录, {len(df_fault.columns)} 列")
    df_fault.to_sql(config.FAULT_TABLE, engine, if_exists="replace", index=False)
    print(f"  → 已写入表 {config.FAULT_TABLE}")

    print("[2/2] 清洗段落方向关系表...")
    df_seg = clean_segment_data(config.SEGMENT_CSV)
    print(f"  → 清洗后 {len(df_seg)} 条记录, {len(df_seg.columns)} 列")
    df_seg.to_sql(config.SEGMENT_TABLE, engine, if_exists="replace", index=False)
    print(f"  → 已写入表 {config.SEGMENT_TABLE}")

    # 创建索引
    with engine.connect() as conn:
        conn.execute(text(f"CREATE INDEX IF NOT EXISTS idx_fault_sys ON {config.FAULT_TABLE}(pop_sys)"))
        conn.execute(text(f"CREATE INDEX IF NOT EXISTS idx_fault_seg ON {config.FAULT_TABLE}(pop_fault_seg)"))
        conn.execute(text(f"CREATE INDEX IF NOT EXISTS idx_fault_time ON {config.FAULT_TABLE}(pop_fault_time)"))
        conn.execute(text(f"CREATE INDEX IF NOT EXISTS idx_fault_status ON {config.FAULT_TABLE}(repair_status)"))
        conn.execute(text(f"CREATE INDEX IF NOT EXISTS idx_seg_sys ON {config.SEGMENT_TABLE}(system_name)"))
        conn.execute(text(f"CREATE INDEX IF NOT EXISTS idx_seg_std ON {config.SEGMENT_TABLE}(segment_standard)"))
        conn.execute(text(f"CREATE INDEX IF NOT EXISTS idx_seg_a ON {config.SEGMENT_TABLE}(site_a)"))
        conn.execute(text(f"CREATE INDEX IF NOT EXISTS idx_seg_b ON {config.SEGMENT_TABLE}(site_b)"))
        conn.commit()
    print("  → 索引创建完成")

    print("\n✅ 数据入库完成:", config.DB_PATH)


if __name__ == "__main__":
    load_to_db()
