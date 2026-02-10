"""
从海缆段落方向静态关系表和故障表自动构建标准别名词典
生成三份 JSON:
  - cable_aliases.json   海缆系统别名
  - segment_aliases.json 缆段别名
  - city_aliases.json    城市/站点别名
"""
import json
import os
import re
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import config


def _generate_cable_aliases(system_names: list[str]) -> dict[str, list[str]]:
    """
    根据标准海缆系统名生成常见别名变体
    AAE-1 → AAE1, AAE_1, aae-1, aae1
    SMW5  → SMW-5, smw5, SEA-ME-WE5
    """
    aliases: dict[str, list[str]] = {}
    for name in sorted(set(system_names)):
        name = name.strip()
        if not name:
            continue
        variants = set()
        # 小写
        variants.add(name.lower())
        # 去掉连字符
        variants.add(name.replace("-", ""))
        variants.add(name.replace("-", "").lower())
        # 连字符替换为下划线
        variants.add(name.replace("-", "_"))
        variants.add(name.replace("-", "_").lower())
        # 加连字符 (如 SMW5 → SMW-5)
        m = re.match(r"^([A-Za-z]+)(\d.*)$", name)
        if m:
            with_dash = f"{m.group(1)}-{m.group(2)}"
            variants.add(with_dash)
            variants.add(with_dash.lower())
        # 特殊映射
        special = {
            "SMW5": ["SEA-ME-WE5", "SEA-ME-WE 5", "SeaMeWe5"],
            "SMW4": ["SEA-ME-WE4", "SEA-ME-WE 4", "SeaMeWe4"],
            "SMW3": ["SEA-ME-WE3", "SEA-ME-WE 3"],
            "AAE-1": ["Asia Africa Europe-1", "Asia-Africa-Europe-1"],
            "APCN2": ["APCN-2", "Asia Pacific Cable Network 2"],
            "NCP": ["New Cross Pacific", "NewCrossPacific"],
            "TPE": ["Trans-Pacific Express"],
            "TSE-1": ["TSE1"],
        }
        if name in special:
            variants.update(special[name])

        # 移除标准名本身
        variants.discard(name)
        aliases[name] = sorted(variants)
    return aliases


def _generate_segment_aliases(segments: list[str]) -> dict[str, list[str]]:
    """
    根据标准缆段名生成别名
    S1.8 → S1-8, S1_8, Segment1.8, segment 1.8
    """
    aliases: dict[str, list[str]] = {}
    for seg in sorted(set(segments)):
        seg = seg.strip()
        if not seg:
            continue
        variants = set()
        variants.add(seg.lower())
        # . ↔ - 替换
        if "." in seg:
            variants.add(seg.replace(".", "-"))
            variants.add(seg.replace(".", "-").lower())
            variants.add(seg.replace(".", "_"))
            variants.add(seg.replace(".", "_").lower())
        if "-" in seg:
            variants.add(seg.replace("-", "."))
            variants.add(seg.replace("-", ".").lower())
            variants.add(seg.replace("-", "_"))
        # 加 Segment 前缀
        variants.add(f"Segment{seg.lstrip('Ss')}")
        variants.add(f"segment {seg.lstrip('Ss')}")
        # 去掉大写 S 变小写
        if seg.startswith("S"):
            variants.add("s" + seg[1:])
        variants.discard(seg)
        aliases[seg] = sorted(variants)
    return aliases


def _generate_city_aliases(sites: list[str]) -> dict[str, list[str]]:
    """
    根据站点名生成中英文别名
    自动处理 '中国香港将军澳' → '香港', 'HK', 'Hong Kong', '将军澳' 等
    """
    # 手工维护的核心城市别名 (中文标准名 → 别名列表)
    core_aliases = {
        "中国香港": ["香港", "HK", "HongKong", "Hong Kong", "中国香港将军澳", "将军澳", "Tseung Kwan O", "Cape D' Augilar", "鹤咀", "香港鹤咀"],
        "中国香港将军澳": ["香港将军澳", "将军澳", "Tseung Kwan O", "TKO", "香港", "HK"],
        "香港鹤咀": ["鹤咀", "Cape D' Augilar", "Cape D Augilar", "香港", "HK"],
        "韩国釜山": ["釜山", "Busan", "busan", "Korea Busan"],
        "韩国巨济": ["巨济", "Keoje", "Geoje"],
        "新加坡东岸": ["East Coast", "新加坡", "Singapore", "SG"],
        "新加坡图阿斯": ["图阿斯", "Tuas", "新加坡", "Singapore"],
        "新加坡加东": ["加东", "Katong", "新加坡", "Singapore"],
        "中国上海崇明": ["崇明", "上海崇明", "上海", "Shanghai", "Chongming"],
        "中国上海南汇": ["南汇", "上海南汇", "Nanhui"],
        "中国上海临港": ["临港", "上海临港", "LinGang", "Lingang"],
        "中国台湾头城": ["头城", "台湾头城", "Toucheng", "台湾"],
        "中国台湾淡水": ["淡水", "台湾淡水", "Tanshui", "Tamsui", "台湾"],
        "中国青岛": ["青岛", "Qingdao"],
        "中国汕头": ["汕头", "Shantou"],
        "中国厦门": ["厦门", "Xiamen"],
        "中国福州长乐": ["福州", "长乐", "Fuzhou"],
        "日本北茨城": ["北茨城", "Kitaibaraki"],
        "日本千仓": ["千仓", "Chukura"],
        "日本丸山": ["丸山", "Maruyama", "Shinmaruyama", "新丸山"],
        "日本志摩": ["志摩", "Shima"],
        "日本新瓦山": ["新瓦山", "Shinmaruyama"],
        "马来西亚关丹": ["关丹", "Kuantan"],
        "马来西亚槟榔": ["槟榔", "Penang", "槟城"],
        "越南头顿": ["头顿", "Vung Tau", "VungTau"],
        "越南达南": ["达南", "岘港", "Danang", "Da Nang"],
        "泰国宋卡": ["宋卡", "Songkhla"],
        "泰国沙墩": ["沙墩", "Satun"],
        "柬埔寨西哈努克": ["西哈努克", "Sihanoukville"],
        "菲律宾八打雁": ["八打雁", "Batangas"],
        "埃及扎法拉纳": ["扎法拉纳", "Zafarana"],
        "吉布提": ["Djibouti"],
        "沙特吉达": ["吉达", "Jeddah", "Jidda"],
        "沙特盐步": ["盐步", "Yanbu"],
        "印度孟买": ["孟买", "Mumbai"],
        "巴基斯坦卡拉奇": ["卡拉奇", "Karachi"],
        "斯里兰卡马特勒": ["马特勒", "Matara"],
        "阿曼阿尔布斯坦": ["阿尔布斯坦", "Qalhat"],
        "阿曼哈特": ["哈特", "Qalhat"],
        "阿联酋富吉拉": ["富吉拉", "Fujairah"],
        "阿联酋卡尔巴": ["卡尔巴", "Kalba"],
        "美国希尔斯伯勒": ["希尔斯伯勒", "Hillsboro"],
        "美国俄勒冈": ["俄勒冈", "Oregon"],
        "法国土伦": ["土伦", "Toulon"],
        "法国马赛": ["马赛", "Marseille"],
        "意大利卡塔尼亚": ["卡塔尼亚", "Catania"],
        "土耳其马尔马里斯": ["马尔马里斯", "Marmaris"],
        "缅甸威双": ["威双"],
        "缅甸勃生": ["勃生", "Pathein"],
        "孟加拉库卡塔": ["库卡塔", "Kuakata"],
    }

    # 从 CSV 站点中提取实际站点，与核心别名合并
    aliases: dict[str, list[str]] = {}
    seen_sites = set()
    for site in sites:
        if pd.isna(site) or not str(site).strip():
            continue
        site = str(site).strip()
        seen_sites.add(site)

    for std_name, alias_list in core_aliases.items():
        variants = set(alias_list)
        variants.discard(std_name)
        aliases[std_name] = sorted(variants)

    # 对于 CSV 中出现但不在核心列表里的站点，也加入词典
    for site in sorted(seen_sites):
        if site not in aliases:
            aliases[site] = []

    return aliases


def build_all_dictionaries():
    """构建所有词典并保存"""
    os.makedirs(config.DICT_DIR, exist_ok=True)

    # 读取段落方向关系表
    df_seg = pd.read_csv(config.SEGMENT_CSV, encoding="utf-8")

    # ─── 海缆别名 ───
    system_names = df_seg["系统"].dropna().unique().tolist()
    cable_aliases = _generate_cable_aliases(system_names)
    with open(config.CABLE_DICT_PATH, "w", encoding="utf-8") as f:
        json.dump(cable_aliases, f, ensure_ascii=False, indent=2)
    print(f"[1/3] 海缆别名词典: {len(cable_aliases)} 个标准名 → {config.CABLE_DICT_PATH}")

    # ─── 缆段别名 ───
    segments = df_seg["段落标准"].dropna().unique().tolist()
    segment_aliases = _generate_segment_aliases(segments)
    with open(config.SEGMENT_DICT_PATH, "w", encoding="utf-8") as f:
        json.dump(segment_aliases, f, ensure_ascii=False, indent=2)
    print(f"[2/3] 缆段别名词典: {len(segment_aliases)} 个标准名 → {config.SEGMENT_DICT_PATH}")

    # ─── 城市别名 ───
    all_sites = []
    if "站点A" in df_seg.columns:
        all_sites.extend(df_seg["站点A"].dropna().tolist())
    if "站点B" in df_seg.columns:
        all_sites.extend(df_seg["站点B"].dropna().tolist())
    city_aliases = _generate_city_aliases(all_sites)
    with open(config.CITY_DICT_PATH, "w", encoding="utf-8") as f:
        json.dump(city_aliases, f, ensure_ascii=False, indent=2)
    print(f"[3/3] 城市别名词典: {len(city_aliases)} 个标准名 → {config.CITY_DICT_PATH}")

    print("\n✅ 所有词典构建完成")


if __name__ == "__main__":
    build_all_dictionaries()
