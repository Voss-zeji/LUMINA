# -*- coding: utf-8 -*-
"""
ensemble_utils_value.py

数值提取（value）任务的标准化处理、标准化结果统计与投票集成工具。

更新日期: 2025-12-10

主要公开函数
------------
- normalize_numeric_string(value):
    仅对“数值部分”做规范化（去括号、统一范围写法、± 取中心值、转换为统一字符串），
    不保留单位。适合用来判断“数值是否相同”，以及识别无意义数值（如 999999、-1 等）。

- normalize_numeric_with_unit(value):
    在 normalize_numeric_string 的基础上，同时提取单位信息，将“数值 + 单位”
    组合成标准化字符串。例如:
        "0.01 kgC m-2 yr-1" -> "0.01 c kgc m-2 yr-1"
        "0.01 m"           -> "0.01 m"
    对于完全没有单位的情况，返回与 normalize_numeric_string 一致的结果。

- ensemble_numeric_dataframe_all_data(df, ...):
    对原始模型输出按 (paper_index, question_index, item, normalized_value) 做标准化与聚合，
    返回每个“标准化数值（含单位）”的票数和置信度和，用于后续 ensemble。
    输出核心列:
        paper_index, question_index, item, normalized_value, count, confidence_sum, models

- ensemble_numeric_dataframe(df, ...):
    在 ensemble_numeric_dataframe_all_data 的汇总结果基础上，对每个
    (paper_index, question_index) 进行集成：
      * 默认按 (paper_index, question_index, item) 逐 item 投票集成；
      * 当多数有效数值的小数位数 ≥ 2 且超过 majority_decimal_ratio（默认 0.5）时，
        启用“以数值为核心”的集成模式：先按数值聚合票数，再用 ensemble_utils_meta 中
        的 location-specie 方法对对应的 item 文本进行归一化提取，输出代表性的 item 描述。
    在最终输出中不再包含 confidence_sum 列。

- summarize_value_options(df, ...):
    针对单个 (paper_index, item 等价) 列出所有候选数值及统计（含胜出者）。

- summarize_all_value_options(df, ...):
    对整表按 (paper_index, item 等价) 列出所有候选数值及统计。

- summarize_value_majorities_by_numeric(df, paper_index, ...):
    在同一篇文章内部，跨 item 按“标准化数值（含单位）”聚合，统计哪些模型、
    哪些 item 给出了该数值。

更新内容摘要
------------
- 新增 normalize_numeric_with_unit，用于在标准化数值的同时保留并规范单位，区分例如 0.01 kg 与 0.01 m。
- 新增 ensemble_numeric_dataframe_all_data，对原始结果做统一标准化与聚合，供 ensemble_numeric_dataframe 复用。
- 调整 ensemble_numeric_dataframe：
    * 新增参数 majority_decimal_ratio，用于判断“小数位数 ≥2 的数值是否为多数”，并在满足条件时启用“以数值为核心”的集成模式；
    * 在以数值为核心模式下，对同一 (paper_index, question_index, normalized_value) 下的 item 文本，
      调用 ensemble_utils_meta 中 location-specie 路径进行归一化提取代表性描述；
    * 输出的 ensemble_value 在原回答含有单位时保留单位信息，且不再包含 confidence_sum 列。
- 删除不再使用的辅助函数 ensemble_numeric_as_core_dataframe 与 ensemble_numeric_group，接口更精简。
"""



from __future__ import annotations
from typing import Optional, Dict, Any, List, Tuple, Iterable
import re
import pandas as pd

# 可选依赖：从 ensemble_utils_meta 复用 location-specie 集成功能
try:
    from .ensemble_utils_meta import _choose_location_specie, _default_translator
except Exception:  # pragma: no cover
    _choose_location_specie = None
    def _default_translator(s: str):
        return s, None


# =========================
# 基础与配置
# =========================

_INVALID_STRINGS = {
    "", "[]", "not provided", "not specified", "na", "n/a", "nan", "none"
}

# 无意义值（标准化为 str 后匹配）；可在函数里通过 meaningless_values 覆盖
_DEFAULT_MEANINGLESS_VALUES = {"999999", "-1", "na", "n/a", "nan", "none", ""}

# 常见置信度列名
_POSSIBLE_CONF_NAMES = ["confidence_lv", "confidence", "conf", "score", "prob", "probability", "conf_level"]


def _is_invalid_str(s: str) -> bool:
    return (s is None) or (str(s).strip().lower() in _INVALID_STRINGS)


def _safe_sort(df: pd.DataFrame, by: List[str], ascending: List[bool]) -> pd.DataFrame:
    """安全排序：只对存在的列排序，缺列忽略；若全部缺列则直接返回。"""
    present = [col for col in by if col in df.columns]
    if not present:
        return df.reset_index(drop=True)
    asc = [ascending[by.index(col)] for col in present]
    return df.sort_values(present, ascending=asc).reset_index(drop=True)


def _ensure_confidence_column(df: pd.DataFrame, conf_col: Optional[str] = None, default_value: float = 1.0) -> pd.DataFrame:
    """
    确保 df 存在 'confidence_lv' 列：
    - 若 conf_col 指定且存在，重命名为 'confidence_lv'
    - 否则自动在常见别名中识别并重命名
    - 若都没有，则创建 'confidence_lv' 并填 default_value（等权）
    """
    cols_lower = {c.lower(): c for c in df.columns}
    if conf_col:
        if conf_col in df.columns:
            if conf_col != "confidence_lv":
                df = df.rename(columns={conf_col: "confidence_lv"})
            return df
        if conf_col.lower() in cols_lower:
            real = cols_lower[conf_col.lower()]
            if real != "confidence_lv":
                df = df.rename(columns={real: "confidence_lv"})
            return df
        # 指定但没找到 -> 兜底创建
        df = df.copy()
        df["confidence_lv"] = float(default_value)
        return df

    # 自动识别
    for name in _POSSIBLE_CONF_NAMES:
        if name in df.columns:
            if name != "confidence_lv":
                df = df.rename(columns={name: "confidence_lv"})
            return df
        if name.lower() in cols_lower:
            real = cols_lower[name.lower()]
            if real != "confidence_lv":
                df = df.rename(columns={real: "confidence_lv"})
            return df

    # 都没有 -> 创建
    df = df.copy()
    df["confidence_lv"] = float(default_value)
    return df


# =========================
# Item 规整与等价
# =========================

def normalize_item_name(item: str, theme: Optional[str] = None) -> str:
    """与旧版一致的基础规整：下划线->空格，合并多空格；Wildfire 下去掉末尾 _None/_Mixed。"""
    if item is None:
        return ""
    s = str(item).strip()
    if _is_invalid_str(s):
        return ""
    s0 = s.replace("_", " ")
    s0 = re.sub(r"\s+", " ", s0).strip()
    if theme and theme.lower().startswith("wild"):
        s0 = re.sub(r"\s*_(none|mixed)\s*$", "", s0, flags=re.IGNORECASE)
    return s0


def canonicalize_item_suffixes(text: str) -> str:
    """去掉末尾 None/Mixed 后缀（空格/下划线写法，大小写不敏感），返回更通用的“基项”名称。"""
    if text is None:
        return ""
    s = str(text).strip()
    if _is_invalid_str(s):
        return ""
    s = s.replace("_", " ")
    s = re.sub(r"\s+", " ", s).strip()
    s = re.sub(r"\s+(none|mixed)\s*$", "", s, flags=re.IGNORECASE)
    return s.strip()


def _singularize_token(tok: str) -> str:
    """保守的简单单复数规整：去掉结尾单个 s（长度>3 且不以 ss 结尾）。"""
    t = tok
    if len(t) > 3 and t.endswith('s') and not t.endswith('ss'):
        t = t[:-1]
    return t


_SPELLING_EQUIV = {
    # flaming 家族
    "flaming": "flaming",
    "flamming": "flaming",
    # smoldering 家族
    "smouldering": "smoldering",
    "smoldering": "smoldering",
}


def item_equivalence_key(text: str, merge_none_mixed: bool = True) -> str:
    """
    生成 item 的“等价键”，用于将（空格/大小写/简单单复数/拼写）差异折叠在一起，
    但保持 flaming 与 smoldering 两大类互不等价。
    merge_none_mixed=True 时去掉末尾 None/Mixed 后缀（不影响 flaming vs smoldering 区别）。
    """
    if text is None:
        return ""
    s = str(text).strip()
    if _is_invalid_str(s):
        return ""
    # 统一
    s = s.replace("_", " ")
    s = re.sub(r"\s+", " ", s).strip().lower()
    if merge_none_mixed:
        s = re.sub(r"\s+(none|mixed)\s*$", "", s, flags=re.IGNORECASE)

    tokens = []
    for tok in s.split(" "):
        tok = _SPELLING_EQUIV.get(tok, tok)
        tok = _singularize_token(tok)
        tokens.append(tok)

    # 忽略空格差异
    key_no_space = "".join(tokens)
    return key_no_space


# =========================
# 数值解析与判定
# =========================

# ---- Ensemble: Numeric Items — Pure numeric normalization ----
# Strips units, resolves ± to center value, normalizes ranges to "min-max"
# Examples:
#   "12.3 ± 0.4" → "12.3"
#   "10-20" / "10 to 20" / "10~20"  → "10-20"
#   "0.01 kgC m-2 yr-1" → "0.01" (unit stripped)
# Returns None for unparseable values
def normalize_numeric_string(value: Any) -> Optional[str]:
    """
    将任意数值字符串解析为规范表达：
    - 单值：返回小数/整数（字符串）
    - ± 表达：保留中心值（'12.3 ± 0.4' -> '12.3'）
    - 范围：'x–y'/'x-y'/'x to y'/'x ~ y' -> 'min-max'
    - 自动去单位（仅保留数值/区间）；解析失败 -> None
    """
    if value is None:
        return None
    s = str(value).strip()
    if _is_invalid_str(s):
        return None

    # 去括号
    if s.startswith("(") and s.endswith(")"):
        s = s[1:-1].strip()

    # 标准化连字符与范围词
    s = s.replace("–", "-").replace("—", "-").replace("−", "-")
    s = re.sub(r"\bto\b", "-", s, flags=re.IGNORECASE)
    s = re.sub(r"~", "-", s)

    # ± 表达：提取中心值
    m_pm = re.search(r"^\s*([+-]?\d[\d,]*(?:\.\d+)?)\s*±\s*([+-]?\d[\d,]*(?:\.\d+)?)\s*$", s)
    if m_pm:
        center = m_pm.group(1).replace(",", "")
        return str(float(center)) if re.search(r"\.", center) else str(int(float(center)))

    # 范围（两端数字）
    nums = re.findall(r"[+-]?\d[\d,]*(?:\.\d+)?", s)
    nums_clean = [n.replace(",", "") for n in nums]
    if "-" in s or " to " in s or "~" in s:
        if len(nums_clean) >= 2:
            a, b = float(nums_clean[0]), float(nums_clean[1])
            lo, hi = (a, b) if a <= b else (b, a)
            def _fmt(x: float) -> str:
                xi = int(x)
                return str(xi) if abs(x - xi) < 1e-12 else str(x)
            return f"{_fmt(lo)}-{_fmt(hi)}"

    # 单值：取首个数字
    if nums_clean:
        v = nums_clean[0]
        return str(float(v)) if re.search(r"\.", v) else str(int(float(v)))

    return None

# ---- Ensemble: Numeric Items — Numeric + unit normalization ----
# Extends normalize_numeric_string by also extracting and normalizing unit tokens
# Example: "0.01 kgC m-2 yr-1" → "0.01 c kgc m-2 yr-1"
# Units are lowercased, deduplicated, sorted → ensures "kg m-2" == "m-2 kg"
def normalize_numeric_with_unit(value: Any) -> Optional[str]:
    """
    在 normalize_numeric_string 的基础上，同时考虑单位信息，返回“数值 + 单位”的标准化字符串。

    规则
    ----
    1. 先用 normalize_numeric_string(value) 得到数值部分 base；
       若 base 为 None，则返回 None。
    2. 再在原始字符串中提取单位 token：
       - 匹配以字母或常见单位符号开头的片段: [A-Za-zμ%°/][A-Za-z0-9μ%°/^-]*
       - 例如: "kg", "mg", "m-2", "yr-1", "%", "c" 等
    3. 若没有任何单位 token，则返回 base（与原逻辑完全一致）。
    4. 否则，将单位 token 转为小写、去重并排序后，用空格连接，拼在 base 后面：
       返回 f"{base} {unit_str}"。
    """
    base = normalize_numeric_string(value)
    if base is None:
        return None

    s = str(value)

    # 提取单位 token
    unit_tokens = re.findall(r"[A-Za-zμ%°/][A-Za-z0-9μ%°/\^\-]*", s)
    if not unit_tokens:
        # 无单位 -> 保持与原逻辑一致
        return base

    # 小写 + 去重 + 排序，保证不同书写顺序得到相同的规范表达
    unit_tokens_norm = sorted({tok.lower() for tok in unit_tokens})
    unit_str = " ".join(unit_tokens_norm)

    return f"{base} {unit_str}"


def _is_meaningless_value(v_norm: Optional[str], meaningless_values: Iterable[str]) -> bool:
    if v_norm is None:
        return True
    return str(v_norm).strip().lower() in {str(x).strip().lower() for x in meaningless_values}


# =========================
# 票数统计（数值）
# =========================

def _tally_votes_numeric(df_group: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
    """对一个 (paper_index, question_index, item 等价) 分组统计各标准化数值(含单位)的票数/置信度和/来源。"""
    tally: Dict[str, Dict[str, Any]] = {}
    for _, r in df_group.iterrows():
        # 使用“数值 + 单位”的标准化形式
        v_norm = normalize_numeric_with_unit(r.get("value"))
        # 无意义标记仍由上层的 _meaningless 控制（基于纯数值）
        if v_norm in (None, "") or r.get("_meaningless", False):
            continue

        conf = r.get("confidence_lv", 1.0)
        try:
            conf_f = float(conf) if conf is not None else 1.0
        except Exception:
            conf_f = 1.0
        model = str(r.get("model", "")).strip()
        raw = str(r.get("value", ""))

        g = tally.setdefault(v_norm, {"count": 0, "confidence_sum": 0.0, "models": set(), "raws": set()})
        g["count"] += 1
        g["confidence_sum"] += conf_f
        if model:
            g["models"].add(model)
        if raw:
            g["raws"].add(raw)
    return tally


# =========================
# A) 集成（整表/单组）
# =========================


def ensemble_numeric_dataframe_all_data(
    df: pd.DataFrame,
    theme: Optional[str] = None,
    prefer: str = "votes",
    second: str = "confidence",
    drop_ilegal: bool = True,
    conf_col: Optional[str] = None,
    merge_none_mixed: bool = True,
    use_item_equivalence: bool = True,
    meaningless_values: Optional[Iterable[str]] = None,
    majority_decimal_ratio: float = 0.5,
) -> pd.DataFrame:
    """
    对原始数值表进行标准化，返回 (paper_index, question_index, item, normalized_value) 粒度的汇总结果。

    输入参数
    --------
    与 ensemble_numeric_dataframe 保持一致：
    - df : 原始结果 DataFrame（至少包含 paper_index, item, value, model 列）
    - theme, prefer, second, drop_ilegal, conf_col, merge_none_mixed, use_item_equivalence, meaningless_values
      其中 prefer/second 在本函数中仅为对齐接口，不影响标准化与聚合逻辑。

    输出列
    ------
    - paper_index
    - question_index
    - item              : 代表性 item 名称（同一等价类内出现次数最多的 item_norm）
    - normalized_value  : 由 normalize_numeric_with_unit 得到的“数值 + 单位”的规范表达
    - count             : 该标准化值在该 (paper, question, item 等价) 内的票数
    - confidence_sum    : 上述票对应的置信度和
    - models            : 给出该值的模型集合，逗号分隔
    """
    required = {"paper_index", "item", "value", "model"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Input DataFrame is missing required columns: {missing}")

    use_df = df.copy()
    if drop_ilegal and "item" in use_df.columns:
        use_df = use_df[use_df["item"] != "ilegal"]

    # 若没有 question_index，则补一列常数 1
    if "question_index" not in use_df.columns:
        use_df["question_index"] = 1

    # 置信度兜底
    use_df = _ensure_confidence_column(use_df, conf_col=conf_col, default_value=1.0)

    # 归一化 item（基础）+ 等价键
    theme_l = (theme or "").lower()
    use_df["item_norm"] = use_df["item"].map(lambda x: normalize_item_name(x, theme=theme_l))

    if use_item_equivalence:
        use_df["item_equiv"] = use_df["item_norm"].map(
            lambda x: item_equivalence_key(x, merge_none_mixed=merge_none_mixed)
        )
    else:
        use_df["item_equiv"] = (
            use_df["item_norm"].map(canonicalize_item_suffixes) if merge_none_mixed else use_df["item_norm"]
        )

    # 代表性 item 名称（每个等价类的输出展示名）
    rep_names = (
        use_df
        .groupby(["paper_index", "question_index", "item_equiv"])["item_norm"]
        .agg(lambda s: s.mode().iloc[0] if not s.mode().empty else s.iloc[0])
        .reset_index()
    )
    rep_map = {
        (r["paper_index"], r["question_index"], r["item_equiv"]): r["item_norm"]
        for _, r in rep_names.iterrows()
    }

    # 无意义值标记：基于“纯数值”
    meaningless_values = set(meaningless_values) if meaningless_values is not None else _DEFAULT_MEANINGLESS_VALUES
    use_df["_v_numeric"] = use_df["value"].map(normalize_numeric_string)
    use_df["_meaningless"] = use_df["_v_numeric"].map(lambda v: _is_meaningless_value(v, meaningless_values))

    # 数值 + 单位 的标准化
    use_df["normalized_value"] = use_df["value"].map(normalize_numeric_with_unit)

    # 过滤无意义或无法解析的值
    mask_valid = (~use_df["_meaningless"]) & use_df["normalized_value"].notna() & (use_df["normalized_value"] != "")
    use_df = use_df[mask_valid].copy()
    if use_df.empty:
        return pd.DataFrame(
            columns=[
                "paper_index", "question_index", "item",
                "normalized_value", "count", "confidence_sum", "models",
            ]
        )

    # 按 (paper, question, item_equiv, normalized_value) 聚合
    grouped = use_df.groupby(
        ["paper_index", "question_index", "item_equiv", "normalized_value"],
        dropna=False,
        sort=True,
    )

    rows: List[Dict[str, Any]] = []
    for (pid, qid, iteq, vnorm), g in grouped:
        # 聚合票数与置信度
        count = len(g)
        conf_sum = float(g["confidence_lv"].fillna(1.0).astype(float).sum())
        models = sorted({str(m).strip() for m in g["model"].tolist() if pd.notna(m)})

        item_display = rep_map.get((pid, qid, iteq), g["item_norm"].iloc[0])

        rows.append(
            {
                "paper_index": pid,
                "question_index": qid,
                "item": item_display,
                "normalized_value": vnorm,
                "count": count,
                "confidence_sum": conf_sum,
                "models": ",".join(models),
            }
        )

    out = pd.DataFrame(rows)
    if not out.empty:
        out = _safe_sort(
            out,
            ["paper_index", "question_index", "item", "count", "confidence_sum", "normalized_value"],
            [True, True, True, False, False, True],
        )

    return out

# ---- Ensemble: Numeric Items — Main ensemble function ----
# Two modes:
#   1. Default mode (per-item voting):
#      Group by (paper, question, item) → vote → winner by count+confidence
#   2. Decimal-majority mode (cross-item by numeric value):
#      When ≥50% of values have ≥2 decimal places, group by (paper, question, normalized_value)
#      across items, then use location-specie logic to extract representative item description
# Both modes output: paper_index, question_index, item, ensemble_value, vote_count, models
def ensemble_numeric_dataframe(
    df: pd.DataFrame,
    theme: Optional[str] = None,
    prefer: str = "votes",
    second: str = "confidence",
    drop_ilegal: bool = True,
    conf_col: Optional[str] = None,
    merge_none_mixed: bool = True,
    use_item_equivalence: bool = True,
    meaningless_values: Optional[Iterable[str]] = None,
    majority_decimal_ratio: float = 0.5,
) -> pd.DataFrame:
    """
    在“标准化 + 聚合”的基础上，对每个 (paper_index, question_index, item) 做数值集成。

    步骤
    ----
    1. 调用 ensemble_numeric_dataframe_all_data 得到按
       (paper_index, question_index, item, normalized_value) 聚合后的计数和置信度。
    2. 在每个 (paper_index, question_index, item) 组内，根据 prefer / second 选择赢家：
       - prefer == "votes" 时：优先比较 count，再比较 confidence_sum
       - prefer == "confidence" 时：优先比较 confidence_sum，再比较 count
       - 最后使用 normalized_value 作为稳定的字符串 tie-break。
    3. 输出每个 (paper_index, question_index, item) 的 ensemble_value 及其统计信息。
    """
    # 判断全表中“有效数值”中，小数位数 >= 2 的比例是否超过 majority_decimal_ratio
    v_series = df.get("value")
    decimals_majority = False
    if v_series is not None:
        v_norm_all = v_series.map(normalize_numeric_string)
        meaningless_set = set(meaningless_values) if meaningless_values is not None else _DEFAULT_MEANINGLESS_VALUES
        def _is_valid_num(x: Optional[str]) -> bool:
            if x is None:
                return False
            return not _is_meaningless_value(x, meaningless_set)
        mask_valid = v_norm_all.map(_is_valid_num)
        v_valid = v_norm_all[mask_valid]
        def _has_two_decimals(txt: Optional[str]) -> bool:
            if txt is None:
                return False
            s = str(txt)
            if "." not in s:
                return False
            frac = s.split(".", 1)[1]
            return len(frac) >= 2
        if len(v_valid) > 0:
            cnt_two = sum(_has_two_decimals(x) for x in v_valid)
            ratio_two = cnt_two / float(len(v_valid))
            decimals_majority = ratio_two >= float(majority_decimal_ratio)

    base = ensemble_numeric_dataframe_all_data(
        df,
        theme=theme,
        prefer=prefer,
        second=second,
        drop_ilegal=drop_ilegal,
        conf_col=conf_col,
        merge_none_mixed=merge_none_mixed,
        use_item_equivalence=use_item_equivalence,
        meaningless_values=meaningless_values,
        majority_decimal_ratio=majority_decimal_ratio,
    )


    if base.empty:
        return pd.DataFrame(columns=[
            "paper_index", "question_index", "item",
            "ensemble_value", "vote_count", "models"
        ])

    prefer_l = (prefer or "votes").lower()
    second_l = (second or "confidence").lower()

    rows: List[Dict[str, Any]] = []

    if not decimals_majority:
        # 默认模式：仍按 (paper_index, question_index, item) 分组集成
        for (pid, qid, item), g in base.groupby(["paper_index", "question_index", "item"], dropna=False, sort=True):
            g = g.copy()

            def _score(row):
                primary = row["count"] if prefer_l == "votes" else row.get("confidence_sum", 0.0)
                secondary = row.get("confidence_sum", 0.0) if second_l == "confidence" else row["count"]
                return (primary, secondary, str(row.get("normalized_value", "")))

            winner_idx = g.apply(_score, axis=1).idxmax()
            best = g.loc[winner_idx]

            rows.append({
                "paper_index": pid,
                "question_index": qid,
                "item": item,
                "ensemble_value": best["normalized_value"],
                "vote_count": int(best["count"]),
                "models": str(best.get("models", "")),
            })
    else:
        # 以“数值”为核心的模式：
        # 先按 (paper_index, question_index, normalized_value) 聚合票数，
        # 再对这些数值对应的 item 文本调用 location-specie 逻辑提取代表性描述。
        for (pid, qid), g_pq in base.groupby(["paper_index", "question_index"], dropna=False, sort=True):
            # 对每个 normalized_value 聚合 count 和 models，并收集 item 文本
            agg_rows: List[Dict[str, Any]] = []
            for vnorm, g_v in g_pq.groupby("normalized_value", dropna=False, sort=True):
                if pd.isna(vnorm) or vnorm in ("", None):
                    continue
                total_count = int(g_v["count"].sum())
                all_models = set()
                items_txt: List[str] = []
                for _, r in g_v.iterrows():
                    # 模型合并
                    m_str = r.get("models", "")
                    if isinstance(m_str, str) and m_str:
                        for m in m_str.split(","):
                            m = m.strip()
                            if m:
                                all_models.add(m)
                    # item 文本
                    it = r.get("item", "")
                    if pd.notna(it) and str(it).strip():
                        items_txt.append(str(it).strip())

                # 通过 ensemble_utils_meta 的 location-specie 路径归一化 item 文本
                if items_txt and _choose_location_specie is not None:
                    try:
                        loc_ens, _support = _choose_location_specie(items_txt, translator=_default_translator)
                        item_repr = loc_ens or items_txt[0]
                    except Exception:
                        item_repr = items_txt[0]
                else:
                    item_repr = items_txt[0] if items_txt else ""

                agg_rows.append({
                    "normalized_value": vnorm,
                    "item": item_repr,
                    "vote_count": total_count,
                    "models": ",".join(sorted(all_models)),
                })

            if not agg_rows:
                continue

            g_agg = pd.DataFrame(agg_rows)

            # 这里保留所有候选数值的行，便于后续分析；如需仅保留赢家，可在此处筛选 max-score 行
            for _, r in g_agg.iterrows():
                rows.append({
                    "paper_index": pid,
                    "question_index": qid,
                    "item": r["item"],
                    "ensemble_value": r["normalized_value"],
                    "vote_count": int(r["vote_count"]),
                    "models": str(r.get("models", "")),
                })

    out = pd.DataFrame(rows)
    if not out.empty:
        out = _safe_sort(
            out,
            ["paper_index", "question_index", "item"],
            [True, True, True],
        )
    return out





# =========================
# D) 全自动：逐 paper 的“以数值为核心”的集成结果
# =========================

# =========================
# B) 候选汇总（单组/整表）
# =========================

def summarize_value_options(
    df: pd.DataFrame,
    paper_index: int,
    item: str,
    theme: Optional[str] = None,
    prefer: str = "votes",
    second: str = "confidence",
    conf_col: Optional[str] = None,
    merge_none_mixed: bool = True,
    use_item_equivalence: bool = True,
    meaningless_values: Optional[Iterable[str]] = None,
) -> pd.DataFrame:
    """
    针对单个 (paper_index, item 等价) 列出所有候选数值：
    - 同 ensemble 的等价规则与无意义值过滤保持一致
    - 输出 count/confidence_sum/占比/来源样例/排名/赢家等
    """
    required = {"paper_index", "item", "value", "model"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Input DataFrame is missing required columns: {missing}")

    df = _ensure_confidence_column(df, conf_col=conf_col, default_value=1.0)
    theme_l = (theme or "").lower()
    it_norm_target = normalize_item_name(item, theme=theme_l)

    use_df = df.copy()
    use_df["item_norm"] = use_df["item"].map(lambda x: normalize_item_name(x, theme=theme_l))

    if use_item_equivalence:
        use_df["item_equiv"] = use_df["item_norm"].map(lambda x: item_equivalence_key(x, merge_none_mixed=merge_none_mixed))
        target_equiv = item_equivalence_key(it_norm_target, merge_none_mixed=merge_none_mixed)
    else:
        use_df["item_equiv"] = use_df["item_norm"].map(canonicalize_item_suffixes) if merge_none_mixed else use_df["item_norm"]
        target_equiv = canonicalize_item_suffixes(it_norm_target) if merge_none_mixed else it_norm_target

    sub = use_df[(use_df["paper_index"] == paper_index) & (use_df["item_equiv"] == target_equiv)].copy()
    if sub.empty:
        return pd.DataFrame(columns=[
            "paper_index","item","value_norm","count","confidence_sum","vote_share","conf_share",
            "models","raw_samples","rank_by_votes","rank_by_conf",
            "is_winner_by_votes","is_winner_by_conf","is_winner"
        ])

    # 标记无意义值（基于纯数值），并使用“数值 + 单位”作为 value_norm
    meaningless_values = set(meaningless_values) if meaningless_values is not None else _DEFAULT_MEANINGLESS_VALUES
    sub["_v_numeric"] = sub["value"].map(normalize_numeric_string)
    sub["_meaningless"] = sub["_v_numeric"].map(lambda v: _is_meaningless_value(v, meaningless_values))
    sub["value_norm"] = sub["value"].map(normalize_numeric_with_unit)

    tally: Dict[str, Dict[str, Any]] = {}
    total_votes = 0
    total_conf = 0.0
    for _, r in sub.iterrows():
        v_norm = r.get("value_norm")
        if v_norm in (None, "") or r.get("_meaningless", False):
            continue
        conf = r.get("confidence_lv", 1.0)
        try:
            conf_f = float(conf) if conf is not None else 1.0
        except Exception:
            conf_f = 1.0
        model = str(r.get("model",""))
        raw = str(r.get("value",""))

        g = tally.setdefault(v_norm, {"count":0,"confidence_sum":0.0,"models":set(),"raws":set()})
        g["count"] += 1
        g["confidence_sum"] += conf_f
        total_votes += 1
        total_conf += conf_f
        if model:
            g["models"].add(model)
        if raw:
            g["raws"].add(raw)

    if not tally:
        return pd.DataFrame(columns=[
            "paper_index","item","value_norm","count","confidence_sum","vote_share","conf_share",
            "models","raw_samples","rank_by_votes","rank_by_conf",
            "is_winner_by_votes","is_winner_by_conf","is_winner"
        ])

    rows = []
    total_votes = total_votes or 1
    total_conf = total_conf or 1.0
    # 展示名：取该票箱内出现次数最多的 item_norm
    item_display = sub["item_norm"].mode().iloc[0] if not sub["item_norm"].mode().empty else it_norm_target

    for val, g in tally.items():
        rows.append({
            "paper_index": paper_index,
            "item": item_display,
            "value_norm": val,
            "count": g["count"],
            "confidence_sum": g["confidence_sum"],
            "vote_share": g["count"] / total_votes,
            "conf_share": g["confidence_sum"] / total_conf,
            "models": ",".join(sorted(g["models"])),
            "raw_samples": ", ".join(sorted(g["raws"])[:5]),
        })
    res = pd.DataFrame(rows)

    for col, default in (("count", 0), ("confidence_sum", 0.0), ("value_norm", "")):
        if col not in res.columns:
            res[col] = default

    res = _safe_sort(res, ["count","confidence_sum","value_norm"], [False, False, True])
    res["rank_by_votes"] = range(1, len(res) + 1)
    res = _safe_sort(res, ["confidence_sum","count","value_norm"], [False, False, True])
    res["rank_by_conf"] = range(1, len(res) + 1)
    res = _safe_sort(res, ["count","confidence_sum","value_norm"], [False, False, True])

    max_count = res["count"].max() if "count" in res else 0
    max_conf  = res["confidence_sum"].max() if "confidence_sum" in res else 0.0
    res["is_winner_by_votes"] = res["count"].eq(max_count) if "count" in res else False
    res["is_winner_by_conf"]  = res["confidence_sum"].eq(max_conf) if "confidence_sum" in res else False

    prefer_l = (prefer or "votes").lower()
    second_l = (second or "confidence").lower()
    def _score(row):
        primary   = row["count"] if prefer_l == "votes" else row["confidence_sum"]
        secondary = row["confidence_sum"] if second_l == "confidence" else row["count"]
        return (primary, secondary, str(row.get("value_norm","")))
    winner_idx = res.apply(_score, axis=1).idxmax()
    res["is_winner"] = False
    res.loc[winner_idx, "is_winner"] = True

    return res[[
        "paper_index","item","value_norm","count","confidence_sum","vote_share","conf_share",
        "models","raw_samples","rank_by_votes","rank_by_conf",
        "is_winner_by_votes","is_winner_by_conf","is_winner"
    ]]


def summarize_all_value_options(
    df: pd.DataFrame,
    group_cols: Tuple[str, ...] = ("paper_index","item"),
    theme: Optional[str] = None,
    prefer: str = "votes",
    second: str = "confidence",
    drop_ilegal: bool = True,
    topk: Optional[int] = None,
    conf_col: Optional[str] = None,
    merge_none_mixed: bool = True,
    use_item_equivalence: bool = True,
    meaningless_values: Optional[Iterable[str]] = None,
) -> pd.DataFrame:
    """
    对整表进行分组统计，列出每个 (paper_index, item 等价) 的所有候选数值及其统计；
    规则与 ensemble 完全一致；可选 topk 每组仅保留前 k 个候选。
    """
    required = {"paper_index","item","value","model"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Input DataFrame is missing required columns: {missing}")

    use_df = df.copy()
    if drop_ilegal and "item" in use_df.columns:
        use_df = use_df[use_df["item"] != "ilegal"]

    use_df = _ensure_confidence_column(use_df, conf_col=conf_col, default_value=1.0)

    theme_l = (theme or "").lower()
    use_df["item_norm"] = use_df["item"].map(lambda x: normalize_item_name(x, theme=theme_l))
    if use_item_equivalence:
        use_df["item_equiv"] = use_df["item_norm"].map(lambda x: item_equivalence_key(x, merge_none_mixed=merge_none_mixed))
    else:
        use_df["item_equiv"] = use_df["item_norm"].map(canonicalize_item_suffixes) if merge_none_mixed else use_df["item_norm"]

    parts: List[pd.DataFrame] = []
    for (pid, iteq), gdf in use_df.groupby(["paper_index","item_equiv"], dropna=False, sort=True):
        # 用一致的规则调用单组统计
        it_display = gdf["item_norm"].mode().iloc[0] if not gdf["item_norm"].mode().empty else gdf["item_norm"].iloc[0]
        summary = summarize_value_options(
            gdf, paper_index=pid, item=it_display, theme=theme, prefer=prefer, second=second,
            conf_col="confidence_lv", merge_none_mixed=merge_none_mixed, use_item_equivalence=use_item_equivalence,
            meaningless_values=meaningless_values
        )
        if topk is not None and not summary.empty:
            summary = _safe_sort(summary, ["count","confidence_sum","value_norm"], [False, False, True]).head(topk)
        parts.append(summary)

    if not parts:
        return pd.DataFrame(columns=[*group_cols,"value_norm","count","confidence_sum","vote_share","conf_share",
                                     "models","raw_samples","rank_by_votes","rank_by_conf",
                                     "is_winner_by_votes","is_winner_by_conf","is_winner"])

    out = pd.concat(parts, ignore_index=True)

    # 保证输出列顺序，且每组赢家优先显示
    by_cols = ["paper_index", "item", "__winner_order__", "count", "confidence_sum", "value_norm"]
    out["__winner_order__"] = (~out.get("is_winner", False)).astype(int)
    asc = [True, True, True, False, False, True]
    out = _safe_sort(out, by_cols, asc).drop(columns="__winner_order__", errors="ignore")
    return out


# =========================
# C) 新函数：按“数值为核心”的多数统计
# =========================

def summarize_value_majorities_by_numeric(
    df: pd.DataFrame,
    paper_index: int,
    theme: Optional[str] = None,
    conf_col: Optional[str] = None,
    merge_none_mixed: bool = True,
    use_item_equivalence: bool = True,
    meaningless_values: Optional[Iterable[str]] = None,
) -> pd.DataFrame:
    """
    在给定 paper_index 内，跨 item（按等价规则归并）地按“标准化数值”进行多数统计：
    - 对每个标准化数值 value_norm 汇总 vote_count、confidence_sum
    - 给出来源明细 source：'model -> item_display'（item_display 为该模型给出的 item 归一名）
    - 附加 columns：items（出现过的 item_display 集合），models（模型集合）

    用法契合你的例子：
    若 value_norm=1489 被 3 个模型给出，则 vote_count=3，并在 source 列指明 3 个模型各自的 item 判断。
    """
    required = {"paper_index","item","value","model"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Input DataFrame is missing required columns: {missing}")

    df = _ensure_confidence_column(df, conf_col=conf_col, default_value=1.0)
    theme_l = (theme or "").lower()

    use_df = df.copy()
    use_df = use_df[use_df["paper_index"] == paper_index].copy()
    if use_df.empty:
        return pd.DataFrame(columns=[
            "paper_index","value_norm","vote_count","confidence_sum","models","items","source"
        ])

    # item 归一 + 等价键
    use_df["item_norm"] = use_df["item"].map(lambda x: normalize_item_name(x, theme=theme_l))
    if use_item_equivalence:
        use_df["item_equiv"] = use_df["item_norm"].map(lambda x: item_equivalence_key(x, merge_none_mixed=merge_none_mixed))
    else:
        use_df["item_equiv"] = use_df["item_norm"].map(canonicalize_item_suffixes) if merge_none_mixed else use_df["item_norm"]

    # 每条记录的标准化值与无意义值标记：无意义判断基于纯数值，value_norm 使用“数值 + 单位”
    meaningless_values = set(meaningless_values) if meaningless_values is not None else _DEFAULT_MEANINGLESS_VALUES
    use_df["_v_numeric"] = use_df["value"].map(normalize_numeric_string)
    use_df["_meaningless"] = use_df["_v_numeric"].map(lambda v: _is_meaningless_value(v, meaningless_values))
    use_df["value_norm"] = use_df["value"].map(normalize_numeric_with_unit)

    # 过滤掉无意义/空值
    use_df = use_df[~use_df["_meaningless"]].copy()
    if use_df.empty:
        return pd.DataFrame(columns=[
            "paper_index","value_norm","vote_count","confidence_sum","models","items","source"
        ])

    rows = []
    for v, g in use_df.groupby("value_norm", dropna=False, sort=True):
        models = sorted(set(str(m).strip() for m in g["model"].tolist() if pd.notna(m)))
        items = sorted(set(str(it).strip() for it in g["item_norm"].tolist() if pd.notna(it)))
        # source 明细
        pairs = []
        conf_sum = 0.0
        for _, r in g.iterrows():
            model = str(r.get("model","")).strip()
            it_display = str(r.get("item_norm",""))
            conf = r.get("confidence_lv", 1.0)
            try:
                conf_f = float(conf) if conf is not None else 1.0
            except Exception:
                conf_f = 1.0
            conf_sum += conf_f
            if model:
                pairs.append(f"{model} -> {it_display}")
        rows.append({
            "paper_index": paper_index,
            "value_norm": v,
            "vote_count": len(models),
            "confidence_sum": conf_sum,
            "models": ",".join(models),
            "items": ",".join(items),
            "source": "; ".join(pairs),
        })

    out = pd.DataFrame(rows)
    # 排序：票数 desc，置信度 desc，数值升序
    def _val_key(s: str):
        try:
            return float(str(s).split("-")[0])
        except Exception:
            return s
    out = out.sort_values(
        by=["vote_count","confidence_sum","value_norm"],
        ascending=[False, False, True],
        key=lambda col: col.map(_val_key) if col.name=="value_norm" else col
    ).reset_index(drop=True)

    return out[["paper_index","value_norm","vote_count","confidence_sum","models","items","source"]]
