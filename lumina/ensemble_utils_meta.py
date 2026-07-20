# -*- coding: utf-8 -*-
"""
更新日志 (2025-12-13)
- 新增“空值占多数则选择空值”的统一规则：对经纬度、Study period 以及其他条目，在标准化后若空值（""）计数超过该组总数的一半，则最终 ensemble_value 直接输出为空（NaN），并将 support 设为空值支持数。
- 检查并修正：Study period 过去会忽略无法解析/空值而导致在空值绝对多数时仍输出非空范围；现已与经纬度逻辑一致，优先输出空值。

更新日志 (2025-12-13)
- 经纬度（Longitude/Latitude）集成：在 ensemble_dataframe 中按 precision_digits 先对所有候选 normalized_value 进行统一四舍五入归并，再进行投票选择；support 计数与归并后的结果一致（例如 119.6279/119.628/119.63 在 precision_digits=2 时统一归并为 119.63 再统计）。
- 其余未提及部分保持不变。

ensemble_utils_meta.py

对“同一 paper_index + question_index + item 下、不同 model 的 value”进行集成（ensemble）。
支持三类核心条目：
A) Study Location / Specie（物种/地点短语投票）
B) Study Period（起止时间范围规范化与加权投票：day > month > year）
C) Longitude / Latitude（经纬度区间格式化与稳健聚合）

公开函数：
    ensemble_dataframe(df, translator=None, ...)
    ensemble_dataframe_all_data(df, translator=None, ...)

注意：
- ensemble_dataframe_all_data：对每个 (paper_index, question_index, item) 的原始取值进行标准化，
  返回 normalized_value + 频数，用于后续分析/调试。
- ensemble_dataframe：在 ensemble_dataframe_all_data 的基础上，根据 item 类型（Study Period /
  Longitude / Latitude / 其他）应用对应的 ensemble 逻辑，得到最终 ensemble_value。

最近更新记录：
2025-12-11 更新（协调 Study Period 与经纬度范围处理）：
- Study Period：确保混合文本/日期表达（如 'December 10, 2011 to January 20, 2012'）在解析后统一为
  YYYY-MM-DD 起止日期再参与加权投票。
- Longitude / Latitude：经纬度答案统一按区间解析与输出（若为范围，则输出 low–high；若为单点，则
  视作 [v, v]），ensemble 过程中也在区间空间中进行投票和支持度统计。

2025-12-11 更新（Study Period 月范围表达补充）：
- 增强对类似 'May-December 2010' / 'May to December 2010' 这类“月份-月份 + 年份”表达的解析，
  在集成前统一标准化为起止月份区间（例如 [2010-05, 2010-12]，即 2010-05 to 2010-12），
  并作为 month 精度的 Study Period 参与加权投票。
"""

from __future__ import annotations

import re
import math
from collections import Counter, defaultdict
from datetime import date
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import pandas as pd

# 可选：langid，用于中英文判断
_HAS_LANGID = True
try:  # pragma: no cover
    import langid
except Exception:  # pragma: no cover
    _HAS_LANGID = False


# =====================
# 通用正则与常量
# =====================

_WHITESPACE_RE = re.compile(r"\s+")
_PARENS_RE = re.compile(r"\([^)]*\)")
_COMMA_SPLIT_RE = re.compile(r"[,，]+")
_KEEP_CHARS_RE = re.compile(r"[^a-z0-9\s\-]")
_EN_DASHES_RE = re.compile(r"[–—−]+")
_CHN_CHAR_RE = re.compile(r"[\u4e00-\u9fff]")

_STOPWORD_SET = {
    "the",
    "a",
    "an",
    "of",
    "and",
    "in",
    "at",
    "on",
    "for",
    "to",
    "with",
    "by",
    "from",
    "near",
    "area",
    "region",
    "province",
    "city",
    "district",
    "county",
    "prefecture",
    "municipality",
    "town",
    "village",
    "river",
    "estuary",
    "wetland",
    "lake",
    "bay",
    "island",
    "islands",
    "sea",
    "ocean",
    "gulf",
    "coast",
    "valley",
    "plain",
    "plateau",
}

# 月份映射
MONTH_MAP: Dict[str, int] = {
    "jan": 1,
    "january": 1,
    "feb": 2,
    "february": 2,
    "mar": 3,
    "march": 3,
    "apr": 4,
    "april": 4,
    "may": 5,
    "jun": 6,
    "june": 6,
    "jul": 7,
    "july": 7,
    "aug": 8,
    "august": 8,
    "sep": 9,
    "sept": 9,
    "september": 9,
    "oct": 10,
    "october": 10,
    "nov": 11,
    "november": 11,
    "dec": 12,
    "december": 12,
}

# Study Period 英文日期正则
_YEAR_RE = re.compile(r"\b(20\d{2}|19\d{2}|1[6-8]\d{2})\b")
_MONTH_YEAR_RE = re.compile(
    r"\b("
    r"jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|"
    r"jul(?:y)?|aug(?:ust)?|sep(?:t(?:ember)?)?|oct(?:ober)?|"
    r"nov(?:ember)?|dec(?:ember)?"
    r")\b[, ]*(20\d{2}|19\d{2}|1[6-8]\d{2})\b",
    flags=re.IGNORECASE,
)
_MONTH_DAY_YEAR_RE = re.compile(
    r"\b("
    r"jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|"
    r"jul(?:y)?|aug(?:ust)?|sep(?:t(?:ember)?)?|oct(?:ober)?|"
    r"nov(?:ember)?|dec(?:ember)?"
    r")\b[ ,]+(\d{1,2})[ ,]+(20\d{2}|19\d{2}|1[6-8]\d{2})\b",
    flags=re.IGNORECASE,
)

# May-December 2010 / May to December 2010 这类“月份-月份 + 年份”模式
_MONTH_RANGE_SAME_YEAR_RE = re.compile(
    r"\b("
    r"jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|"
    r"jul(?:y)?|aug(?:ust)?|sep(?:t(?:ember)?)?|oct(?:ober)?|"
    r"nov(?:ember)?|dec(?:ember)?"
    r")\b\s*(?:-|–|—|to)\s*\b("
    r"jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|"
    r"jul(?:y)?|aug(?:ust)?|sep(?:t(?:ember)?)?|oct(?:ober)?|"
    r"nov(?:ember)?|dec(?:ember)?"
    r")\b\s+(20\d{2}|19\d{2}|1[6-8]\d{2})\b",
    flags=re.IGNORECASE,
)

# 简单范围连接词，用于 Study Period 与经纬度共同复用
_RANGE_SPLIT_RE = re.compile(
    r"\b(?:to|and|until|through|thru|between)\b|[-–—]",
    flags=re.IGNORECASE,
)


# =====================
# 语言与文本预处理
# =====================

def _is_chinese_text(s: str) -> bool:
    """判断字符串是否为中文文本（用于 location/specie 整理时的特例处理）。"""
    if not s:
        return False
    if _HAS_LANGID:
        try:
            lang, _score = langid.classify(s)
            return lang == "zh"
        except Exception:
            pass
    return bool(_CHN_CHAR_RE.search(s))


def _default_translator(s: str) -> Tuple[str, Optional[str]]:
    """默认翻译器：不做翻译，仅回传原文。"""
    return s, None


# =====================
# Study Period 相关工具
# =====================

class ParsedPeriod:
    """
    Study Period 的内部表示结构：起始日期、结束日期、时间精度（year/month/day）。
    """

    __slots__ = ("start", "end", "precision")

    def __init__(self, start: date, end: date, precision: str) -> None:
        self.start = start
        self.end = end
        self.precision = precision


def _parse_one_token_to_date(token: str) -> Tuple[Optional[date], str]:
    """
    将一个字符串 token 尽可能解析为日期：
    - 支持格式：YYYY；Month YYYY；Month DD YYYY 等。
    返回：(日期, 精度)，精度为 "year" / "month" / "day"。
    若无法解析，返回 (None, "")。
    """
    t = token.strip()
    if not t:
        return None, ""

    # Month DD YYYY
    m = _MONTH_DAY_YEAR_RE.search(t)
    if m:
        mon = MONTH_MAP[m.group(1).lower()]
        day = int(m.group(2))
        yr = int(m.group(3))
        try:
            return date(yr, mon, day), "day"
        except ValueError:
            pass

    # Month YYYY
    m = _MONTH_YEAR_RE.search(t)
    if m:
        mon = MONTH_MAP[m.group(1).lower()]
        yr = int(m.group(2))
        try:
            return date(yr, mon, 1), "month"
        except ValueError:
            pass

    # Year only
    m = _YEAR_RE.search(t)
    if m:
        yr = int(m.group(1))
        return date(yr, 1, 1), "year"

    return None, ""


# ---- Ensemble: Meta Items — Study Period parsing ----
# Handles multiple date formats:
#   "2011-2012"                     → year range
#   "December 10, 2011 to January 20, 2012" → day precision
#   "May-December 2010" / "May to December 2010" → month range
# Also handles blocks separated by ";" or "；"
# Returns ParsedPeriod(start, end, precision) or None
def _merge_blocks_to_period(raw: str) -> Optional[ParsedPeriod]:
    """
    将原始 Study Period 字符串解析为 ParsedPeriod 对象：

    - 先按分号/逗号等拆分为 blocks；
    - 对每个 block 进一步按 range 连接词拆分；
    - 支持如：
      * "2011–2012" / "2011 to 2012"
      * "December 10, 2011 to January 20, 2012"
      * "May-December 2010" / "May to December 2010"
    """
    if not raw:
        return None

    s = raw.strip()
    if not s:
        return None

    s = _EN_DASHES_RE.sub("-", s)
    blocks = [b.strip() for b in re.split(r"[;；]+", s) if b.strip()]

    starts: List[date] = []
    ends: List[date] = []
    precisions: List[str] = []

    for blk in blocks:
        # 特例：同一年内月份范围，如 "May-December 2010" 或 "May to December 2010"
        blk_norm = _EN_DASHES_RE.sub("-", blk)
        m_mrange = _MONTH_RANGE_SAME_YEAR_RE.search(blk_norm)
        if m_mrange:
            mon1 = MONTH_MAP[m_mrange.group(1).lower()]
            mon2 = MONTH_MAP[m_mrange.group(2).lower()]
            yr = int(m_mrange.group(3))
            m_start, m_end = (mon1, mon2) if mon1 <= mon2 else (mon2, mon1)
            try:
                sdt = date(yr, m_start, 1)
                edt = date(yr, m_end, 1)
                starts.append(sdt)
                ends.append(edt)
                precisions.append("month")
                continue
            except ValueError:
                # 若失败则回到通用逻辑
                pass

        # 一般性的 range 拆分
        parts = [p.strip() for p in _RANGE_SPLIT_RE.split(blk) if p.strip()]
        if len(parts) == 1:
            d, p = _parse_one_token_to_date(parts[0])
            if d is not None:
                starts.append(d)
                ends.append(d)
                precisions.append(p)
        else:
            left, right = parts[0], parts[-1]
            d1, p1 = _parse_one_token_to_date(left)
            d2, p2 = _parse_one_token_to_date(right)
            if d1 is not None and d2 is not None:
                starts.append(d1)
                ends.append(d2)
                # 精度取更细的那一个
                prec = "year"
                if p1 == "day" or p2 == "day":
                    prec = "day"
                elif p1 == "month" or p2 == "month":
                    prec = "month"
                precisions.append(prec)

    if not starts:
        return None

    # 合并多个子区间：取整体最小 start 与最大 end；精度取最细
    smin = min(starts)
    smax = max(ends)

    if "day" in precisions:
        prec = "day"
    elif "month" in precisions:
        prec = "month"
    else:
        prec = "year"

    return ParsedPeriod(smin, smax, prec)


def _format_period(p: ParsedPeriod) -> str:
    """将 ParsedPeriod 输出为统一的字符串形式。"""
    if p.precision == "day":
        s = p.start.strftime("%Y-%m-%d")
        e = p.end.strftime("%Y-%m-%d")
    elif p.precision == "month":
        s = p.start.strftime("%Y-%m")
        e = p.end.strftime("%Y-%m")
    else:  # year
        s = p.start.strftime("%Y")
        e = p.end.strftime("%Y")
    if s == e:
        return s
    return f"{s} to {e}"


def _choose_study_period(periods: List[ParsedPeriod]) -> Tuple[str, Dict[int, bool]]:
    """
    对若干 Study Period（ParsedPeriod）做加权投票：
    - day > month > year；
    - 以 (start, end, precision) 作为 key 计数；
    - 并返回最终字符串与支持 map。
    """
    if not periods:
        return "", {}

    # key: (start, end, precision)
    keys = [(p.start, p.end, p.precision) for p in periods]
    counter = Counter(keys)
    max_count = max(counter.values())
    best_keys = [k for k, c in counter.items() if c == max_count]

    # 精度优先 day > month > year
    def prec_rank(prec: str) -> int:
        return {"day": 2, "month": 1, "year": 0}.get(prec, 0)

    best_keys.sort(key=lambda k: (-prec_rank(k[2]), k[0], k[1]))
    chosen = best_keys[0]
    chosen_period = ParsedPeriod(chosen[0], chosen[1], chosen[2])
    out = _format_period(chosen_period)

    support_map: Dict[int, bool] = {}
    for i, k in enumerate(keys):
        support_map[i] = (k == chosen)
    return out, support_map


# =====================
# 经纬度解析与区间处理
# =====================

_COORD_NUM_RE = re.compile(
    r"""(?P<sign>[+-])?
        (?P<deg>\d+(?:\.\d+)?)
        (?:[°\s]*(
            (?P<min>\d+(?:\.\d+)?)['’]?
            (?:\s*(?P<sec>\d+(?:\.\d+)?)["”]?)?
        ))?
        \s*(?P<hem>[NnSsEeWw])?
    """,
    flags=re.VERBOSE,
)


def _dms_to_decimal(deg: float, minutes: float = 0.0, seconds: float = 0.0) -> float:
    """度分秒转十进制度数。"""
    sign = 1.0
    if deg < 0:
        sign = -1.0
        deg = abs(deg)
    return sign * (deg + minutes / 60.0 + seconds / 3600.0)


def _parse_coord_value(s: Union[str, float, int]) -> Optional[float]:
    """将单一经纬度字符串解析为十进制度数（不处理范围）。"""
    if s is None:
        return None
    if isinstance(s, (int, float)):
        v = float(s)
        return v if not math.isnan(v) else None

    txt = str(s).strip()
    if not txt:
        return None

    txt = txt.replace("。", ".").replace("．", ".")
    txt = txt.replace("′", "'").replace("″", '"')
    txt = _EN_DASHES_RE.sub("-", txt)

    # 简单纯数值
    try:
        v = float(txt)
        return v
    except Exception:
        pass

    # DMS 格式
    m = _COORD_NUM_RE.search(txt)
    if not m:
        return None

    sign = -1.0 if m.group("sign") == "-" else 1.0
    deg = float(m.group("deg"))
    minutes = float(m.group("min") or 0.0)
    seconds = float(m.group("sec") or 0.0)
    hem = (m.group("hem") or "").upper()

    val = _dms_to_decimal(deg, minutes, seconds)
    val *= sign
    if hem in ("S", "W"):
        val = -abs(val)
    elif hem in ("N", "E"):
        val = abs(val)
    return val


def _format_decimal(x: float, min_decimals: int = 2, max_decimals: int = 4) -> str:
    """对十进制数做四舍五入并格式化，保留 [min_decimals, max_decimals] 位小数。"""
    if x is None or math.isnan(x):
        return ""
    for d in range(max_decimals, min_decimals - 1, -1):
        s = f"{x:.{d}f}"
        s = s.rstrip("0").rstrip(".") if d > min_decimals else s
        return s
    return f"{x:.{min_decimals}f}"


def _parse_coord_interval(s: Union[str, float, int]) -> Optional[Tuple[float, float]]:
    """
    将经纬度表达解析为区间 (low, high)。

    - 若为单点：返回 (v, v)
    - 若为范围（如 "119.5°–120.0°" / "119°30'00\"E to 120°00'00\"E"），
      用 _RANGE_SPLIT_RE 拆分并分别解析端点。
    """
    if s is None:
        return None
    if isinstance(s, (int, float)):
        v = float(s)
        if math.isnan(v):
            return None
        return v, v

    txt = str(s).strip()
    if not txt:
        return None

    txt = txt.replace("。", ".").replace("．", ".")
    txt = txt.replace("′", "'").replace("″", '"')
    txt = _EN_DASHES_RE.sub("-", txt)

    # 简单形如 "119.5-120.0" 的范围
    m_simple = re.match(
        r"^\s*([-+]?\d+(?:\.\d+)?)°?\s*-\s*([-+]?\d+(?:\.\d+)?)°?\s*[NSEWnsew]?\s*$",
        txt,
    )
    if m_simple:
        v1 = float(m_simple.group(1))
        v2 = float(m_simple.group(2))
        return (min(v1, v2), max(v1, v2))

    parts = [p.strip() for p in _RANGE_SPLIT_RE.split(txt) if p.strip()]
    if len(parts) <= 1:
        v = _parse_coord_value(txt)
        if v is None or math.isnan(v):
            return None
        return v, v

    vals: List[float] = []
    v1 = _parse_coord_value(parts[0])
    v2 = _parse_coord_value(parts[-1])
    for vv in (v1, v2):
        if vv is not None and not math.isnan(vv):
            vals.append(vv)
    if len(vals) < 2:
        vals = []
        for p in parts:
            vv = _parse_coord_value(p)
            if vv is not None and not math.isnan(vv):
                vals.append(vv)
    if not vals:
        return None
    low, high = min(vals), max(vals)
    return low, high


def _format_coord_interval(
    interval: Tuple[float, float],
    min_decimals: int = 2,
    max_decimals: int = 4,
) -> str:
    """
    将经纬度区间格式化为字符串：
    - 若 low 与 high 在 max_decimals 精度下相等，则输出单值；
    - 否则输出 "low–high"（en dash）。
    """
    low, high = interval
    if low is None or high is None or math.isnan(low) or math.isnan(high):
        return ""
    low_s = _format_decimal(low, min_decimals=min_decimals, max_decimals=max_decimals)
    high_s = _format_decimal(high, min_decimals=min_decimals, max_decimals=max_decimals)
    try:
        if abs(float(low_s) - float(high_s)) < 10 ** (-max_decimals):
            return low_s
    except Exception:
        pass
    return f"{low_s}–{high_s}"


# ---- Ensemble: Meta Items — Coordinate ensemble in interval space ----
# 1. Parse each answer as (low, high) interval
# 2. Vote in interval space: key = (round(low,4), round(high,4))
# 3. Tie-break: 1) narrower span, 2) center closer to median
# 4. Format output: single value or "low–high" range
def _choose_coordinate(answers: List[Union[str, float, int]]) -> Tuple[str, Dict[int, bool]]:
    """
    对经纬度答案在“区间空间”进行 ensemble：
    - 每个答案解析为 (low, high)；
    - 以 (round(low, 4), round(high, 4)) 为 key 计数投票；
    - 若出现并列，则优先：
      1) 区间跨度更小；
      2) 区间中心更接近所有中心的中位数。
    """
    intervals: List[Tuple[float, float]] = []
    idx_list: List[int] = []
    for i, v in enumerate(answers):
        iv = _parse_coord_interval(v)
        if iv is None:
            continue
        lo, hi = iv
        if lo is None or hi is None or math.isnan(lo) or math.isnan(hi):
            continue
        intervals.append((lo, hi))
        idx_list.append(i)

    if not intervals:
        return "", {i: False for i in range(len(answers))}

    def key_of(iv: Tuple[float, float]) -> Tuple[float, float]:
        lo, hi = iv
        return (round(float(lo), 4), round(float(hi), 4))

    keys = [key_of(iv) for iv in intervals]
    counter: Counter[Tuple[float, float]] = Counter(keys)
    max_count = max(counter.values())
    best_keys = [k for k, c in counter.items() if c == max_count]

    def span(k: Tuple[float, float]) -> float:
        return k[1] - k[0]

    if len(best_keys) > 1:
        centers = [0.5 * (lo + hi) for (lo, hi) in keys]
        centers_sorted = sorted(centers)
        if len(centers_sorted) % 2 == 1:
            med_center = centers_sorted[len(centers_sorted) // 2]
        else:
            med_center = 0.5 * (
                centers_sorted[len(centers_sorted) // 2 - 1]
                + centers_sorted[len(centers_sorted) // 2]
            )

        def center_of(k: Tuple[float, float]) -> float:
            return 0.5 * (k[0] + k[1])

        best_keys.sort(key=lambda k: (span(k), abs(center_of(k) - med_center)))

    chosen_key = best_keys[0]
    out = _format_coord_interval(chosen_key, min_decimals=2, max_decimals=4)

    support_map: Dict[int, bool] = {i: False for i in range(len(answers))}
    for iv, idx in zip(intervals, idx_list):
        if key_of(iv) == chosen_key:
            support_map[idx] = True

    return out, support_map


def _choose_coordinate_by_precision_from_normalized_counts(
    norm_counts: Dict[str, int],
    precision_digits: int,
) -> Tuple[str, int]:
    """基于 normalized_value 的计数，在给定 precision_digits 下对经纬度做归并投票。

    规则：
    - 先将每个非空 normalized_value 解析为 (low, high)，并对 low/high 四舍五入到 precision_digits；
    - 归并后按出现次数投票（support 为归并后的总次数）；
    - 若出现并列，则优先：
      1) 区间跨度更小；
      2) 区间中心更接近所有中心的中位数；
    - 若空值（""）占绝对多数（> 50%），则直接输出空值并返回其 support。
    """
    total = sum(norm_counts.values())
    empty_cnt = norm_counts.get("", 0)
    if total > 0 and empty_cnt > total / 2:
        return "", empty_cnt

    # 解析 + 归并计数
    merged: Dict[Tuple[float, float], int] = {}
    centers: List[float] = []
    for norm, cnt in norm_counts.items():
        if not norm:
            continue
        iv = _parse_coord_interval(norm)
        if iv is None:
            continue
        lo, hi = iv
        if lo is None or hi is None or math.isnan(lo) or math.isnan(hi):
            continue
        lo_r = round(lo, precision_digits)
        hi_r = round(hi, precision_digits)
        key = (lo_r, hi_r) if lo_r <= hi_r else (hi_r, lo_r)
        merged[key] = merged.get(key, 0) + int(cnt)
        centers.append((key[0] + key[1]) / 2.0)

    if not merged:
        # 全部解析失败则回退为空（或无效）
        return "", empty_cnt if empty_cnt else 0

    max_cnt = max(merged.values())
    cand_keys = [k for k, c in merged.items() if c == max_cnt]
    if len(cand_keys) == 1:
        chosen_key = cand_keys[0]
    else:
        # tie-break: span then median-center distance
        def span(k: Tuple[float, float]) -> float:
            return abs(k[1] - k[0])

        centers_all = sorted(centers) if centers else []
        if centers_all:
            mid = len(centers_all) // 2
            median_center = centers_all[mid] if len(centers_all) % 2 == 1 else (centers_all[mid - 1] + centers_all[mid]) / 2.0
        else:
            median_center = 0.0

        def score(k: Tuple[float, float]) -> Tuple[float, float, str]:
            c = (k[0] + k[1]) / 2.0
            return (span(k), abs(c - median_center), f"{k[0]}:{k[1]}")

        chosen_key = sorted(cand_keys, key=score)[0]

    lo, hi = chosen_key
    out = _format_coord_interval((lo, hi), min_decimals=precision_digits, max_decimals=precision_digits)
    return out, merged[chosen_key]



# =====================
# Location / Specie 文本标准化
# =====================

def _normalize_location_specie_string(s: str) -> str:
    """
    对 location/specie 类型的字符串做标准化：
    - 去除括号内容；
    - 小写；
    - 用空格统一空白；
    - 去掉 [].'' 等外围符号；
    - 以逗号等拆分分块；
    - 去停用词后做排序，保证 word 顺序不影响匹配。
    """
    if not s:
        return ""

    s = str(s)
    # 去掉外层 [] 和引号
    if s.startswith("[") and s.endswith("]"):
        s = s[1:-1]
    s = s.strip(" '\"")

    s = _PARENS_RE.sub(" ", s)
    s = s.replace("/", " ")
    s = _WHITESPACE_RE.sub(" ", s)
    s = s.strip().lower()

    # 拆分成块（逗号）
    parts = [p.strip() for p in _COMMA_SPLIT_RE.split(s) if p.strip()]
    all_tokens: List[str] = []
    for p in parts:
        # 去除非字母数字
        keep = _KEEP_CHARS_RE.sub(" ", p)
        tokens = [t for t in _WHITESPACE_RE.split(keep) if t]
        all_tokens.extend(tokens)

    if not all_tokens:
        return ""

    # 去停用词并排序
    tokens = [t for t in all_tokens if t not in _STOPWORD_SET]
    if not tokens:
        tokens = all_tokens
    tokens = sorted(tokens)
    return " ".join(tokens)


# =====================
# 单值标准化入口
# =====================

def _detect_item_category(item: str) -> str:
    """
    根据 item 名称粗略判断类型：
    - study_period: 包含 'period' 或 'study period'
    - coord: 包含 'longitude', 'latitude', 'lat', 'lon'
    - text: 其他 location/specie 文本
    """
    if not item:
        return "text"
    name = str(item).strip().lower()
    if "period" in name:
        return "period"
    if "longitude" in name or "latitude" in name or name in ("lon", "lat"):
        return "coord"
    return "text"


def _normalize_single_value(
    value: Any,
    item_name: str,
    translator: Callable[[str], Tuple[str, Optional[str]]] = _default_translator,
) -> str:
    """
    单个 value 的标准化入口：
    - 对 Study Period：解析为 ParsedPeriod，再统一输出格式；
    - 对 Longitude / Latitude：解析为区间，并格式化输出；
    - 对文本类：去除括号、停用词等，返回标准化 token 串。
    """
    if value is None:
        return ""

    s = str(value).strip()
    if not s:
        return ""

    cat = _detect_item_category(item_name)

    # A) Study Period
    if cat == "period":
        p = _merge_blocks_to_period(s)
        return _format_period(p) if p is not None else ""

    # B) 经纬度
    if cat == "coord":
        iv = _parse_coord_interval(s)
        return _format_coord_interval(iv, 2, 4) if iv is not None else ""

    # C) 文本（location/specie）
    # 可选翻译（如需要强制中翻英可在外层传入 translator）
    try:
        s_trans, _ = translator(s)
    except Exception:
        s_trans = s
    return _normalize_location_specie_string(s_trans)


# =====================
# group 内 ensemble 逻辑
# =====================

# ---- Ensemble: Meta Items — Main ensemble function ----
# For each (paper_index, question_index, item) group:
#   1. Normalize all values via _normalize_single_value() per item type
#   2. Empty-value check: if >50% are empty, output empty
#   3. Route by item type:
#      Study Period → _choose_study_period() (weighted vote: day > month > year)
#      Coordinates   → _choose_coordinate() (interval-space vote, precision merging)
#      Text (location/specie) → token-normalized vote
#   4. For coordinates, apply precision_digits rounding to merge close values
# Returns: ensemble_value, support count, method, n_models
def _ensemble_group(
    raw_values: List[Any],
    item: str,
    translator: Callable[[str], Tuple[str, Optional[str]]],
    precision_digits: int = 2,
) -> Tuple[str, int, str, int, Dict[int, bool]]:
    """
    对同一 (paper_index, question_index, item) 下的一组 value 做 ensemble。

    返回：
    - ensemble_value: 最终选出的值（可能是空字符串）
    - support: 支持该值的样本数
    - method: 使用的 ensemble 方法说明
    - n_models: 该组中总样本数
    - support_map: 每个原始样本是否支持最终值（用于 ensemble_dataframe_all_data）
    """
    n_models = len(raw_values)
    item_cat = _detect_item_category(item)

    # 统一规则：若标准化后的空值占绝对多数（> 50%），则直接输出空值
    normalized_all = [_normalize_single_value(v, item, translator=translator) for v in raw_values]
    empty_support = sum(1 for nv in normalized_all if nv == "")
    if n_models > 0 and empty_support > n_models / 2:
        support_map = {i: (normalized_all[i] == "") for i in range(n_models)}
        return "", empty_support, "majority_empty", n_models, support_map


    # Study Period：解析为 ParsedPeriod，交给 _choose_study_period
    if item_cat == "period":
        periods: List[ParsedPeriod] = []
        idx_map: List[int] = []
        for i, v in enumerate(raw_values):
            p = _merge_blocks_to_period(str(v)) if v not in (None, "") else None
            if p is not None:
                periods.append(p)
                idx_map.append(i)

        if not periods:
            # 全部为空或无法解析
            return "", 0, "period:no_valid", n_models, {i: False for i in range(n_models)}

        out, sub_support_map = _choose_study_period(periods)
        # 映射回原始 index
        support_map: Dict[int, bool] = {i: False for i in range(n_models)}
        for local_idx, ok in sub_support_map.items():
            support_map[idx_map[local_idx]] = ok
        support = sum(support_map.values())
        return out, support, "period:vote", n_models, support_map

    # 经纬度：使用区间 ensemble
    if item_cat == "coord":
        out, support_map = _choose_coordinate(raw_values)
        support = sum(support_map.values())
        return out, support, "coord:interval_vote", n_models, support_map

    # 文本类：先标准化，再按出现次数投票
    normalized_list: List[str] = []
    for v in raw_values:
        nv = _normalize_single_value(v, item, translator=translator)
        normalized_list.append(nv)

    counter = Counter(normalized_list)
    # 允许空值参与计数：如果空值唯一最高，则 ensemble_value 可以为 ""（NaN）
    max_count = max(counter.values())
    candidates = [k for k, c in counter.items() if c == max_count]

    # 如果有非空候选，优先选非空
    non_empty = [c for c in candidates if c != ""]
    if non_empty:
        chosen = sorted(non_empty)[0]
    else:
        chosen = ""  # 全部为空或空值最高

    support = counter[chosen]
    support_map = {i: (normalized_list[i] == chosen) for i in range(n_models)}
    return chosen, support, "text:vote", n_models, support_map


# =====================
# 主接口：ensemble_dataframe_all_data / ensemble_dataframe
# =====================

def ensemble_dataframe_all_data(
    df: pd.DataFrame,
    translator: Callable[[str], Tuple[str, Optional[str]]] = _default_translator,
) -> pd.DataFrame:
    """
    对输入 df（须包含：paper_index, question_index, item, value）按
    (paper_index, question_index, item, normalized_value) 统计频数。

    返回 DataFrame 包含列：
    - paper_index
    - question_index
    - item
    - normalized_value
    - count
    - fraction
    """
    required_cols = {"paper_index", "question_index", "item", "value"}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"Input df must contain columns: {required_cols}")

    records: Dict[Tuple[Any, Any, str, str], int] = Counter()

    for _, row in df.iterrows():
        paper = row["paper_index"]
        question = row["question_index"]
        item = str(row["item"])
        val = row["value"]
        normalized = _normalize_single_value(val, item, translator=translator)
        key = (paper, question, item, normalized)
        records[key] += 1

    out_rows = []
    # 统计 fraction
    group_totals: Dict[Tuple[Any, Any, str], int] = Counter()
    for (paper, q, item, norm), cnt in records.items():
        group_totals[(paper, q, item)] += cnt

    for (paper, q, item, norm), cnt in records.items():
        total = group_totals[(paper, q, item)]
        frac = cnt / total if total > 0 else 0.0
        out_rows.append(
            {
                "paper_index": paper,
                "question_index": q,
                "item": item,
                "normalized_value": norm,
                "count": cnt,
                "fraction": frac,
            }
        )

    out_df = pd.DataFrame(out_rows)
    # 去掉 paper_index 为空的记录（与之前要求保持一致）
    out_df = out_df[~out_df["paper_index"].isna()].reset_index(drop=True)
    return out_df


def ensemble_dataframe(
    df: pd.DataFrame,
    translator: Callable[[str], Tuple[str, Optional[str]]] = _default_translator,
    precision_digits: int = 2,
) -> pd.DataFrame:
    """
    在 ensemble_dataframe_all_data 的基础上，为每个
    (paper_index, question_index, item) 给出最终 ensemble 结果。

    参数：
    - df: 包含 paper_index, question_index, item, value 的原始 DataFrame；
    - translator: 文本翻译函数（可选）；
    - precision_digits: 对经纬度结果做合并时保留的小数位数（用于对已经标准化的
      normalized_value 再做一层四舍五入、合并 close 值，例如：
      119.8333(4 次)、119.83(2 次)、119.833(2 次)：
        * 若 precision_digits=4，则输出 119.8333，support=4；
        * 若 precision_digits=2，则输出 119.83，support=8）。

    输出列：
    - paper_index
    - question_index
    - item
    - ensemble_value
    - support
    - method
    - n_models
    """
    # 先获得 normalized_value 频数表
    all_data = ensemble_dataframe_all_data(df, translator=translator)

    results: List[Dict[str, Any]] = []

    # 按 group 聚合
    for (paper, q, item), sub in all_data.groupby(["paper_index", "question_index", "item"]):
        # 跳过 paper_index 为空
        if pd.isna(paper):
            continue

        # 还需要原始 value 列以便调用 _ensemble_group
        mask = (df["paper_index"] == paper) & (df["question_index"] == q) & (df["item"] == item)
        raw_values = df.loc[mask, "value"].tolist()

        ensemble_value, support, method, n_models, _ = _ensemble_group(
            raw_values=raw_values,
            item=item,
            translator=translator,
            precision_digits=precision_digits,
        )

        # 对经纬度再按 precision_digits 做一次归并（在 normalized 层面）
        cat = _detect_item_category(item)
        if cat == "coord":
            # 基于 normalized_value 的分布，在 precision_digits 下先归并再投票
            norm_counts: Dict[str, int] = {str(r["normalized_value"]): int(r["count"]) for _, r in sub.iterrows()}
            coord_value, coord_support = _choose_coordinate_by_precision_from_normalized_counts(
                norm_counts=norm_counts,
                precision_digits=precision_digits,
            )
            # 以归并后的结果覆盖 ensemble_value/support（保持空值绝对多数可被选中）
            if coord_value == "":
                ensemble_value = ""
            else:
                ensemble_value = coord_value
            if coord_support:
                support = coord_support
            method = f"{method}+coord_round{precision_digits}"
        if cat == "coord" and ensemble_value:
            # ensemble_value 可能是单值或 "low–high"
            if "–" in ensemble_value:
                # 区间
                low_str, high_str = ensemble_value.split("–", 1)
                try:
                    low = float(low_str)
                    high = float(high_str)
                    low_r = round(low, precision_digits)
                    high_r = round(high, precision_digits)
                    ensemble_value = _format_coord_interval(
                        (low_r, high_r), min_decimals=precision_digits, max_decimals=precision_digits
                    )
                except Exception:
                    pass
            else:
                try:
                    v = float(ensemble_value)
                    v_r = round(v, precision_digits)
                    ensemble_value = _format_decimal(
                        v_r, min_decimals=precision_digits, max_decimals=precision_digits
                    )
                except Exception:
                    pass

        results.append(
            {
                "paper_index": paper,
                "question_index": q,
                "item": item,
                "ensemble_value": ensemble_value if ensemble_value != "" else float("nan"),
                "support": support,
                "method": method,
                "n_models": n_models,
            }
        )

    out_df = pd.DataFrame(results)
    # 去掉 paper_index 为空
    out_df = out_df[~out_df["paper_index"].isna()].reset_index(drop=True)
    return out_df
