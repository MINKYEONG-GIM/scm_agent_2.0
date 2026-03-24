import os
import json
import re
import urllib.error
import urllib.request
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import streamlit as st
import gspread
from google.oauth2.service_account import Credentials
from streamlit_gsheets import GSheetsConnection
from typing import List, Optional, Tuple

# =========================
# 기준값
# =========================
INTRO_MAX_WEEKS = 9
INTRO_MAX_WOW_RATIO = 0.35
OFF_SEASON_MIN_WEEKS = 5
OFF_SEASON_RATIO_TO_MEAN = 0.60
OFF_SEASON_RATIO_TO_MEDIAN = 0.60
LOW_TAIL_RATIO_TO_PEAK = 0.25
LOW_TAIL_MIN_WEEKS = 4


def parse_year_week(value: str) -> Tuple[Optional[int], Optional[int]]:
    if pd.isna(value):
        return None, None
    text = str(value).strip()
    m = re.match(r"^(\d{4})[-/_]?(\d{1,2})$", text)
    if m:
        return int(m.group(1)), int(m.group(2))
    m = re.match(r"^(\d{4})(\d{2})$", text)
    if m:
        return int(m.group(1)), int(m.group(2))
    return None, None


def yearweek_to_sort_key(year_week: str) -> int:
    year, week = parse_year_week(year_week)
    if year is None or week is None:
        return 99999999
    return year * 100 + week


def safe_to_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series.astype(str).str.replace(",", "", regex=False).str.strip(), errors="coerce")


def week_over_week_rate(qty: np.ndarray) -> np.ndarray:
    q = np.asarray(qty, dtype=float)
    n = len(q)
    wow = np.full(n, np.nan)
    eps = 1e-9
    for i in range(1, n):
        prev = q[i - 1]
        if prev <= eps:
            wow[i] = np.inf if q[i] > eps else 0.0
        else:
            wow[i] = (q[i] - prev) / prev
    return wow


def is_stage_column(col_name: str) -> bool:
    return str(col_name).strip().endswith("시즌분류")


def get_item_name_from_stage_col(col_name: str) -> str:
    return str(col_name).replace(" 시즌분류", "").strip()


def extract_reference_stage_wide(df: pd.DataFrame) -> pd.DataFrame:
    stage_cols = [c for c in df.columns if c != "연도/주" and is_stage_column(c)]
    if not stage_cols:
        return pd.DataFrame(columns=["year_week", "item", "reference_stage"])
    long_stage = df[["연도/주"] + stage_cols].melt(
        id_vars=["연도/주"],
        value_vars=stage_cols,
        var_name="stage_col",
        value_name="reference_stage",
    )
    long_stage["item"] = long_stage["stage_col"].apply(get_item_name_from_stage_col)
    long_stage = long_stage.rename(columns={"연도/주": "year_week"})
    long_stage["reference_stage"] = long_stage["reference_stage"].astype(str).str.strip()
    long_stage["reference_stage"] = long_stage["reference_stage"].replace({"": np.nan, "nan": np.nan, "None": np.nan})
    return long_stage[["year_week", "item", "reference_stage"]]


def prepare_data(raw_df: pd.DataFrame) -> pd.DataFrame:
    df = raw_df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    if "연도/주" not in df.columns:
        raise ValueError("'연도/주' 컬럼이 필요합니다.")

    qty_cols = [c for c in df.columns if c != "연도/주" and not is_stage_column(c)]
    long_df = df[["연도/주"] + qty_cols].melt(
        id_vars=["연도/주"],
        value_vars=qty_cols,
        var_name="item",
        value_name="qty",
    )
    long_df["qty"] = safe_to_numeric(long_df["qty"])
    long_df = long_df.dropna(subset=["qty"])
    long_df = long_df.rename(columns={"연도/주": "year_week"})
    long_df["item"] = long_df["item"].astype(str).str.strip()

    ref_df = extract_reference_stage_wide(df)
    long_df = long_df.merge(ref_df, on=["year_week", "item"], how="left")

    yw = long_df["year_week"].apply(parse_year_week)
    long_df["year"] = yw.apply(lambda x: x[0])
    long_df["week"] = yw.apply(lambda x: x[1])
    long_df["sort_key"] = long_df["year_week"].astype(str).apply(yearweek_to_sort_key)
    return long_df


def _find_runs(mask: np.ndarray, min_len: int = 1):
    runs = []
    n = len(mask)
    i = 0
    while i < n:
        if not bool(mask[i]):
            i += 1
            continue
        j = i
        while j < n and bool(mask[j]):
            j += 1
        if j - i >= min_len:
            runs.append((i, j - 1))
        i = j
    return runs


def classify_item_plc(item_df: pd.DataFrame) -> pd.DataFrame:
    df = item_df.sort_values("sort_key").reset_index(drop=True).copy()
    q = df["qty"].fillna(0).astype(float).values
    n = len(q)
    wow = week_over_week_rate(q)
    avg = float(np.mean(q)) if n else 0.0
    med = float(np.median(q)) if n else 0.0
    peak_qty = float(np.max(q)) if n else 0.0
    peak_idx = int(np.argmax(q)) if n else 0

    stages = [""] * n
    ma3 = pd.Series(q).rolling(3, min_periods=1).mean().values if n else np.array([])

    pre_low_cut = max(avg * OFF_SEASON_RATIO_TO_MEAN, med * OFF_SEASON_RATIO_TO_MEDIAN)
    pre_low_mask = (q <= pre_low_cut) & (ma3 <= pre_low_cut)
    pre_low_runs = [r for r in _find_runs(pre_low_mask, OFF_SEASON_MIN_WEEKS) if r[0] == 0 and r[1] < peak_idx]
    if pre_low_runs:
        s, e = pre_low_runs[-1]
        for i in range(s, e + 1):
            stages[i] = "비시즌"

    intro_end = -1
    if n and stages[0] != "비시즌":
        intro_end = 0
        for i in range(1, min(n, INTRO_MAX_WEEKS)):
            ratio_to_peak = q[i] / peak_qty if peak_qty > 0 else 0.0
            wow_i = wow[i] if np.isfinite(wow[i]) else 0.0
            if ratio_to_peak >= 0.35 and wow_i >= INTRO_MAX_WOW_RATIO:
                break
            intro_end = i
        for i in range(0, intro_end + 1):
            stages[i] = "도입"

    mature_min = max(avg, peak_qty * 0.45)
    mature_mask = (q >= mature_min) & (ma3 >= mature_min * 0.95)
    mature_runs = [r for r in _find_runs(mature_mask, 2) if r[0] <= peak_idx <= r[1]]
    if mature_runs:
        m_s, m_e = mature_runs[0]
    else:
        m_s, m_e = max(0, peak_idx - 1), min(n - 1, peak_idx + 1)
    for i in range(m_s, m_e + 1):
        stages[i] = "성숙"

    tail_cut = max(avg * 0.40, peak_qty * LOW_TAIL_RATIO_TO_PEAK)
    tail_mask = (q <= tail_cut) & (ma3 <= max(avg * 0.45, peak_qty * 0.30))
    tail_runs = [r for r in _find_runs(tail_mask, LOW_TAIL_MIN_WEEKS) if r[0] > peak_idx]
    tail_off = tail_runs[-1] if tail_runs else None

    growth_start = intro_end + 1
    growth_end = m_s - 1
    if growth_start <= growth_end:
        for i in range(growth_start, growth_end + 1):
            if not stages[i]:
                stages[i] = "성장"

    decline_start = m_e + 1
    decline_end = (tail_off[0] - 1) if tail_off else (n - 1)
    if decline_start <= decline_end:
        for i in range(decline_start, decline_end + 1):
            if not stages[i]:
                stages[i] = "쇠퇴"

    if tail_off:
        for i in range(tail_off[0], n):
            stages[i] = "비시즌"

    for i in range(n):
        if not stages[i]:
            if i <= intro_end and stages[0] != "비시즌":
                stages[i] = "도입"
            elif i < m_s:
                stages[i] = "성장"
            elif i <= m_e:
                stages[i] = "성숙"
            elif tail_off and i >= tail_off[0]:
                stages[i] = "비시즌"
            else:
                stages[i] = "쇠퇴"

    # reference_stage가 있으면 최종 우선 반영
    if "reference_stage" in df.columns:
        ref_mask = df["reference_stage"].notna()
        for idx in df.index[ref_mask]:
            stages[idx] = str(df.loc[idx, "reference_stage"]).strip()

    df["plc_stage"] = stages
    return df


def run_full_classification(raw_df: pd.DataFrame) -> pd.DataFrame:
    base = prepare_data(raw_df)
    out = base.groupby("item", group_keys=False).apply(classify_item_plc).reset_index(drop=True)
    return out


def validate_against_reference(raw_df: pd.DataFrame):
    result = run_full_classification(raw_df)
    check_df = result[result["reference_stage"].notna()].copy()
    check_df["is_match"] = check_df["reference_stage"].astype(str).str.strip() == check_df["plc_stage"].astype(str).str.strip()
    summary = check_df.groupby("item", as_index=False).agg(
        checked_rows=("is_match", "size"),
        match_rows=("is_match", "sum"),
    )
    summary["accuracy"] = summary["match_rows"] / summary["checked_rows"]
    mismatches = check_df.loc[~check_df["is_match"], ["item", "year_week", "qty", "reference_stage", "plc_stage"]].copy()
    return summary.sort_values(["accuracy", "item"]).reset_index(drop=True), mismatches.reset_index(drop=True)
