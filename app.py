import json
import os
from typing import Dict

import pandas as pd
import streamlit as st
from supabase import create_client


SUPABASE_URL = st.secrets["SUPABASE_URL"]
SUPABASE_KEY = st.secrets["SUPABASE_KEY"]
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)


def normalize_value(value):
    # pandas/numpy scalar -> python scalar, NaN -> None
    if pd.isna(value):
        return None
    if hasattr(value, "item"):
        try:
            value = value.item()
        except Exception:
            pass
    return value


def create_forecast_run(row: Dict) -> int:
    data = {
        "SKU": normalize_value(row["sku"]),
        "style_code": normalize_value(row.get("style_code") or row.get("sty")),
        "plant": normalize_value(row.get("plant")),
        "shape_type": normalize_value(row.get("shape_type")),
        "shape_reason": normalize_value(row.get("shape_reason")),
        "peak_week": normalize_value(row.get("peak_week")),
        "peak_month": normalize_value(row.get("peak_month")),
        "season_start_week": normalize_value(row.get("season_start_week")),
        "season_end_week": normalize_value(row.get("season_end_week")),
    }
    res = supabase.table("sku_forecast_run").insert(data).execute()
    if not res.data:
        raise ValueError("sku_forecast_run insert 실패: 결과 data가 비어 있습니다.")
    return res.data[0]["id"]


def insert_monthly(run_id: int, df: pd.DataFrame):
    rows = []
    for _, r in df.iterrows():
        sty = normalize_value(r.get("style_code")) or normalize_value(r.get("sty"))
        rows.append(
            {
                "forecast_run_id": run_id,
                "sku": normalize_value(r["sku"]),
                "sty": sty,
                "year_month": normalize_value(r["year_month"]),
                "forecast_qty": int(normalize_value(r["forecast_qty"]) or 0),
                "stage": normalize_value(r.get("stage")),
                "is_peak_month": bool(normalize_value(r.get("is_peak_month", False)) or False),
            }
        )
    if rows:
        supabase.table("sku_monthly_forecast").insert(rows).execute()


def insert_weekly(run_id: int, df: pd.DataFrame):
    rows = []
    for _, r in df.iterrows():
        sty = normalize_value(r.get("style_code")) or normalize_value(r.get("sty"))
        rows.append(
            {
                "forecast_run_id": run_id,
                "sku": normalize_value(r["sku"]),
                "sty": sty,
                "year_week": normalize_value(r["year_week"]),
                "forecast_qty": float(normalize_value(r["forecast_qty"]) or 0),
                "stage": normalize_value(r.get("stage")),
                "is_peak_week": bool(normalize_value(r.get("is_peak_week", False)) or False),
            }
        )
    if rows:
        supabase.table("sku_weekly_forecast").insert(rows).execute()


def save_to_supabase(run_info: Dict, monthly_df: pd.DataFrame, weekly_df: pd.DataFrame):
    run_id = create_forecast_run(run_info)
    insert_monthly(run_id, monthly_df)
    insert_weekly(run_id, weekly_df)
    print("saved:", run_id)
