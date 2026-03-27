import json
import os
from typing import Dict, List

import gspread
import pandas as pd
import streamlit as st
from google.oauth2.service_account import Credentials
from supabase import create_client


SUPABASE_URL = st.secrets["SUPABASE_URL"]
SUPABASE_KEY = st.secrets["SUPABASE_KEY"]
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)


def make_unique_headers(headers: List[str]) -> List[str]:
    seen = {}
    result = []
    for h in headers:
        col = str(h).strip() or "unnamed"
        seen[col] = seen.get(col, 0) + 1
        result.append(col if seen[col] == 1 else f"{col}_{seen[col]}")
    return result


def get_gspread_client():
    scopes = [
        "https://www.googleapis.com/auth/spreadsheets",
        "https://www.googleapis.com/auth/drive.readonly",
    ]
    creds_dict = dict(st.secrets["gcp_service_account"])
    credentials = Credentials.from_service_account_info(creds_dict, scopes=scopes)
    return gspread.authorize(credentials)


def load_sheet_as_df(sheet_id: str, worksheet_name: str) -> pd.DataFrame:
    client = get_gspread_client()
    sh = client.open_by_key(sheet_id)
    ws = sh.worksheet(worksheet_name)
    values = ws.get_all_values()
    if not values:
        return pd.DataFrame()

    headers = make_unique_headers(values[0])
    rows = values[1:] if len(values) > 1 else []
    if not rows:
        return pd.DataFrame(columns=headers)

    max_cols = len(headers)
    normalized_rows = []
    for row in rows:
        r = list(row)
        if len(r) < max_cols:
            r += [""] * (max_cols - len(r))
        elif len(r) > max_cols:
            r = r[:max_cols]
        normalized_rows.append(r)
    return pd.DataFrame(normalized_rows, columns=headers)


def get_spreadsheet(sheet_id: str):
    client = get_gspread_client()
    return client.open_by_key(sheet_id)


def get_available_sheet_names(sh) -> List[str]:
    return [ws.title for ws in sh.worksheets()]


def load_sheet_as_df_from_spreadsheet(sh, worksheet_name: str, fallback_first: bool = False) -> pd.DataFrame:
    try:
        ws = sh.worksheet(worksheet_name)
    except Exception:
        if not fallback_first:
            available = ", ".join(get_available_sheet_names(sh))
            raise ValueError(f"시트 '{worksheet_name}'를 찾지 못했습니다. 사용 가능 시트: [{available}]")
        worksheets = sh.worksheets()
        if not worksheets:
            raise ValueError("스프레드시트에 워크시트가 없습니다.")
        ws = worksheets[0]
        st.warning(f"'{worksheet_name}' 시트를 찾지 못해 첫 번째 시트('{ws.title}')를 사용합니다.")

    values = ws.get_all_values()
    if not values:
        return pd.DataFrame()
    headers = make_unique_headers(values[0])
    rows = values[1:] if len(values) > 1 else []
    if not rows:
        return pd.DataFrame(columns=headers)

    max_cols = len(headers)
    normalized_rows = []
    for row in rows:
        r = list(row)
        if len(r) < max_cols:
            r += [""] * (max_cols - len(r))
        elif len(r) > max_cols:
            r = r[:max_cols]
        normalized_rows.append(r)
    return pd.DataFrame(normalized_rows, columns=headers)


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


def run_from_google_sheet():
    sheets_cfg = dict(st.secrets["sheets"])
    sheet_id = sheets_cfg["sheet_id"]
    run_sheet = sheets_cfg.get("run_sheet", "run_info")
    monthly_sheet = sheets_cfg.get("monthly_sheet", "monthly_forecast")
    weekly_sheet = sheets_cfg.get("weekly_sheet", "weekly_forecast")

    sh = get_spreadsheet(sheet_id)
    run_df = load_sheet_as_df_from_spreadsheet(sh, run_sheet, fallback_first=True)
    monthly_df = load_sheet_as_df_from_spreadsheet(sh, monthly_sheet, fallback_first=False)
    weekly_df = load_sheet_as_df_from_spreadsheet(sh, weekly_sheet, fallback_first=False)

    if run_df.empty:
        raise ValueError(f"'{run_sheet}' 시트가 비어 있습니다.")
    if monthly_df.empty:
        raise ValueError(f"'{monthly_sheet}' 시트가 비어 있습니다.")
    if weekly_df.empty:
        raise ValueError(f"'{weekly_sheet}' 시트가 비어 있습니다.")

    required_run_cols = ["sku"]
    required_monthly_cols = ["sku", "year_month", "forecast_qty"]
    required_weekly_cols = ["sku", "year_week", "forecast_qty"]

    for c in required_run_cols:
        if c not in run_df.columns:
            raise ValueError(f"run 시트 필수 컬럼 누락: {c}")
    for c in required_monthly_cols:
        if c not in monthly_df.columns:
            raise ValueError(f"monthly 시트 필수 컬럼 누락: {c}")
    for c in required_weekly_cols:
        if c not in weekly_df.columns:
            raise ValueError(f"weekly 시트 필수 컬럼 누락: {c}")

    success = 0
    fail = 0
    logs = []

    for _, run_row in run_df.iterrows():
        try:
            sku = normalize_value(run_row["sku"])
            sty = normalize_value(run_row.get("style_code") or run_row.get("sty"))

            m = monthly_df[monthly_df["sku"].astype(str) == str(sku)].copy()
            w = weekly_df[weekly_df["sku"].astype(str) == str(sku)].copy()

            if sty is not None:
                if "style_code" in m.columns:
                    m = m[m["style_code"].astype(str) == str(sty)]
                elif "sty" in m.columns:
                    m = m[m["sty"].astype(str) == str(sty)]
                if "style_code" in w.columns:
                    w = w[w["style_code"].astype(str) == str(sty)]
                elif "sty" in w.columns:
                    w = w[w["sty"].astype(str) == str(sty)]

            if m.empty or w.empty:
                raise ValueError("해당 SKU의 monthly/weekly 데이터가 없습니다.")

            save_to_supabase(dict(run_row), m, w)
            success += 1
            logs.append({"sku": sku, "status": "success", "message": ""})
        except Exception as e:
            fail += 1
            logs.append({"sku": normalize_value(run_row.get("sku")), "status": "fail", "message": str(e)})

    return success, fail, pd.DataFrame(logs)


def main():
    st.set_page_config(page_title="Sheet -> Supabase ETL", layout="wide")
    st.title("Google Sheet -> Supabase 저장")
    st.write("버튼을 누르면 시트 데이터를 읽어 3개 테이블에 저장합니다.")

    if st.button("실행하기", type="primary", use_container_width=True):
        with st.spinner("구글 시트 읽는 중 / Supabase 저장 중..."):
            try:
                success, fail, log_df = run_from_google_sheet()
                st.success(f"완료: 성공 {success}건 / 실패 {fail}건")
                st.dataframe(log_df, use_container_width=True, hide_index=True)
            except Exception as e:
                st.error(f"실행 실패: {e}")


if __name__ == "__main__":
    main()
