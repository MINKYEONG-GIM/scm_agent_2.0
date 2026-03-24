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




OPENAI_CHAT_URL = "https://api.openai.com/v1/chat/completions"


def get_gpt_gpi() -> Optional[str]:
    """
    Streamlit secrets의 gpt_gpi(또는 OPENAI_API_KEY) 또는 환경변수.
    secrets.toml 예: gpt_gpi = "sk-..."
    """
    try:
        if hasattr(st, "secrets"):
            sec = st.secrets
            if "gpt_gpi" in sec:
                v = sec["gpt_gpi"]
                if v is not None and str(v).strip():
                    return str(v).strip()
            if "OPENAI_API_KEY" in sec:
                v = sec["OPENAI_API_KEY"]
                if v is not None and str(v).strip():
                    return str(v).strip()
    except Exception:
        pass
    return (os.getenv("gpt_gpi") or os.getenv("OPENAI_API_KEY") or "").strip() or None


# =========================
# 공통 유틸
# =========================
def make_unique_headers(headers: List[str]) -> List[str]:
    """
    중복 컬럼명이 있을 때 고유한 이름으로 바꿉니다.
    예: ['A', 'A', 'B'] -> ['A', 'A_2', 'B']
    """
    seen = {}
    result = []

    for h in headers:
        col = str(h).strip()
        if not col:
            col = "unnamed"

        if col not in seen:
            seen[col] = 1
            result.append(col)
        else:
            seen[col] += 1
            result.append(f"{col}_{seen[col]}")

    return result


def clean_number(value):
    """
    문자열 숫자를 안전하게 float로 변환합니다.
    예:
    '12,345' -> 12345.0
    '' -> np.nan
    """
    if pd.isna(value):
        return np.nan

    s = str(value).strip()
    if s == "":
        return np.nan

    s = s.replace(",", "")
    try:
        return float(s)
    except Exception:
        return np.nan


def parse_yearweek_to_date(yearweek: str) -> pd.Timestamp:
    """
    '2025-01' 같은 값을 해당 ISO 주차의 월요일 날짜로 변환합니다.
    """
    s = str(yearweek).strip()

    if not re.match(r"^\d{4}-\d{1,2}$", s):
        return pd.NaT

    year_str, week_str = s.split("-")
    year = int(year_str)
    week = int(week_str)

    try:
        return pd.to_datetime(f"{year}-W{week:02d}-1", format="%G-W%V-%u", errors="coerce")
    except Exception:
        return pd.NaT


# =========================
# 구글 시트 로딩 함수
# =========================
def get_gspread_client():
    """
    Streamlit secrets 또는 환경변수에서 구글 서비스계정 정보를 읽어
    gspread client를 생성합니다.
    """
    scopes = [
        "https://www.googleapis.com/auth/spreadsheets",
        "https://www.googleapis.com/auth/drive.readonly",
    ]

    if "gcp_service_account" in st.secrets:
        creds_dict = dict(st.secrets["gcp_service_account"])
        credentials = Credentials.from_service_account_info(creds_dict, scopes=scopes)
        return gspread.authorize(credentials)

    service_account_json = os.getenv("GOOGLE_SERVICE_ACCOUNT_JSON")
    if service_account_json:
        creds_dict = json.loads(service_account_json)
        credentials = Credentials.from_service_account_info(creds_dict, scopes=scopes)
        return gspread.authorize(credentials)

    raise ValueError("구글 서비스 계정 정보가 없습니다.")


def get_sheets_config() -> dict:
    """
    secrets.toml의 [sheets] 섹션을 dict로 반환합니다.
    필수 키: sheet_id
    선택 키: worksheet
    """
    if "sheets" not in st.secrets:
        raise ValueError("st.secrets['sheets'] 설정이 없습니다. secrets.toml에 [sheets] 섹션을 추가하세요.")
    return dict(st.secrets["sheets"])


@st.cache_data(ttl=300)
def load_sheet_as_df(worksheet_name: str) -> pd.DataFrame:
    """
    구글시트의 특정 워크시트를 DataFrame으로 읽습니다.
    """
    client = get_gspread_client()
    sheets_cfg = get_sheets_config()
    sheet_id = sheets_cfg.get("sheet_id")

    if not sheet_id:
        raise ValueError("secrets.toml의 [sheets].sheet_id 가 비어있습니다.")

    sh = client.open_by_key(sheet_id)

    try:
        ws = sh.worksheet(worksheet_name)
    except Exception as e:
        available = [w.title for w in sh.worksheets()]
        raise ValueError(
            f"워크시트 '{worksheet_name}'를 찾지 못했습니다. 사용 가능한 워크시트: {available}"
        ) from e

    values = ws.get_all_values()
    if not values:
        return pd.DataFrame()

    raw_headers = values[0]
    headers = make_unique_headers([str(h) for h in raw_headers])

    rows = values[1:] if len(values) > 1 else []
    if not rows:
        return pd.DataFrame(columns=headers)

    max_cols = len(headers)
    normalized_rows = []

    for row in rows:
        row = list(row)
        if len(row) < max_cols:
            row = row + [""] * (max_cols - len(row))
        elif len(row) > max_cols:
            row = row[:max_cols]
        normalized_rows.append(row)

    return pd.DataFrame(normalized_rows, columns=headers)


@st.cache_data(ttl=300)
def load_sheet_data() -> pd.DataFrame:
    sheets_cfg = get_sheets_config()

    sheet_id = sheets_cfg.get("sheet_id")
    worksheet_name = (
        sheets_cfg.get("WORKSHEET_NAME")
        or sheets_cfg.get("worksheet")
        or sheets_cfg.get("forecast_base_sheet")
        or "plc db"
    )

    if not sheet_id:
        raise ValueError("secrets.toml의 [sheets].sheet_id 가 비어있습니다.")

    return load_sheet_as_df(worksheet_name)


# =========================
# 데이터 전처리
# =========================
def get_item_columns(df: pd.DataFrame) -> List[str]:
    """
    아이템 선택용 컬럼 목록 반환
    '연도/주'를 제외한 컬럼 중 값이 있는 컬럼만 반환
    """
    exclude_cols = {"연도/주", "", " "}
    candidate_cols = [c for c in df.columns if str(c).strip() not in exclude_cols]

    item_cols = []
    for col in candidate_cols:
        series = df[col].astype(str).str.strip().replace("", np.nan)
        if series.notna().any():
            item_cols.append(col)

    return item_cols


def prepare_item_timeseries(df: pd.DataFrame, item_name: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    선택한 아이템의
    1) 주차별 매출 데이터
    2) 월별 매출 데이터
    를 만들어 반환합니다.
    """
    if df.empty:
        raise ValueError("시트 데이터가 비어 있습니다.")

    if "연도/주" not in df.columns:
        raise ValueError("필수 컬럼 '연도/주'가 없습니다.")

    if item_name not in df.columns:
        raise ValueError(f"선택한 아이템 컬럼 '{item_name}'이 없습니다.")

    temp = df[["연도/주", item_name]].copy()
    temp.columns = ["year_week", "sales"]

    # 연도/주 형식 데이터만 사용
    temp["year_week"] = temp["year_week"].astype(str).str.strip()
    temp = temp[temp["year_week"].str.match(r"^\d{4}-\d{1,2}$", na=False)].copy()

    if temp.empty:
        raise ValueError("연도/주 형식의 데이터가 없습니다. 예: 2025-01")

    # 숫자 변환
    temp["sales"] = temp["sales"].apply(clean_number)

    # 날짜 변환
    temp["week_start"] = temp["year_week"].apply(parse_yearweek_to_date)
    temp = temp.dropna(subset=["week_start"]).copy()

    # 숫자 없는 건 0 처리
    temp["sales"] = temp["sales"].fillna(0)

    # 주차 정렬
    weekly_df = temp.sort_values("week_start").reset_index(drop=True)

    # 월 컬럼 생성
    weekly_df["month"] = weekly_df["week_start"].dt.to_period("M").dt.to_timestamp()

    # 월별 매출 합산
    monthly_df = (
        weekly_df.groupby("month", as_index=False)["sales"]
        .sum()
        .sort_values("month")
        .reset_index(drop=True)
    )

    return weekly_df, monthly_df


# =========================
# 차트 생성
# =========================
def build_dual_line_chart(item_name: str, weekly_df: pd.DataFrame, monthly_df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()

    # 주차별 매출 선
    fig.add_trace(
        go.Scatter(
            x=weekly_df["week_start"],
            y=weekly_df["sales"],
            mode="lines+markers",
            name="주차별 매출",
            hovertemplate="주차 시작일: %{x|%Y-%m-%d}<br>매출: %{y:,.0f}<extra></extra>",
        )
    )

    # 월별 매출 선
    fig.add_trace(
        go.Scatter(
            x=monthly_df["month"],
            y=monthly_df["sales"],
            mode="lines+markers",
            name="월별 매출",
            hovertemplate="월: %{x|%Y-%m}<br>매출: %{y:,.0f}<extra></extra>",
        )
    )

    fig.update_layout(
        title=f"{item_name} 주차별/월별 매출 추이",
        xaxis_title="연도/주",
        yaxis_title="매출",
        height=620,
        hovermode="x unified",
        margin=dict(l=30, r=30, t=70, b=30),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
    )

    fig.update_yaxes(tickformat=",.0f")

    return fig


# =========================
# 월별 매출 형태 판별 (단봉 / 다봉)
# =========================
def count_peaks(values: np.ndarray) -> int:
    """
    local peak 개수
    y[i-1] < y[i] >= y[i+1]
    """
    if len(values) < 3:
        return 0

    peaks = 0

    for i in range(1, len(values) - 1):
        if values[i] > values[i - 1] and values[i] >= values[i + 1]:
            peaks += 1

    return peaks


def classify_monthly_shape(monthly_df: pd.DataFrame) -> str:
    """
    월별 매출 기준
    단봉형 / 다봉형 / 무봉형
    """

    if monthly_df.empty:
        return "판단불가"

    y = monthly_df["sales"].values.astype(float)

    if len(y) < 3:
        return "판단불가"

    peaks = count_peaks(y)

    if peaks == 0:
        return "무봉형"

    if peaks == 1:
        return "단봉형"

    if peaks >= 2:
        return "다봉형"

    return "판단불가"

# =========================
# 메인 화면
# =========================
def main():
    st.set_page_config(page_title="아이템 매출 추이", layout="wide")

    df = load_sheet_data()

    if df.empty:
        st.warning("불러온 데이터가 없습니다.")
        return

    item_cols = get_item_columns(df)
    if not item_cols:
        st.warning("선택 가능한 아이템 컬럼이 없습니다.")
        return

    st.markdown("개별 차트 확인할 아이템")
    selected_item = st.selectbox(
        "개별 차트 확인할 아이템",
        options=item_cols,
        index=item_cols.index("가디건") if "가디건" in item_cols else 0,
        label_visibility="collapsed",
    )

    weekly_df, monthly_df = prepare_item_timeseries(df, selected_item)
    shape_label = classify_monthly_shape(monthly_df)
    
    st.markdown(f"### 형태: {shape_label}")
    
    fig = build_dual_line_chart(selected_item, weekly_df, monthly_df)

    st.plotly_chart(fig, use_container_width=True)

    


if __name__ == "__main__":
    main()
