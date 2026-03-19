import os
import json
from typing import Tuple

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import gspread
from google.oauth2.service_account import Credentials


# =========================
# 1) 기본 설정
# =========================
st.set_page_config(
    page_title="스타일별 매장 PLC 대시보드",
    page_icon="📈",
    layout="wide",
)

st.title("스타일별 매장 PLC 대시보드")
st.caption("유사상품의 작년 매출을 기반으로 스타일별 매장 PLC를 조회합니다.")


# =========================
# 2) 구글시트 연결
# =========================
def get_gspread_client():
    """
    Streamlit secrets 또는 환경변수에서 구글 서비스계정 정보를 읽어
    gspread client를 생성합니다.

    방법 1) .streamlit/secrets.toml 사용
    방법 2) 환경변수 GOOGLE_SERVICE_ACCOUNT_JSON 사용
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

    raise ValueError(
        "구글 서비스 계정 정보가 없습니다. "
        "st.secrets['gcp_service_account'] 또는 GOOGLE_SERVICE_ACCOUNT_JSON을 설정하세요."
    )


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
    values = ws.get_all_records()
    df = pd.DataFrame(values)
    return df


def get_sheets_config() -> dict:
    """
    secrets.toml의 [sheets] 섹션을 dict로 반환합니다.
    """
    if "sheets" not in st.secrets:
        raise ValueError("st.secrets['sheets'] 설정이 없습니다. secrets.toml에 [sheets] 섹션을 추가하세요.")
    return dict(st.secrets["sheets"])


def get_forecast_base_sheet_name() -> str:
    """
    기본으로 사용할 워크시트명(=forecast_base)을 반환합니다.
    - 권장 키: sheets.forecast_base_sheet
    - 하위 호환: sheets.worksheet
    """
    sheets_cfg = get_sheets_config()
    return (
        sheets_cfg.get("forecast_base_sheet")
        or sheets_cfg.get("worksheet")
        or "forecast_base"
    )


# =========================
# 3) 데이터 전처리
# =========================
def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    컬럼명 정리 및 타입 변환
    """
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]

    required_cols = [
        "style_code",
        "similar_style_code",
        "similar_store_code",
        "similar_store_name",
        "similar_week",
        "similar_gross_sales",
    ]

    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        raise ValueError(f"필수 컬럼이 없습니다: {missing_cols}")

    # 문자열 컬럼 정리
    str_cols = [
        "style_code",
        "similar_style_code",
        "similar_color",
        "similar_size",
        "similar_sku",
        "similar_store_code",
        "similar_store_name",
        "similar_week",
        "similar_style_name",
    ]
    for col in str_cols:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()

    # 숫자 컬럼 정리
    num_cols = ["similar_forecast_qty", "similar_gross_sales"]
    for col in num_cols:
        if col in df.columns:
            df[col] = (
                df[col]
                .astype(str)
                .str.replace(",", "", regex=False)
                .str.replace("₩", "", regex=False)
                .str.strip()
            )
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    # week 정렬용 숫자 컬럼 생성
    # 예: 2025-10 -> 202510
    df["week_sort"] = (
        df["similar_week"]
        .str.replace("-", "", regex=False)
        .str.replace(" ", "", regex=False)
    )
    df["week_sort"] = pd.to_numeric(df["week_sort"], errors="coerce")

    return df


# =========================
# 4) PLC 계산 함수
# =========================
def build_style_summary(df: pd.DataFrame, selected_style: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    선택한 style_code에 대해
    1) 주차별 전체 매출(전체 PLC)
    2) 매장별 주차 매출
    를 반환합니다.
    """
    style_df = df[df["style_code"] == selected_style].copy()

    if style_df.empty:
        return pd.DataFrame(), pd.DataFrame()

    # 같은 주차/매장에 여러 행이 있을 수 있으므로 합산
    store_week_df = (
        style_df.groupby(
            ["style_code", "similar_store_code", "similar_store_name", "similar_week", "week_sort"],
            as_index=False
        )["similar_gross_sales"]
        .sum()
        .sort_values(["similar_store_name", "week_sort"])
    )

    # 스타일 전체 주차별 매출
    total_week_df = (
        store_week_df.groupby(["style_code", "similar_week", "week_sort"], as_index=False)["similar_gross_sales"]
        .sum()
        .sort_values("week_sort")
    )

    # PLC 비중 계산: 각 주차 매출 / 최고 주차 매출
    max_total_sales = total_week_df["similar_gross_sales"].max()
    if max_total_sales == 0:
        total_week_df["plc_index"] = 0
    else:
        total_week_df["plc_index"] = total_week_df["similar_gross_sales"] / max_total_sales

    # 매장별 PLC 비중 계산
    store_week_df = store_week_df.merge(
        total_week_df[["similar_week", "week_sort", "similar_gross_sales"]].rename(
            columns={"similar_gross_sales": "total_week_sales"}
        ),
        on=["similar_week", "week_sort"],
        how="left"
    )

    store_week_df["store_share"] = store_week_df.apply(
        lambda x: x["similar_gross_sales"] / x["total_week_sales"] if x["total_week_sales"] > 0 else 0,
        axis=1
    )

    # 매장 자체 PLC index: 매장의 주차 매출 / 매장 최고 주차 매출
    store_max = (
        store_week_df.groupby(["similar_store_code"], as_index=False)["similar_gross_sales"]
        .max()
        .rename(columns={"similar_gross_sales": "store_max_sales"})
    )

    store_week_df = store_week_df.merge(store_max, on="similar_store_code", how="left")
    store_week_df["store_plc_index"] = store_week_df.apply(
        lambda x: x["similar_gross_sales"] / x["store_max_sales"] if x["store_max_sales"] > 0 else 0,
        axis=1
    )

    return total_week_df, store_week_df


def build_store_rank_table(store_week_df: pd.DataFrame) -> pd.DataFrame:
    """
    매장별 요약 테이블 생성
    """
    if store_week_df.empty:
        return pd.DataFrame()

    summary = (
        store_week_df.groupby(["similar_store_code", "similar_store_name"], as_index=False)
        .agg(
            total_sales=("similar_gross_sales", "sum"),
            avg_store_share=("store_share", "mean"),
            peak_sales=("similar_gross_sales", "max"),
            weeks=("similar_week", "nunique"),
        )
        .sort_values("total_sales", ascending=False)
    )

    summary["avg_store_share_pct"] = (summary["avg_store_share"] * 100).round(2)
    return summary


# =========================
# 6) 데이터 로드
# =========================
try:
    forecast_base_sheet = get_forecast_base_sheet_name()
    sheets_cfg = get_sheets_config()
    sheet_id = sheets_cfg.get("sheet_id", "")
    if sheet_id:
        masked = f"{sheet_id[:6]}...{sheet_id[-6:]}" if len(sheet_id) > 12 else sheet_id
        st.caption(f"연결 대상: sheet_id={masked}, worksheet={forecast_base_sheet}")
    raw_df = load_sheet_as_df(forecast_base_sheet)
    df = clean_data(raw_df)
except Exception as e:
    st.error("데이터 로드 중 오류가 발생했습니다.")
    st.exception(e)
    st.stop()


if df.empty:
    st.warning("시트에 데이터가 없습니다.")
    st.stop()


style_list = sorted(df["style_code"].dropna().unique().tolist())
store_list = sorted(df["similar_store_name"].dropna().unique().tolist())

col1, col2 = st.columns(2)

with col1:
    selected_style = st.selectbox("스타일코드 선택", style_list)

with col2:
    selected_store = st.selectbox(
        "store_name",
        ["전체"] + store_list
    )


# =========================
# 7) 스타일별 집계
# =========================
total_week_df, store_week_df = build_style_summary(df, selected_style)
if selected_store != "전체":
    store_week_df = store_week_df[
        store_week_df["similar_store_name"] == selected_store
    ]

store_summary_df = build_store_rank_table(store_week_df)

if total_week_df.empty:
    st.warning("선택한 스타일의 데이터가 없습니다.")
    st.stop()


# 스타일명 하나 가져오기
style_name = ""
temp_name_df = df[df["style_code"] == selected_style]
if "similar_style_name" in temp_name_df.columns and not temp_name_df.empty:
    names = temp_name_df["similar_style_name"].dropna().unique().tolist()
    if names:
        style_name = names[0]

# KPI
total_sales = int(store_week_df["similar_gross_sales"].sum())
store_count = int(store_week_df["similar_store_code"].nunique())
week_count = int(store_week_df["similar_week"].nunique())
peak_week = total_week_df.loc[total_week_df["similar_gross_sales"].idxmax(), "similar_week"]

k1, k2, k3, k4 = st.columns(4)
k1.metric("스타일코드", selected_style)
k2.metric("유사스타일 총매출", f"{total_sales:,.0f}")
k3.metric("매장 수", f"{store_count}")
k4.metric("피크 주차", peak_week)

if style_name:
    st.caption(f"스타일명 참고: {style_name}")


# =========================
# 8) 전체 PLC 그래프
# =========================
st.subheader("1. 전체 평균 PLC")
fig_total = px.line(
    total_week_df,
    x="similar_week",
    y="plc_index",
    markers=True,
    title=f"{selected_style} 전체 PLC 곡선"
)
fig_total.update_layout(
    xaxis_title="주차",
    yaxis_title="PLC Index (0~1)",
    yaxis=dict(range=[0, 1.1]),
    height=420
)
st.plotly_chart(fig_total, use_container_width=True)


# =========================
# 9) 매장 PLC 그래프
# =========================
st.subheader("2. 매장별 PLC")

top_store_df = store_week_df.copy()

fig_store = px.line(
    top_store_df,
    x="similar_week",
    y="store_plc_index",
    color="similar_store_name",
    markers=True,
    title=f"{selected_style} 매장별 PLC"
)
fig_store.update_layout(
    xaxis_title="주차",
    yaxis_title="매장 PLC Index (0~1)",
    yaxis=dict(range=[0, 1.1]),
    height=500,
    legend_title="매장명"
)
st.plotly_chart(fig_store, use_container_width=True)


# =========================
# 10) 매장 점유율 히트맵
# =========================
st.subheader("3. 매장별 주차 점유율 히트맵")

heatmap_df = (
    top_store_df.pivot_table(
        index="similar_store_name",
        columns="similar_week",
        values="store_share",
        aggfunc="sum",
        fill_value=0
    )
    .sort_index()
)

if not heatmap_df.empty:
    fig_heatmap = go.Figure(
        data=go.Heatmap(
            z=heatmap_df.values,
            x=heatmap_df.columns,
            y=heatmap_df.index,
            text=[[f"{v:.1%}" for v in row] for row in heatmap_df.values],
            texttemplate="%{text}",
            hovertemplate="주차=%{x}<br>매장=%{y}<br>점유율=%{z:.2%}<extra></extra>",
        )
    )
    fig_heatmap.update_layout(
        title="주차별 매장 점유율",
        xaxis_title="주차",
        yaxis_title="매장명",
        height=500
    )
    st.plotly_chart(fig_heatmap, use_container_width=True)


# =========================
# 11) 매장 요약표
# =========================
st.subheader("4. 매장별 요약")
display_summary = store_summary_df.copy()
display_summary = display_summary.rename(columns={
    "similar_store_code": "매장코드",
    "similar_store_name": "매장명",
    "total_sales": "총매출",
    "avg_store_share_pct": "평균 점유율(%)",
    "peak_sales": "피크매출",
    "weeks": "주차수"
})
st.dataframe(display_summary, use_container_width=True, hide_index=True)


# =========================
# 12) 원본 집계표
# =========================
st.subheader("5. 주차별 매장 매출 상세")
detail_df = store_week_df.copy().rename(columns={
    "similar_store_code": "매장코드",
    "similar_store_name": "매장명",
    "similar_week": "주차",
    "similar_gross_sales": "매출",
    "total_week_sales": "전체주차매출",
    "store_share": "주차점유율",
    "store_plc_index": "매장PLC"
})
detail_df["주차점유율"] = (detail_df["주차점유율"] * 100).round(2)
detail_df["매장PLC"] = detail_df["매장PLC"].round(4)

st.dataframe(
    detail_df[
        ["매장코드", "매장명", "주차", "매출", "전체주차매출", "주차점유율", "매장PLC"]
    ],
    use_container_width=True,
    hide_index=True
)
