import os
import json
from typing import Tuple
from datetime import date

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
# 2-1) 보조 시트: sales_actual
# =========================
@st.cache_data(ttl=300)
def load_sales_actual_df() -> pd.DataFrame:
    """
    sales_actual 워크시트를 DataFrame으로 읽습니다.
    컬러 필터 옵션을 얻기 위해 사용합니다.
    """
    return load_sheet_as_df("sales_actual")


def resolve_sales_actual_sales_column(sales_actual_df: pd.DataFrame):
    """
    sales_actual 워크시트에서 '올해 매출'로 사용할 컬럼명을 추정합니다.
    (실제 컬럼명은 시트마다 다를 수 있어, 후보를 순서대로 탐색합니다.)
    """
    candidates = [
        "sales_amount",
        "sales_qty",
        "gross_sales",
        "sales",
        "actual_sales",
        "net_sales",
        "sales_amount",
        "amount",
        "revenue",
    ]
    normalized = {str(c).strip(): c for c in sales_actual_df.columns}
    for name in candidates:
        if name in normalized:
            return normalized[name]  # 원본 컬럼명 반환
    return None


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

    # 할인율 정리 (0~1 비율로 정규화)
    if "similar_discount_rate" in df.columns:
        dr = (
            df["similar_discount_rate"]
            .astype(str)
            .str.replace("%", "", regex=False)
            .str.replace(",", "", regex=False)
            .str.strip()
        )
        dr = pd.to_numeric(dr, errors="coerce")
        # 30(%) 같은 값이면 0.30으로
        if dr.notna().any() and dr.max(skipna=True) > 1:
            dr = dr / 100.0
        df["similar_discount_rate"] = dr

    # week 정렬용 숫자 컬럼 생성
    # 예: 2025-10 -> 202510
    df["week_sort"] = (
        df["similar_week"]
        .str.replace("-", "", regex=False)
        .str.replace(" ", "", regex=False)
    )
    df["week_sort"] = pd.to_numeric(df["week_sort"], errors="coerce")

    def _week_to_month_week_label(v: str) -> str:
        """
        'YYYY-WW' 형태를 'M월 N주차'로 변환.
        ISO week 기준(해당 주의 월요일)으로 월/월내주차 계산.
        변환 실패 시 원본 반환.
        """
        s = str(v).strip()
        if not s or s.lower() == "nan" or "-" not in s:
            return s
        y, w = s.split("-", 1)
        try:
            year = int(y)
            week = int(w)
            d = date.fromisocalendar(year, week, 1)  # Monday of ISO week
            week_in_month = (d.day - 1) // 7 + 1
            return f"{d.month}월 {week_in_month}주차"
        except Exception:
            return s

    df["week_label"] = df["similar_week"].apply(_week_to_month_week_label)

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
            ["style_code", "similar_store_code", "similar_store_name", "similar_week", "week_sort", "week_label"],
            as_index=False
        )["similar_gross_sales"]
        .sum()
        .sort_values(["similar_store_name", "week_sort"])
    )

    # 스타일 전체 주차별 매출
    total_week_df = (
        store_week_df.groupby(["style_code", "similar_week", "week_sort", "week_label"], as_index=False)["similar_gross_sales"]
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

col1, col2, col3, col4 = st.columns(4)

with col1:
    default_style = "SPRWG37G01"
    style_idx = style_list.index(default_style) if default_style in style_list else 0
    selected_style = st.selectbox("스타일", style_list, index=style_idx)

style_df_for_filter = df[df["style_code"] == selected_style].copy()
store_list = sorted(style_df_for_filter["similar_store_name"].dropna().unique().tolist())

with col2:
    # 기본값은 항상 "코엑스몰" (옵션에 없으면 강제로 포함)
    store_options = ["코엑스몰", "전체"] + [s for s in store_list if s not in ["코엑스몰", "전체"]]
    selected_store = st.selectbox("매장", store_options, index=0)

with col3:
    try:
        sales_actual_df = load_sales_actual_df()
        sales_actual_df.columns = [str(c).strip() for c in sales_actual_df.columns]

        color_source = sales_actual_df.copy()
        # sales_actual에 style_code가 있으면 선택 스타일 기준으로 컬러 후보를 좁힘
        if "style_code" in color_source.columns:
            color_source["style_code"] = color_source["style_code"].astype(str).str.strip()
            color_source = color_source[color_source["style_code"] == selected_style]

        if "color" in color_source.columns:
            color_source["color"] = color_source["color"].astype(str).str.strip()
            color_options = sorted([c for c in color_source["color"].dropna().unique().tolist() if c and c != "nan"])
            selected_colors = st.multiselect("컬러", color_options, default=color_options)
        else:
            selected_colors = None
            st.caption("sales_actual 워크시트에 color 컬럼이 없어 컬러 필터를 생략합니다.")
    except Exception:
        selected_colors = None
        st.caption("sales_actual 워크시트 로드에 실패해 컬러 필터를 생략합니다.")

with col4:
    if "similar_size" in style_df_for_filter.columns:
        size_options = sorted(style_df_for_filter["similar_size"].dropna().unique().tolist())
        selected_sizes = st.multiselect("사이즈", size_options, default=size_options)
    else:
        selected_sizes = None
        st.caption("사이즈 컬럼이 없어 사이즈 필터를 생략합니다.")


# =========================
# 7) 스타일별 집계
# =========================
df_filtered = df.copy()
df_filtered = df_filtered[df_filtered["style_code"] == selected_style]
if selected_colors is not None:
    if "color" in df_filtered.columns:
        df_filtered = df_filtered[df_filtered["color"].isin(selected_colors)]
if selected_sizes is not None:
    df_filtered = df_filtered[df_filtered["similar_size"].isin(selected_sizes)]

total_week_df, store_week_df_all = build_style_summary(df_filtered, selected_style)
store_week_df = store_week_df_all.copy()
if selected_store != "전체":
    store_week_df = store_week_df[store_week_df["similar_store_name"] == selected_store]

store_summary_df = build_store_rank_table(store_week_df)

if total_week_df.empty:
    st.warning("선택한 스타일의 데이터가 없습니다.")
    st.stop()


# 스타일명: sales_actual 워크시트의 style_name에서 가져오기
style_name = ""
try:
    _sales_actual_df = load_sales_actual_df()
    _sales_actual_df.columns = [str(c).strip() for c in _sales_actual_df.columns]
    if "style_code" in _sales_actual_df.columns:
        _sales_actual_df["style_code"] = _sales_actual_df["style_code"].astype(str).str.strip()
        _sales_actual_df = _sales_actual_df[_sales_actual_df["style_code"] == selected_style]
    if "style_name" in _sales_actual_df.columns and not _sales_actual_df.empty:
        _sales_actual_df["style_name"] = _sales_actual_df["style_name"].astype(str).str.strip()
        _names = [n for n in _sales_actual_df["style_name"].dropna().unique().tolist() if n and n != "nan"]
        if _names:
            style_name = _names[0]
except Exception:
    style_name = ""

# 스타일코드 옆에 스타일명 표시(말줄임 방지)
st.markdown(
    """
<style>
.km-style-row{display:flex;gap:16px;align-items:baseline;flex-wrap:wrap}
.km-style-code{font-size:56px;font-weight:800;line-height:1.05;letter-spacing:-0.02em}
.km-style-name{font-size:44px;font-weight:700;line-height:1.1;word-break:keep-all;overflow-wrap:anywhere}
</style>
""",
    unsafe_allow_html=True,
)

st.markdown("스타일코드", help=None)
if style_name:
    st.markdown(
        f'<div class="km-style-row"><div class="km-style-code">{selected_style}</div><div class="km-style-name">{style_name}</div></div>',
        unsafe_allow_html=True,
    )
else:
    st.markdown(f'<div class="km-style-code">{selected_style}</div>', unsafe_allow_html=True)


# =========================
# 8) 최상단: 전체 vs 선택 매장 매출 추세 (오버레이)
# =========================
st.subheader("매출 추세")

sales_total_week_df = (
    df_filtered.groupby(["similar_week", "week_sort", "week_label"], as_index=False)
    .agg(
        sales_qty=("similar_forecast_qty", "sum") if "similar_forecast_qty" in df_filtered.columns else ("similar_gross_sales", "size"),
        avg_discount_rate=("similar_discount_rate", "mean") if "similar_discount_rate" in df_filtered.columns else ("similar_gross_sales", "size"),
    )
    .sort_values("week_sort")
)

week_order = sales_total_week_df.sort_values("week_sort")["week_label"].tolist()

fig_sales = go.Figure()
fig_sales.add_trace(
    go.Scatter(
        x=sales_total_week_df["week_label"],
        y=sales_total_week_df["sales_qty"],
        name="작년 전체 판매량(유사)",
        mode="lines",
        line=dict(color="rgba(120,120,120,0.55)", width=2),
        fill="tozeroy",
        fillcolor="rgba(120,120,120,0.22)",
        customdata=(
            sales_total_week_df[["sales_qty", "avg_discount_rate"]].to_numpy()
            if ("similar_forecast_qty" in df_filtered.columns and "similar_discount_rate" in df_filtered.columns)
            else (
                sales_total_week_df[["sales_qty"]].to_numpy()
                if "similar_forecast_qty" in df_filtered.columns
                else (
                    sales_total_week_df[["avg_discount_rate"]].to_numpy()
                    if "similar_discount_rate" in df_filtered.columns
                    else [[None]] * len(sales_total_week_df)
                )
            )
        ),
        hovertemplate=(
            "%{x}<br>"
            "판매수량=%{customdata[0]:,.0f}<br>"
            "전체 판매수량=%{y:,.0f}<br>"
            "평균 할인율=%{customdata[1]:.1%}"
            "<extra></extra>"
        )
        if ("similar_forecast_qty" in df_filtered.columns and "similar_discount_rate" in df_filtered.columns)
        else (
            "%{x}<br>"
            "판매수량=%{customdata[0]:,.0f}<br>"
            "전체 판매수량=%{y:,.0f}<br>"
            "평균 할인율=-"
            "<extra></extra>"
        )
        if "similar_forecast_qty" in df_filtered.columns
        else (
            "판매수량=-<br>"
            "%{x}<br>"
            "전체 판매수량=%{y:,.0f}<br>"
            "평균 할인율=%{customdata[0]:.1%}"
            "<extra></extra>"
        )
        if "similar_discount_rate" in df_filtered.columns
        else (
            "판매수량=-<br>"
            "%{x}<br>"
            "전체 판매수량=%{y:,.0f}<br>"
            "평균 할인율=-"
            "<extra></extra>"
        ),
    )
)

if selected_store != "전체" and not store_week_df.empty:
    df_store_filtered = df_filtered[df_filtered["similar_store_name"] == selected_store].copy()
    sales_store_week_df = (
        df_store_filtered.groupby(["similar_week", "week_sort", "week_label"], as_index=False)
        .agg(
            sales_qty=("similar_forecast_qty", "sum") if "similar_forecast_qty" in df_store_filtered.columns else ("similar_gross_sales", "size"),
            avg_discount_rate=("similar_discount_rate", "mean") if "similar_discount_rate" in df_store_filtered.columns else ("similar_gross_sales", "size"),
        )
        .sort_values("week_sort")
    )
    fig_sales.add_trace(
        go.Scatter(
            x=sales_store_week_df["week_label"],
            y=sales_store_week_df["sales_qty"],
            name="예측 판매량",
            mode="lines",
            line=dict(color="rgba(220,50,50,0.70)", width=2),
            fill="tozeroy",
            fillcolor="rgba(220,50,50,0.25)",
            customdata=(
                sales_store_week_df[["sales_qty", "avg_discount_rate"]].to_numpy()
                if ("similar_forecast_qty" in df_store_filtered.columns and "similar_discount_rate" in df_store_filtered.columns)
                else (
                    sales_store_week_df[["sales_qty"]].to_numpy()
                    if "similar_forecast_qty" in df_store_filtered.columns
                    else (
                        sales_store_week_df[["avg_discount_rate"]].to_numpy()
                        if "similar_discount_rate" in df_store_filtered.columns
                        else [[None]] * len(sales_store_week_df)
                    )
                )
            ),
            hovertemplate=(
                "%{x}<br>"
                "판매수량=%{customdata[0]:,.0f}<br>"
                "선택 매장 판매수량=%{y:,.0f}<br>"
                "평균 할인율=%{customdata[1]:.1%}"
                "<extra></extra>"
            )
            if ("similar_forecast_qty" in df_store_filtered.columns and "similar_discount_rate" in df_store_filtered.columns)
            else (
                "%{x}<br>"
                "판매수량=%{customdata[0]:,.0f}<br>"
                "선택 매장 판매수량=%{y:,.0f}<br>"
                "평균 할인율=-"
                "<extra></extra>"
            )
            if "similar_forecast_qty" in df_store_filtered.columns
            else (
                "판매수량=-<br>"
                "%{x}<br>"
                "선택 매장 판매수량=%{y:,.0f}<br>"
                "평균 할인율=%{customdata[0]:.1%}"
                "<extra></extra>"
            )
            if "similar_discount_rate" in df_store_filtered.columns
            else (
                "판매수량=-<br>"
                "%{x}<br>"
                "선택 매장 판매수량=%{y:,.0f}<br>"
                "평균 할인율=-"
                "<extra></extra>"
            ),
        )
    )

# 올해(sales_actual) 라인 추가: 파란색 굵은 선, 채움 없음
try:
    sa = load_sales_actual_df().copy()
    sa.columns = [str(c).strip() for c in sa.columns]

    if "week" in sa.columns:
        sa["week"] = sa["week"].astype(str).str.strip()

        # forecast_base.similar_week ↔ sales_actual.week 매핑(표시용 week_label/week_sort 부여)
        week_map_df = (
            df_filtered[["similar_week", "week_sort", "week_label"]]
            .drop_duplicates()
            .rename(columns={"similar_week": "week"})
        )
        sa_mapped = sa.merge(week_map_df, on="week", how="inner")

        # 주차가 서로 안 겹치면(예: 작년 2025-xx vs 올해 2026-xx) sales_actual 자체로 라벨/정렬 생성
        if sa_mapped.empty:
            sa_mapped = sa.copy()
            sa_mapped["week_sort"] = (
                sa_mapped["week"]
                .astype(str)
                .str.replace("-", "", regex=False)
                .str.replace(" ", "", regex=False)
            )
            sa_mapped["week_sort"] = pd.to_numeric(sa_mapped["week_sort"], errors="coerce")
            sa_mapped["week_label"] = sa_mapped["week"].apply(lambda v: clean_data(pd.DataFrame({
                "style_code": [selected_style],
                "similar_style_code": [""],
                "similar_store_code": [""],
                "similar_store_name": [""],
                "similar_week": [str(v)],
                "similar_gross_sales": [0],
            }))["week_label"].iloc[0])

        # 가능한 경우, 필터를 sales_actual에도 적용
        if "style_code" in sa_mapped.columns:
            sa_mapped["style_code"] = sa_mapped["style_code"].astype(str).str.strip()
            sa_mapped = sa_mapped[sa_mapped["style_code"] == selected_style]

        if selected_store != "전체":
            for store_col in ["store_name", "store", "store_nm", "similar_store_name", "similar_store"]:
                if store_col in sa_mapped.columns:
                    sa_mapped[store_col] = sa_mapped[store_col].astype(str).str.strip()
                    sa_mapped = sa_mapped[sa_mapped[store_col] == selected_store]
                    break

        if selected_colors is not None and "color" in sa_mapped.columns:
            sa_mapped["color"] = sa_mapped["color"].astype(str).str.strip()
            sa_mapped = sa_mapped[sa_mapped["color"].isin(selected_colors)]

        if selected_sizes is not None:
            if "size" in sa_mapped.columns:
                sa_mapped["size"] = sa_mapped["size"].astype(str).str.strip()
                sa_mapped = sa_mapped[sa_mapped["size"].isin(selected_sizes)]
            elif "similar_size" in sa_mapped.columns:
                sa_mapped["similar_size"] = sa_mapped["similar_size"].astype(str).str.strip()
                sa_mapped = sa_mapped[sa_mapped["similar_size"].isin(selected_sizes)]

        # sales_actual에서 올해 "판매량"은 sales_amount를 최우선 사용
        sales_col = "sales_amount" if "sales_amount" in sa_mapped.columns else resolve_sales_actual_sales_column(sa_mapped)
        if sales_col:
            sa_mapped[sales_col] = (
                sa_mapped[sales_col]
                .astype(str)
                .str.replace(",", "", regex=False)
                .str.replace("₩", "", regex=False)
                .str.strip()
            )
            sa_mapped[sales_col] = pd.to_numeric(sa_mapped[sales_col], errors="coerce").fillna(0)

            sa_week = (
                sa_mapped.groupby(["week_label", "week_sort"], as_index=False)[sales_col]
                .sum()
                .sort_values("week_sort")
            )

            line_name = "올해 판매량" if sales_col in ["sales_amount", "sales_qty"] else "올해 매출"
            fig_sales.add_trace(
                go.Scatter(
                    x=sa_week["week_label"],
                    y=sa_week[sales_col],
                    name=line_name,
                    mode="lines",
                    line=dict(color="rgba(30,90,220,0.95)", width=4),
                    hovertemplate=f"%{{x}}<br>{line_name}=%{{y:,.0f}}<extra></extra>",
                )
            )
except Exception:
    pass

fig_sales.update_layout(
    title=f"{selected_style} 판매수량 추세 (필터 적용됨)",
    xaxis_title="주차",
    yaxis_title="판매수량",
    height=420,
    legend_title="추세선",
    xaxis=dict(categoryorder="array", categoryarray=week_order),
)
st.plotly_chart(fig_sales, use_container_width=True)
