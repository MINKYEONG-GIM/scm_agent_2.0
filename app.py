import re
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import gspread
from google.oauth2.service_account import Credentials


# -------------------------------------------------
# 1. 기본 설정
# -------------------------------------------------
st.set_page_config(
    page_title="아이템 매출/재고 대시보드",
    layout="wide"
)

st.title("아이템 매출/재고 대시보드")
st.caption("final 시트 + PLC DB 기준으로 아이템별 매출/재고/주차별 PLC 추이를 보여주는 화면")


# -------------------------------------------------
# 2. 구글시트 연결
# -------------------------------------------------
def get_gspread_client():
    service_account_info = {
        "type": st.secrets["gcp_service_account"]["type"],
        "project_id": st.secrets["gcp_service_account"]["project_id"],
        "private_key_id": st.secrets["gcp_service_account"]["private_key_id"],
        "private_key": st.secrets["gcp_service_account"]["private_key"],
        "client_email": st.secrets["gcp_service_account"]["client_email"],
        "client_id": st.secrets["gcp_service_account"]["client_id"],
        "auth_uri": st.secrets["gcp_service_account"]["auth_uri"],
        "token_uri": st.secrets["gcp_service_account"]["token_uri"],
        "auth_provider_x509_cert_url": st.secrets["gcp_service_account"]["auth_provider_x509_cert_url"],
        "client_x509_cert_url": st.secrets["gcp_service_account"]["client_x509_cert_url"],
    }

    scopes = [
        "https://www.googleapis.com/auth/spreadsheets",
        "https://www.googleapis.com/auth/drive.readonly"
    ]

    credentials = Credentials.from_service_account_info(
        service_account_info,
        scopes=scopes
    )

    return gspread.authorize(credentials)


# -------------------------------------------------
# 3. 시트 로드
# -------------------------------------------------
@st.cache_data(ttl=300)
def load_sheet(sheet_name: str) -> pd.DataFrame:
    client = get_gspread_client()

    sheet_id = st.secrets["sheets"]["sheet_id"]
    spreadsheet = client.open_by_key(sheet_id)
    ws = spreadsheet.worksheet(sheet_name)

    values = ws.get_all_values()

    if not values or len(values) < 2:
        return pd.DataFrame()

    df = pd.DataFrame(values[1:], columns=values[0])
    return df


@st.cache_data(ttl=300)
def load_final_sheet() -> pd.DataFrame:
    final_sheet = st.secrets["sheets"]["final"]
    return load_sheet(final_sheet)


@st.cache_data(ttl=300)
def load_plc_sheet() -> pd.DataFrame:
    plc_sheet_name = st.secrets["sheets"]["PLC DB"]
    return load_sheet(plc_sheet_name)


# -------------------------------------------------
# 4. 공통 유틸
# -------------------------------------------------
def to_numeric_safe(series):
    return pd.to_numeric(
        series.astype(str).str.replace(",", "", regex=False).str.strip(),
        errors="coerce"
    ).fillna(0)


def clean_item_name(name: str) -> str:
    """
    sku_name에서 색상/사이즈 제거
    예:
    [코튼스판] 캡소매 절개 반팔티, (19)Black, S
    -> [코튼스판] 캡소매 절개 반팔티
    """
    if pd.isna(name):
        return ""

    name = str(name).strip()
    parts = [p.strip() for p in name.split(",")]

    if len(parts) >= 3:
        return parts[0]

    return name


def normalize_text_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    return df


def find_first_existing_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for col in candidates:
        if col in df.columns:
            return col
    return None


def extract_item_code_from_code(code) -> str:
    """
    스타일코드 / SKU 코드의 3, 4번째 자리 추출
    예:
    ABCK1234 -> CK
    0123CK99 -> 23  (문자열 기준 3,4번째)
    주의: 사용자가 말한 기준대로 '문자열의 3,4번째'를 그대로 사용
    """
    if pd.isna(code):
        return ""

    code = str(code).strip()
    if len(code) < 4:
        return ""

    return code[2:4].upper()


def stock_status(cover_weeks):
    if cover_weeks <= 4:
        return "부족"
    elif cover_weeks <= 8:
        return "적정"
    else:
        return "과다"


# -------------------------------------------------
# 5. final 전처리
# -------------------------------------------------
def preprocess_final_data(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    df = normalize_text_columns(df)

    required_cols = [
        "sku", "sku_name", "날짜", "기초재고", "판매량",
        "분배량", "출고량(회전 등)", "로스", "판매가능주차"
    ]

    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        st.error(f"final 시트 필수 컬럼이 없습니다: {missing_cols}")
        st.stop()

    # 스타일코드 후보 컬럼 찾기
    style_code_col = find_first_existing_column(
        df,
        ["스타일코드", "style_code", "style_cd", "style", "품번", "스타일 코드"]
    )

    # 날짜 변환
    df["날짜"] = pd.to_datetime(df["날짜"], errors="coerce")
    df = df.dropna(subset=["날짜"]).copy()

    # 숫자 컬럼 변환
    numeric_cols = ["기초재고", "판매량", "분배량", "출고량(회전 등)", "로스", "판매가능주차"]
    for col in numeric_cols:
        df[col] = to_numeric_safe(df[col])

    # 대표 아이템명
    df["item_name"] = df["sku_name"].apply(clean_item_name)

    # 매칭용 코드 만들기
    df["sku_item_code"] = df["sku"].apply(extract_item_code_from_code)

    if style_code_col:
        df["style_item_code"] = df[style_code_col].apply(extract_item_code_from_code)
        # 스타일코드가 있으면 스타일코드 우선, 없으면 sku 기준
        df["match_item_code"] = df["style_item_code"].where(
            df["style_item_code"].astype(str).str.len() > 0,
            df["sku_item_code"]
        )
    else:
        df["style_item_code"] = ""
        df["match_item_code"] = df["sku_item_code"]

    df["match_item_code"] = df["match_item_code"].astype(str).str.upper().str.strip()

    # 추정 기말재고
    df["추정기말재고"] = (
        df["기초재고"] + df["분배량"] - df["판매량"] - df["출고량(회전 등)"] - df["로스"]
    )

    # 판매율
    df["판매율"] = df.apply(
        lambda row: (row["판매량"] / row["기초재고"] * 100) if row["기초재고"] > 0 else 0,
        axis=1
    )

    # 재고 상태
    df["재고상태"] = df["판매가능주차"].apply(stock_status)

    return df


# -------------------------------------------------
# 6. PLC DB 전처리
# -------------------------------------------------
def preprocess_plc_data(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    return:
      plc_wide_df : 원본형
      plc_long_df : 주차별 long 형태
    """
    if df.empty:
        return pd.DataFrame(), pd.DataFrame()

    df = normalize_text_columns(df)

    required_cols = ["아이템명", "아이템코드"]
    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        st.error(f"PLC DB 필수 컬럼이 없습니다: {missing_cols}")
        st.stop()

    df = df.copy()
    df["아이템명"] = df["아이템명"].astype(str).str.strip()
    df["아이템코드"] = df["아이템코드"].astype(str).str.upper().str.strip()

    # 주차 컬럼 찾기: 2025-01, 2025-02 같은 형태
    week_cols = [c for c in df.columns if re.fullmatch(r"\d{4}-\d{2}", str(c).strip())]

    if not week_cols:
        st.error("PLC DB에서 주차 컬럼(예: 2025-01)을 찾지 못했습니다.")
        st.stop()

    for col in week_cols:
        df[col] = to_numeric_safe(df[col])

    plc_long = df.melt(
        id_vars=["아이템명", "아이템코드"],
        value_vars=week_cols,
        var_name="연도주차",
        value_name="plc_판매량"
    )

    plc_long["연도"] = plc_long["연도주차"].str[:4]
    plc_long["주차"] = plc_long["연도주차"].str[-2:].astype(int)

    return df, plc_long


# -------------------------------------------------
# 7. 데이터 로드
# -------------------------------------------------
final_raw_df = load_final_sheet()
plc_raw_df = load_plc_sheet()

final_df = preprocess_final_data(final_raw_df)
plc_wide_df, plc_long_df = preprocess_plc_data(plc_raw_df)

if final_df.empty:
    st.warning("final 시트 데이터가 없습니다.")
    st.stop()

if plc_wide_df.empty or plc_long_df.empty:
    st.warning("PLC DB 데이터가 없습니다.")
    st.stop()


# -------------------------------------------------
# 8. 사이드바 필터
# -------------------------------------------------
st.sidebar.header("필터")

view_level = st.sidebar.radio(
    "보기 기준",
    ["대표 아이템", "SKU"],
    horizontal=False
)

date_min = final_df["날짜"].min().date()
date_max = final_df["날짜"].max().date()

date_range = st.sidebar.date_input(
    "기간 선택",
    value=(date_min, date_max),
    min_value=date_min,
    max_value=date_max
)

if isinstance(date_range, tuple) and len(date_range) == 2:
    start_date, end_date = date_range
else:
    start_date, end_date = date_min, date_max

filtered_df = final_df[
    (final_df["날짜"].dt.date >= start_date) &
    (final_df["날짜"].dt.date <= end_date)
].copy()

if view_level == "대표 아이템":
    item_options = sorted(filtered_df["item_name"].dropna().unique().tolist())
    selected_item = st.sidebar.selectbox("아이템 선택", item_options)
    view_df = filtered_df[filtered_df["item_name"] == selected_item].copy()
    title_name = selected_item
else:
    sku_options = sorted(filtered_df["sku"].dropna().unique().tolist())
    selected_sku = st.sidebar.selectbox("SKU 선택", sku_options)
    view_df = filtered_df[filtered_df["sku"] == selected_sku].copy()
    title_name = selected_sku

if view_df.empty:
    st.warning("선택한 조건에 해당하는 데이터가 없습니다.")
    st.stop()


# -------------------------------------------------
# 9. final 집계
# -------------------------------------------------
if view_level == "대표 아이템":
    agg_df = (
        view_df.groupby("날짜", as_index=False)
        .agg({
            "기초재고": "sum",
            "판매량": "sum",
            "분배량": "sum",
            "출고량(회전 등)": "sum",
            "로스": "sum",
            "판매가능주차": "mean",
            "추정기말재고": "sum"
        })
        .sort_values("날짜")
    )
else:
    agg_df = view_df.sort_values("날짜").copy()

latest_row = agg_df.sort_values("날짜").iloc[-1]
total_sales = agg_df["판매량"].sum()
total_loss = agg_df["로스"].sum()
avg_cover_weeks = agg_df["판매가능주차"].mean()


# -------------------------------------------------
# 10. 선택 항목과 PLC DB 매칭
# -------------------------------------------------
matched_codes = (
    view_df["match_item_code"]
    .dropna()
    .astype(str)
    .str.strip()
    .str.upper()
)
matched_codes = [code for code in matched_codes.unique().tolist() if code]

plc_view_df = plc_long_df[plc_long_df["아이템코드"].isin(matched_codes)].copy()

if not plc_view_df.empty:
    plc_chart_df = (
        plc_view_df.groupby(["연도주차"], as_index=False)["plc_판매량"]
        .sum()
        .sort_values("연도주차")
    )
else:
    plc_chart_df = pd.DataFrame(columns=["연도주차", "plc_판매량"])

matched_plc_items_df = (
    plc_wide_df[plc_wide_df["아이템코드"].isin(matched_codes)][["아이템명", "아이템코드"]]
    .drop_duplicates()
    .sort_values(["아이템코드", "아이템명"])
)


# -------------------------------------------------
# 11. 상단 KPI
# -------------------------------------------------
st.subheader(f"{title_name}")

col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("최근 판매량", f"{latest_row['판매량']:.0f}")
col2.metric("최근 기초재고", f"{latest_row['기초재고']:.0f}")
col3.metric("누적 판매량", f"{total_sales:.0f}")
col4.metric("평균 재고보유주수", f"{avg_cover_weeks:.1f}")
col5.metric("누적 로스", f"{total_loss:.0f}")

with st.expander("PLC 매칭 정보 보기", expanded=True):
    st.write("final 기준 매칭 코드:", ", ".join(matched_codes) if matched_codes else "-")

    if not matched_plc_items_df.empty:
        st.dataframe(matched_plc_items_df, use_container_width=True, hide_index=True)
    else:
        st.warning("PLC DB에서 매칭되는 아이템코드를 찾지 못했습니다.")


# -------------------------------------------------
# 12. 메인 차트
# -------------------------------------------------
left_col, right_col = st.columns([2, 1])

with left_col:
    st.markdown("### final 기준 매출/재고 추이")

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=agg_df["날짜"],
        y=agg_df["판매량"],
        mode="lines+markers",
        name="판매량"
    ))

    fig.add_trace(go.Scatter(
        x=agg_df["날짜"],
        y=agg_df["기초재고"],
        mode="lines+markers",
        name="기초재고",
        yaxis="y2"
    ))

    fig.update_layout(
        height=500,
        hovermode="x unified",
        xaxis=dict(title="날짜"),
        yaxis=dict(title="판매량"),
        yaxis2=dict(
            title="기초재고",
            overlaying="y",
            side="right"
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        margin=dict(l=20, r=20, t=20, b=20)
    )

    st.plotly_chart(fig, use_container_width=True)

with right_col:
    st.markdown("### 최근 상태")

    latest_status_df = pd.DataFrame({
        "지표": ["기초재고", "판매량", "분배량", "출고량", "로스", "재고보유주수", "추정기말재고"],
        "값": [
            latest_row["기초재고"],
            latest_row["판매량"],
            latest_row["분배량"],
            latest_row["출고량(회전 등)"],
            latest_row["로스"],
            round(latest_row["판매가능주차"], 1),
            latest_row["추정기말재고"],
        ]
    })

    st.dataframe(latest_status_df, use_container_width=True, hide_index=True)


# -------------------------------------------------
# 13. PLC DB 주차별 추이 차트
# -------------------------------------------------
st.markdown("### PLC DB 기준 주차별 매출 추이")

if plc_chart_df.empty:
    st.info("선택한 항목과 매칭되는 PLC DB 데이터가 없습니다.")
else:
    fig_plc = px.line(
        plc_chart_df,
        x="연도주차",
        y="plc_판매량",
        markers=True
    )
    fig_plc.update_layout(
        height=400,
        margin=dict(l=20, r=20, t=20, b=20),
        xaxis_title="연도-주차",
        yaxis_title="PLC 판매량"
    )
    st.plotly_chart(fig_plc, use_container_width=True)


# -------------------------------------------------
# 14. 보조 차트
# -------------------------------------------------
c1, c2 = st.columns(2)

with c1:
    st.markdown("### 흐름 비교")

    flow_df = agg_df.melt(
        id_vars="날짜",
        value_vars=["판매량", "분배량", "출고량(회전 등)", "로스"],
        var_name="구분",
        value_name="수량"
    )

    fig_flow = px.bar(
        flow_df,
        x="날짜",
        y="수량",
        color="구분",
        barmode="group"
    )
    fig_flow.update_layout(height=400, margin=dict(l=20, r=20, t=20, b=20))
    st.plotly_chart(fig_flow, use_container_width=True)

with c2:
    st.markdown("### 재고보유주수 추이")

    fig_cover = px.line(
        agg_df,
        x="날짜",
        y="판매가능주차",
        markers=True
    )
    fig_cover.update_layout(height=400, margin=dict(l=20, r=20, t=20, b=20))
    st.plotly_chart(fig_cover, use_container_width=True)


# -------------------------------------------------
# 15. 상세 테이블
# -------------------------------------------------
st.markdown("### 상세 데이터")

display_df = view_df.copy()

display_cols = [
    "sku", "sku_name", "item_name",
    "style_item_code", "sku_item_code", "match_item_code",
    "날짜", "기초재고", "판매량", "분배량", "출고량(회전 등)",
    "로스", "판매가능주차", "추정기말재고", "판매율", "재고상태"
]

display_cols = [c for c in display_cols if c in display_df.columns]

display_df = display_df[display_cols].sort_values(
    ["날짜", "sku"],
    ascending=[False, True]
)

display_df["날짜"] = display_df["날짜"].dt.strftime("%Y-%m-%d")
if "판매율" in display_df.columns:
    display_df["판매율"] = display_df["판매율"].round(1)

st.dataframe(
    display_df,
    use_container_width=True,
    hide_index=True,
    height=500
)


# -------------------------------------------------
# 16. 다운로드
# -------------------------------------------------
csv_data = display_df.to_csv(index=False).encode("utf-8-sig")

st.download_button(
    label="현재 화면 데이터 CSV 다운로드",
    data=csv_data,
    file_name="item_dashboard_export.csv",
    mime="text/csv"
)
