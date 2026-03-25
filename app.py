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
st.caption("final 시트 기준으로 아이템별 매출 추이와 재고 지표를 보기 좋게 보여주는 화면")


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
# final 시트 로드
# -------------------------------------------------

@st.cache_data(ttl=300)
def load_final_sheet():

    client = get_gspread_client()

    sheet_id = st.secrets["sheets"]["sheet_id"]
    final_sheet_name = st.secrets["sheets"]["final"]

    spreadsheet = client.open_by_key(sheet_id)

    final_ws = spreadsheet.worksheet(final_sheet_name)

    values = final_ws.get_all_values()

    if not values or len(values) < 2:
        return pd.DataFrame()

    final_df = pd.DataFrame(
        values[1:],
        columns=values[0]
    )

    return final_df

# -------------------------------------------------
# 3. 전처리
# -------------------------------------------------
def clean_item_name(name: str) -> str:
    """
    sku_name에서 색상/사이즈를 제거해서 대표 아이템명 추출
    예:
    [코튼스판] 캡소매 절개 반팔티, (19)Black, S
    -> [코튼스판] 캡소매 절개 반팔티
    """
    if pd.isna(name):
        return ""

    name = str(name).strip()

    # 마지막 ", 색상, 사이즈" 형태 제거
    parts = [p.strip() for p in name.split(",")]
    if len(parts) >= 3:
        return parts[0]

    return name


def to_numeric_safe(series):
    return pd.to_numeric(series.astype(str).str.replace(",", "").str.strip(), errors="coerce").fillna(0)


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]

    # 컬럼명 표준화
    rename_map = {}
    for col in df.columns:
        clean_col = col.strip()
        rename_map[col] = clean_col
    df = df.rename(columns=rename_map)

    required_cols = [
        "sku", "sku_name", "날짜", "기초재고", "판매량",
        "분배량", "출고량(회전 등)", "로스", "판매가능주차"
    ]

    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        st.error(f"필수 컬럼이 없습니다: {missing_cols}")
        st.stop()

    # 날짜 변환
    df["날짜"] = pd.to_datetime(df["날짜"], errors="coerce")
    df = df.dropna(subset=["날짜"]).copy()

    # 숫자 컬럼 변환
    numeric_cols = ["기초재고", "판매량", "분배량", "출고량(회전 등)", "로스", "판매가능주차"]
    for col in numeric_cols:
        df[col] = to_numeric_safe(df[col])

    # 대표 아이템명
    df["item_name"] = df["sku_name"].apply(clean_item_name)

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
    def stock_status(cover_weeks):
        if cover_weeks <= 4:
            return "부족"
        elif cover_weeks <= 8:
            return "적정"
        else:
            return "과다"

    df["재고상태"] = df["판매가능주차"].apply(stock_status)

    return df


# -------------------------------------------------
# 4. 데이터 로드
# -------------------------------------------------
raw_df = load_final_data()
df = preprocess_data(raw_df)

if df.empty:
    st.warning("불러온 데이터가 없습니다.")
    st.stop()


# -------------------------------------------------
# 5. 사이드바 필터
# -------------------------------------------------
st.sidebar.header("필터")

view_level = st.sidebar.radio(
    "보기 기준",
    ["대표 아이템", "SKU"],
    horizontal=False
)

date_min = df["날짜"].min().date()
date_max = df["날짜"].max().date()

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

filtered_df = df[
    (df["날짜"].dt.date >= start_date) &
    (df["날짜"].dt.date <= end_date)
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
# 6. 집계 테이블 생성
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
avg_sales = agg_df["판매량"].mean()


# -------------------------------------------------
# 7. 상단 KPI
# -------------------------------------------------
st.subheader(f"{title_name}")

col1, col2, col3, col4, col5 = st.columns(5)

col1.metric("최근 판매량", f"{latest_row['판매량']:.0f}")
col2.metric("최근 기초재고", f"{latest_row['기초재고']:.0f}")
col3.metric("누적 판매량", f"{total_sales:.0f}")
col4.metric("평균 재고보유주수", f"{avg_cover_weeks:.1f}")
col5.metric("누적 로스", f"{total_loss:.0f}")


# -------------------------------------------------
# 8. 메인 차트
# -------------------------------------------------
left_col, right_col = st.columns([2, 1])

with left_col:
    st.markdown("### 매출/재고 추이")

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
# 9. 보조 차트
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
# 10. 상세 테이블
# -------------------------------------------------
st.markdown("### 상세 데이터")

display_df = view_df.copy()

display_cols = [
    "sku", "sku_name", "item_name", "날짜", "기초재고", "판매량",
    "분배량", "출고량(회전 등)", "로스", "판매가능주차",
    "추정기말재고", "판매율", "재고상태"
]

display_df = display_df[display_cols].sort_values(["날짜", "sku"], ascending=[False, True])

display_df["날짜"] = display_df["날짜"].dt.strftime("%Y-%m-%d")
display_df["판매율"] = display_df["판매율"].round(1)

st.dataframe(
    display_df,
    use_container_width=True,
    hide_index=True,
    height=500
)


# -------------------------------------------------
# 11. 다운로드
# -------------------------------------------------
csv_data = display_df.to_csv(index=False).encode("utf-8-sig")

st.download_button(
    label="현재 화면 데이터 CSV 다운로드",
    data=csv_data,
    file_name="item_dashboard_export.csv",
    mime="text/csv"
)
