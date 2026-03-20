import os
import json
from typing import Tuple
from datetime import date

import pandas as pd
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


def get_sheets_config() -> dict:
    if "sheets" not in st.secrets:
        raise ValueError("st.secrets['sheets'] 설정이 없습니다. secrets.toml에 [sheets] 섹션을 추가하세요.")
    return dict(st.secrets["sheets"])


@st.cache_data(ttl=300)
def load_sheet_as_df(worksheet_name: str) -> pd.DataFrame:
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


# =========================
# 3) 유틸 함수
# =========================
def clean_text(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip()


def to_numeric_safe(series: pd.Series) -> pd.Series:
    return pd.to_numeric(
        series.astype(str).str.replace(",", "", regex=False).str.replace("%", "", regex=False),
        errors="coerce"
    )


def extract_item_from_style(style_code: str) -> str:
    """
    스타일코드의 3,4번째 문자 추출
    예: SPRWG37G01 -> RW
    파이썬 인덱스 기준 [2:4]
    """
    style_code = str(style_code).strip()
    if len(style_code) < 4:
        return ""
    return style_code[2:4].upper()


def parse_year_week(week_value):
    """
    '2024-43', '2026-9' 같은 문자열을 정렬용 튜플 (year, week)로 변환
    """
    s = str(week_value).strip()
    if "-" not in s:
        return (9999, 9999)

    parts = s.split("-")
    if len(parts) != 2:
        return (9999, 9999)

    try:
        y = int(parts[0])
        w = int(parts[1])
        return (y, w)
    except:
        return (9999, 9999)


def sort_weeks(week_list):
    return sorted(week_list, key=parse_year_week)


# =========================
# 4) 데이터 로드
# =========================
sales_df = load_sheet_as_df("sales_actual")
plc_df = load_sheet_as_df("bi_item_plc")

if sales_df.empty:
    st.error("sales_actual 시트에 데이터가 없습니다.")
    st.stop()

if plc_df.empty:
    st.error("bi_item_plc 시트에 데이터가 없습니다.")
    st.stop()


# =========================
# 5) 컬럼 정리
# =========================
sales_df.columns = [str(c).strip() for c in sales_df.columns]
plc_df.columns = [str(c).strip() for c in plc_df.columns]

# sales_actual 필수 컬럼 체크
sales_required_cols = [
    "style_code", "color", "size", "store_name", "store_code"
]
missing_sales = [c for c in sales_required_cols if c not in sales_df.columns]
if missing_sales:
    st.error(f"sales_actual 시트에 필요한 컬럼이 없습니다: {missing_sales}")
    st.stop()

# bi_item_plc 필수 컬럼 체크
plc_required_cols = [
    "similar_color",
    "item",
    "similar_store_code",
    "similar_store_name",
    "similar_week",
    "similar_forecast_qty",
]
missing_plc = [c for c in plc_required_cols if c not in plc_df.columns]
if missing_plc:
    st.error(f"bi_item_plc 시트에 필요한 컬럼이 없습니다: {missing_plc}")
    st.stop()

# 문자열 컬럼 정리
for col in ["style_code", "color", "size", "store_name", "store_code"]:
    sales_df[col] = clean_text(sales_df[col])

for col in ["similar_color", "item", "similar_store_code", "similar_store_name", "similar_week"]:
    plc_df[col] = clean_text(plc_df[col])

plc_df["similar_forecast_qty_num"] = to_numeric_safe(plc_df["similar_forecast_qty"]).fillna(0)
plc_df["item"] = plc_df["item"].str.upper()


# =========================
# 6) 필터 UI
# =========================
style_list = sorted([x for x in sales_df["style_code"].dropna().unique().tolist() if x and x != "nan"])

if not style_list:
    st.error("sales_actual.style_code 에 값이 없습니다.")
    st.stop()

col1, col2, col3, col4 = st.columns(4)

with col1:
    selected_style = st.selectbox("스타일", style_list)

# 선택 스타일 기준으로 필터 후보 좁히기
sales_style_df = sales_df[sales_df["style_code"] == selected_style].copy()

store_map_df = (
    sales_style_df[["store_name", "store_code"]]
    .dropna()
    .drop_duplicates()
    .sort_values(["store_name", "store_code"])
)

store_options = ["전체"] + store_map_df["store_name"].tolist()

color_options = ["전체"] + sorted([
    x for x in sales_style_df["color"].dropna().unique().tolist()
    if x and x != "nan"
])

size_options = ["전체"] + sorted([
    x for x in sales_style_df["size"].dropna().unique().tolist()
    if x and x != "nan"
])

with col2:
    selected_store = st.selectbox("매장", store_options)

with col3:
    selected_color = st.selectbox("컬러", color_options)

with col4:
    selected_size = st.selectbox("사이즈", size_options)

st.info("사이즈 필터는 sales_actual 기준으로만 선택되며, 현재 bi_item_plc 시트에 사이즈 컬럼이 없어 그래프 계산에는 반영되지 않습니다.")


selected_store_code = None

if selected_store != "전체":
    matched_codes = (
        store_map_df.loc[store_map_df["store_name"] == selected_store, "store_code"]
        .dropna()
        .unique()
        .tolist()
    )

    if len(matched_codes) == 0:
        st.warning(f"선택한 매장 '{selected_store}'에 대응되는 store_code를 찾지 못했습니다.")
    else:
        selected_store_code = matched_codes[0]


# =========================
# 7) 그래프용 데이터 필터링
# =========================
style_item = extract_item_from_style(selected_style)

if not style_item:
    st.warning("선택한 스타일코드에서 3,4번째 문자를 추출할 수 없습니다.")
    st.stop()

base_df = plc_df[plc_df["item"] == style_item].copy()

# 컬러 필터 적용
if selected_color != "전체":
    base_df = base_df[base_df["similar_color"] == selected_color].copy()

# 회색 그래프용: 전체 매장
grey_df = (
    base_df.groupby("similar_week", as_index=False)["similar_forecast_qty_num"]
    .sum()
    .rename(columns={"similar_forecast_qty_num": "qty_raw"})
)
# 빨간 그래프용: 선택 매장
if selected_store != "전체" and selected_store_code is not None:
    red_source_df = base_df[base_df["similar_store_code"] == selected_store_code].copy()
elif selected_store != "전체" and selected_store_code is None:
    red_source_df = base_df.iloc[0:0].copy()
else:
    red_source_df = base_df.copy()

red_df = (
    red_source_df.groupby("similar_week", as_index=False)["similar_forecast_qty_num"]
    .sum()
    .rename(columns={"similar_forecast_qty_num": "qty_raw"})
)

# 주차 정렬
all_weeks = sort_weeks(list(set(grey_df["similar_week"].tolist()) | set(red_df["similar_week"].tolist())))

grey_df = grey_df.set_index("similar_week").reindex(all_weeks, fill_value=0).reset_index()
red_df = red_df.set_index("similar_week").reindex(all_weeks, fill_value=0).reset_index()

grey_df.columns = ["similar_week", "qty_raw"]
red_df.columns = ["similar_week", "qty_raw"]

grey_df["qty"] = grey_df["qty_raw"]
red_df["qty"] = red_df["qty_raw"]
# -------------------------
# 비중(%) 계산
# -------------------------
grey_total = grey_df["qty"].sum()
red_total = red_df["qty"].sum()

if grey_total > 0:
    grey_df["ratio"] = grey_df["qty"] / grey_total
else:
    grey_df["ratio"] = 0

if red_total > 0:
    red_df["ratio"] = red_df["qty"] / red_total
else:
    red_df["ratio"] = 0

# 퍼센트 표시용 컬럼
grey_df["ratio_pct"] = grey_df["ratio"] * 100
red_df["ratio_pct"] = red_df["ratio"] * 100




# =========================
# 8) 화면 표시
# =========================
st.markdown("### 스타일코드")
st.markdown(f"## {selected_style}")

st.markdown("### 매출 추세")

sub_title = f"{selected_style} 주차별 판매 비중 추세"
if selected_store != "전체":
    sub_title += f" / 매장: {selected_store}"
if selected_color != "전체":
    sub_title += f" / 컬러: {selected_color}"
if selected_size != "전체":
    sub_title += f" / 사이즈: {selected_size}"

st.markdown(f"**{sub_title}**")

fig = go.Figure()

# 회색 그래프
fig.add_trace(
    go.Scatter(
        x=grey_df["similar_week"],
        y=grey_df["ratio_pct"],
        customdata=grey_df["qty_raw"],
        mode="lines",
        name="유사상품 전체 추세",
        line=dict(color="rgba(150,150,150,1)", width=2),
        fill="tozeroy",
        fillcolor="rgba(180,180,180,0.35)",
        hovertemplate="판매량:%{customdata}장 <br>전체 비중=%{y:.2f}%<extra></extra>",
    )
)

# 빨간 그래프
fig.add_trace(
    go.Scatter(
        x=red_df["similar_week"],
        y=red_df["ratio_pct"],
        customdata=grey_df["qty_raw"],
        mode="lines",
        name="선택 매장 추세",
        line=dict(color="rgba(220,70,70,1)", width=2),
        fill="tozeroy",
        fillcolor="rgba(220,70,70,0.25)",
        hovertemplate="판매량:%{customdata}장 <br>전체 비중=%{y:.2f}%<extra></extra>",
    )
)
fig.update_layout(
    height=520,
    xaxis_title="주차",
    yaxis_title="판매 비중(%)",
    hovermode="x unified",
    legend_title="구분",
    margin=dict(l=20, r=20, t=30, b=20),
)

fig.update_xaxes(type="category", tickangle=90)

fig.update_yaxes(ticksuffix="%")  

st.plotly_chart(fig, use_container_width=True)


# =========================
# 9) 디버깅용 확인
# =========================
with st.expander("디버깅용 데이터 확인"):
    st.write("선택 스타일코드:", selected_style)
    st.write("추출 item:", style_item)
    st.write("선택 매장:", selected_store)
    st.write("선택 컬러:", selected_color)
    st.write("선택 사이즈:", selected_size)
    st.write("그래프 원천 행 수:", len(base_df))
    st.write("선택 매장코드:", selected_store_code)

    st.markdown("#### 회색 그래프 데이터")
    st.dataframe(grey_df, use_container_width=True)

    st.markdown("#### 빨간 그래프 데이터")
    st.dataframe(red_df, use_container_width=True)
