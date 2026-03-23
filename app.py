import os
import json
import re
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import streamlit as st
import gspread
from google.oauth2.service_account import Credentials

# =========================
# 1) 기본 설정
# =========================
st.set_page_config(
    page_title="아이템 PLC 분류 화면",
    layout="wide",
)

st.title("아이템별 PLC 분석 화면")
st.caption("주차별 전체 매출 추이와 PLC 단계(도입/성장/성숙/변곡점/쇠퇴)를 함께 보여줍니다.")

# =========================
# 1-1) 구글시트 연결
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

    raise ValueError(
        "구글 서비스 계정 정보가 없습니다. "
        "st.secrets['gcp_service_account'] 또는 GOOGLE_SERVICE_ACCOUNT_JSON을 설정하세요."
    )


def get_sheets_config() -> dict:
    """
    secrets.toml의 [sheets] 섹션을 dict로 반환합니다.
    필수 키: sheet_id
    선택 키: worksheet(기본값 forecast_base)
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

    values = ws.get_all_records()
    return pd.DataFrame(values)


def normalize_source_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    시트 컬럼명을 PLC 파이프라인 표준명으로 정규화합니다.
    - 공백/개행 제거
    - 흔한 별칭을 표준 컬럼으로 매핑
    """
    out = df.copy()

    def _clean_col_name(v: str) -> str:
        s = str(v)
        # BOM/제로폭/개행/탭 제거 + 좌우 공백 정리
        s = s.replace("\ufeff", "").replace("\u200b", "")
        s = s.replace("\n", "").replace("\r", "").replace("\t", "")
        return s.strip()

    def _canonical(v: str) -> str:
        s = _clean_col_name(v).lower()
        # 표기 흔들림(공백/구분자) 제거
        for ch in [" ", "_", "-", "/", "(", ")", "[", "]"]:
            s = s.replace(ch, "")
        return s

    out.columns = [_clean_col_name(c) for c in out.columns]

    alias_map = {
        "아이템": ["아이템", "item", "품목", "상품", "스타일", "style_code", "style"],
        "연도/주": ["연도/주", "year_week", "week", "주차", "similar_week", "연도주차"],
        "외형매출": ["외형매출", "매출", "sales_amount", "gross_sales", "actual_sales", "net_sales"],
        "정상가": ["정상가", "정가", "list_price", "original_price", "price"],
        "판매수량": ["판매수량", "수량", "판매량", "qty", "sales_qty", "similar_forecast_qty"],
    }

    rename_dict = {}
    normalized_lookup = {_canonical(c): c for c in out.columns}
    for target, aliases in alias_map.items():
        if target in out.columns:
            continue
        for alias in aliases:
            key = _canonical(alias)
            if key in normalized_lookup:
                rename_dict[normalized_lookup[key]] = target
                break

    if rename_dict:
        out = out.rename(columns=rename_dict)

    return out


def infer_and_rename_week_column(df: pd.DataFrame) -> pd.DataFrame:
    """
    연도/주 컬럼명이 비정형일 때 값 패턴(YYYY-WW)으로 주차 컬럼을 추론합니다.
    """
    out = df.copy()
    if "연도/주" in out.columns:
        return out

    week_pattern = re.compile(r"^\s*\d{4}\s*-\s*\d{1,2}\s*$")

    best_col = None
    best_hits = 0
    for col in out.columns:
        s = out[col].astype(str).str.strip()
        hits = s.map(lambda x: bool(week_pattern.match(x))).sum()
        if hits > best_hits:
            best_hits = hits
            best_col = col

    # 최소 2개 이상 주차 패턴이 확인되면 주차 컬럼으로 간주
    if best_col is not None and best_hits >= 2:
        out = out.rename(columns={best_col: "연도/주"})
    return out


def convert_wide_weekly_to_long(df: pd.DataFrame) -> pd.DataFrame:
    """
    wide 포맷(연도/주 + 아이템별 컬럼)을 long 포맷으로 변환합니다.
    예)
    연도/주 | 가디건 | 가방 ...
    ->
    연도/주 | 아이템 | 판매수량 | 외형매출 | 정상가
    """
    out = infer_and_rename_week_column(df.copy())
    if "연도/주" not in out.columns:
        return out

    id_candidates = ["연도/주", "채널", "year", "week", "yearweek_num"]
    id_vars = [c for c in id_candidates if c in out.columns]

    value_vars = [c for c in out.columns if c not in id_vars]
    if not value_vars:
        return out

    melted = out.melt(
        id_vars=id_vars,
        value_vars=value_vars,
        var_name="아이템",
        value_name="판매수량",
    )

    melted["아이템"] = melted["아이템"].astype(str).str.strip()
    melted["판매수량"] = (
        melted["판매수량"]
        .astype(str)
        .str.replace(",", "", regex=False)
        .str.replace("₩", "", regex=False)
        .str.strip()
    )
    melted["판매수량"] = pd.to_numeric(melted["판매수량"], errors="coerce").fillna(0)
    melted = melted[melted["판매수량"] > 0].copy()

    # 수량 기반 원천데이터인 경우, 기존 파이프라인 호환을 위해 기본값 구성
    melted["외형매출"] = melted["판매수량"]
    melted["정상가"] = melted["판매수량"]

    return melted


# =========================
# 2) PLC 분류 함수
# =========================
def prepare_weekly_item_data(df: pd.DataFrame) -> pd.DataFrame:
    data = normalize_source_columns(df)
    data = infer_and_rename_week_column(data)

    required_cols = ["아이템", "연도/주", "외형매출", "정상가", "판매수량"]
    missing_cols = [col for col in required_cols if col not in data.columns]

    # wide 포맷 자동 감지: 연도/주 + 다수 아이템 컬럼
    if missing_cols and "연도/주" in data.columns and "아이템" not in data.columns and "판매수량" not in data.columns:
        data = convert_wide_weekly_to_long(data)
        data = normalize_source_columns(data)
        data = infer_and_rename_week_column(data)
        missing_cols = [col for col in required_cols if col not in data.columns]

    if missing_cols:
        current_cols = [str(c) for c in data.columns]
        raise ValueError(
            f"필수 컬럼이 없습니다: {missing_cols} / 현재 컬럼: {current_cols}"
        )

    for col in ["외형매출", "정상가", "판매수량"]:
        data[col] = (
            data[col]
            .astype(str)
            .str.replace(",", "", regex=False)
            .str.strip()
        )
        data[col] = pd.to_numeric(data[col], errors="coerce").fillna(0)

    # 판매 흐름 분석용이므로 판매수량 0 초과만 사용
    data = data[data["판매수량"] > 0].copy()

    # 연도/주 분리
    data[["year", "week"]] = data["연도/주"].str.split("-", expand=True)
    data["year"] = pd.to_numeric(data["year"], errors="coerce")
    data["week"] = pd.to_numeric(data["week"], errors="coerce")
    data["yearweek_num"] = data["year"] * 100 + data["week"]

    # 아이템-주차 단위 집계
    weekly = (
        data.groupby(["아이템", "연도/주", "year", "week", "yearweek_num"], as_index=False)
        .agg({
            "외형매출": "sum",
            "정상가": "sum",
            "판매수량": "sum"
        })
        .sort_values(["아이템", "yearweek_num"])
        .reset_index(drop=True)
    )

    # 할인율 계산
    weekly["할인율"] = np.where(
        weekly["정상가"] > 0,
        1 - (weekly["외형매출"] / weekly["정상가"]),
        0
    )
    weekly["할인율"] = weekly["할인율"].clip(lower=0, upper=1)

    return weekly


def classify_plc_by_item(
    item_df: pd.DataFrame,
    intro_weeks: int = 3,
    decline_discount_threshold: float = 0.30
) -> pd.DataFrame:
    g = item_df.sort_values("yearweek_num").copy().reset_index(drop=True)

    # 이동평균 판매량
    g["판매수량_ma"] = g["판매수량"].rolling(window=3, min_periods=1).mean()

    # 이전/다음 주
    g["prev_ma"] = g["판매수량_ma"].shift(1)
    g["next_ma"] = g["판매수량_ma"].shift(-1)

    g["prev_ma"] = g["prev_ma"].fillna(g["판매수량_ma"])
    g["next_ma"] = g["next_ma"].fillna(g["판매수량_ma"])

    # 판매 시작 후 몇 번째 주인지
    g["판매주차순번"] = np.arange(1, len(g) + 1)

    # 최고점
    peak_idx = g["판매수량_ma"].idxmax()
    peak_value = g.loc[peak_idx, "판매수량_ma"]

    if peak_value == 0:
        g["peak_ratio"] = 0
    else:
        g["peak_ratio"] = g["판매수량_ma"] / peak_value

    # 증가율
    g["growth_rate"] = np.where(
        g["prev_ma"] > 0,
        (g["판매수량_ma"] - g["prev_ma"]) / g["prev_ma"],
        0
    )

    # 최고점 이후 여부
    g["is_after_peak"] = g.index > peak_idx

    plc_list = []

    for idx, row in g.iterrows():
        stage = None

        # 변곡점(최고점)
        if idx == peak_idx:
            stage = "변곡점"

        # 도입
        elif (
            row["판매주차순번"] <= intro_weeks and
            row["peak_ratio"] < 0.35
        ):
            stage = "도입"

        # 쇠퇴
        elif (
            row["is_after_peak"] and
            row["growth_rate"] < -0.05 and
            (
                row["peak_ratio"] < 0.70 or
                row["할인율"] >= decline_discount_threshold
            )
        ):
            stage = "쇠퇴"

        # 성숙
        elif (
            row["peak_ratio"] >= 0.85 and
            abs(row["growth_rate"]) <= 0.10
        ):
            stage = "성숙"

        # 성장
        elif (
            (not row["is_after_peak"]) and
            row["peak_ratio"] >= 0.35 and
            row["growth_rate"] > 0.05
        ):
            stage = "성장"

        # 보정 규칙
        else:
            if not row["is_after_peak"]:
                if row["peak_ratio"] < 0.35:
                    stage = "도입"
                elif row["peak_ratio"] < 0.85:
                    stage = "성장"
                else:
                    stage = "성숙"
            else:
                if row["peak_ratio"] >= 0.70 and row["할인율"] < decline_discount_threshold:
                    stage = "성숙"
                else:
                    stage = "쇠퇴"

        plc_list.append(stage)

    g["plc"] = plc_list
    return g


def run_plc_classification(
    df: pd.DataFrame,
    intro_weeks: int = 3,
    decline_discount_threshold: float = 0.30
) -> pd.DataFrame:
    weekly = prepare_weekly_item_data(df)

    result = (
        weekly.groupby("아이템", group_keys=False)
        .apply(
            lambda x: classify_plc_by_item(
                x,
                intro_weeks=intro_weeks,
                decline_discount_threshold=decline_discount_threshold
            )
        )
        .reset_index(drop=True)
    )

    return result


# =========================
# 3) 그래프 함수
# =========================
PLC_COLOR_MAP = {
    "도입": "#1f77b4",     # 파랑
    "성장": "#2ca02c",     # 초록
    "성숙": "#ff7f0e",     # 주황
    "변곡점": "#d62728",   # 빨강
    "쇠퇴": "#9467bd",     # 보라
}


def make_plc_chart(plot_df: pd.DataFrame, selected_item: str) -> go.Figure:
    fig = go.Figure()

    # 전체 매출 추이 선
    fig.add_trace(
        go.Scatter(
            x=plot_df["연도/주"],
            y=plot_df["외형매출"],
            mode="lines",
            name="전체 매출 추이",
            line=dict(color="#6b7280", width=3),
            hovertemplate=(
                "<b>%{x}</b><br>"
                "매출: %{y:,.0f}<extra></extra>"
            )
        )
    )

    # PLC 단계별 점
    plc_order = ["도입", "성장", "성숙", "변곡점", "쇠퇴"]

    for plc_stage in plc_order:
        stage_df = plot_df[plot_df["plc"] == plc_stage].copy()

        if stage_df.empty:
            continue

        fig.add_trace(
            go.Scatter(
                x=stage_df["연도/주"],
                y=stage_df["외형매출"],
                mode="markers",
                name=plc_stage,
                marker=dict(
                    size=11,
                    color=PLC_COLOR_MAP[plc_stage],
                    line=dict(color="white", width=1)
                ),
                hovertemplate=(
                    "<b>%{x}</b><br>"
                    f"PLC: {plc_stage}<br>"
                    "매출: %{y:,.0f}<br>"
                    "판매수량: %{customdata[0]:,.0f}<br>"
                    "할인율: %{customdata[1]:.1%}"
                    "<extra></extra>"
                ),
                customdata=stage_df[["판매수량", "할인율"]].values
            )
        )

    fig.update_layout(
        title=f"{selected_item} 주차별 매출 추이 및 PLC 단계",
        xaxis_title="연도/주",
        yaxis_title="외형매출",
        hovermode="x unified",
        legend_title="PLC 단계",
        height=550
    )

    fig.update_xaxes(type="category")

    return fig


# =========================
# 4) 데이터 불러오기
# =========================
try:
    sheets_cfg = get_sheets_config()
    worksheet_name = sheets_cfg.get("worksheet") or "forecast_base"
    raw_df = load_sheet_as_df(worksheet_name)
    raw_df = normalize_source_columns(raw_df)
    st.caption(f"데이터 소스: Google Sheets / 워크시트 `{worksheet_name}`")
except Exception as e:
    st.error("구글시트 데이터 로드 중 오류가 발생했습니다.")
    st.exception(e)
    st.stop()

# =========================
# 5) PLC 계산
# =========================
with st.spinner("PLC 단계 계산 중입니다..."):
    result_df = run_plc_classification(raw_df)

if result_df.empty:
    st.warning("분석할 데이터가 없습니다. 판매수량과 필수 컬럼을 확인하세요.")
    st.stop()

# =========================
# 6) 아이템 선택
# =========================
item_list = sorted(result_df["아이템"].dropna().unique().tolist())

selected_item = st.selectbox("아이템 선택", item_list)

item_df = (
    result_df[result_df["아이템"] == selected_item]
    .sort_values("yearweek_num")
    .copy()
)

# =========================
# 7) 요약 정보
# =========================
latest_row = item_df.iloc[-1]
peak_row = item_df.loc[item_df["외형매출"].idxmax()]

col1, col2, col3, col4 = st.columns(4)

col1.metric("현재 PLC", latest_row["plc"])
col2.metric("최근 주차", latest_row["연도/주"])
col3.metric("최근 매출", f"{latest_row['외형매출']:,.0f}")
col4.metric("최고 매출 주차", peak_row["연도/주"])

# =========================
# 8) 그래프
# =========================
fig = make_plc_chart(item_df, selected_item)
st.plotly_chart(fig, use_container_width=True)

# =========================
# 9) PLC 단계 설명
# =========================
with st.expander("PLC 단계 기준 보기"):
    st.markdown(
        """
        - **도입**: 판매 시작 후 초기 구간이며 최고점 대비 낮은 수준
        - **성장**: 최고점 이전 구간에서 판매가 증가하는 단계
        - **성숙**: 최고점 부근에서 높은 판매 수준이 유지되는 단계
        - **변곡점**: 전체 기간 중 최고 판매 수준의 주차
        - **쇠퇴**: 최고점 이후 판매 하락 또는 할인율 상승이 나타나는 단계
        """
    )

# =========================
# 10) 테이블
# =========================
st.subheader("주차별 PLC 결과")

show_cols = [
    "아이템", "연도/주", "외형매출", "판매수량",
    "할인율", "판매수량_ma", "peak_ratio", "growth_rate", "plc"
]

table_df = item_df[show_cols].copy()
table_df["할인율"] = table_df["할인율"].map(lambda x: f"{x:.1%}")
table_df["peak_ratio"] = table_df["peak_ratio"].map(lambda x: f"{x:.1%}")
table_df["growth_rate"] = table_df["growth_rate"].map(lambda x: f"{x:.1%}")

st.dataframe(table_df, use_container_width=True)
