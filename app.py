import re
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import gspread
from google.oauth2.service_account import Credentials

# -------------------------------------------------
# 1. 기본 설정
# -------------------------------------------------
st.set_page_config(page_title="아이템 주차별 판매 예측", layout="wide")
st.title("아이템 시즌 / PLC / 다음 주 판매량 예측")


# -------------------------------------------------
# 2. 구글시트 연결
# -------------------------------------------------
@st.cache_resource
def get_gsheet_client():
    scope = [
        "https://www.googleapis.com/auth/spreadsheets",
        "https://www.googleapis.com/auth/drive",
    ]

    credentials = Credentials.from_service_account_info(
        st.secrets["gcp_service_account"],
        scopes=scope,
    )
    return gspread.authorize(credentials)


def _sheets_section():
    if "sheets" not in st.secrets:
        raise ValueError("secrets.toml에 [sheets] 섹션이 없습니다.")
    return st.secrets["sheets"]


# -------------------------------------------------
# 3. 워크시트 읽기
# -------------------------------------------------
def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    return df


def _drop_empty_rows_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df.replace(r"^\s*$", np.nan, regex=True)
    df = df.dropna(axis=0, how="all")
    df = df.dropna(axis=1, how="all")
    return df


def read_worksheet_as_df(spreadsheet, worksheet_name: str) -> pd.DataFrame:
    ws = spreadsheet.worksheet(worksheet_name)
    values = ws.get_all_values()

    if not values:
        raise ValueError(f"'{worksheet_name}' 시트가 비어 있습니다.")

    # 첫 번째 완전 비어있지 않은 행을 헤더로 사용
    header_row_idx = None
    for i, row in enumerate(values):
        if any(str(x).strip() != "" for x in row):
            header_row_idx = i
            break

    if header_row_idx is None:
        raise ValueError(f"'{worksheet_name}' 시트에서 헤더를 찾지 못했습니다.")

    header = [str(x).strip() for x in values[header_row_idx]]
    data = values[header_row_idx + 1:]

    # 길이 맞추기
    max_len = len(header)
    fixed_data = []
    for row in data:
        row = row[:max_len] + [""] * (max_len - len(row))
        fixed_data.append(row)

    df = pd.DataFrame(fixed_data, columns=header)
    df = _normalize_columns(df)
    df = _drop_empty_rows_cols(df)
    return df


@st.cache_data(show_spinner=False)
def load_data_from_gsheet():
    client = get_gsheet_client()
    sheets = _sheets_section()

    sheet_url = str(sheets.get("SHEET_URL") or "").strip()
    sheet_id = str(sheets.get("sheet_id") or "").strip()

    if sheet_url:
        spreadsheet = client.open_by_url(sheet_url)
    elif sheet_id:
        spreadsheet = client.open_by_key(sheet_id)
    else:
        raise ValueError("[sheets]에 SHEET_URL 또는 sheet_id 중 하나가 필요합니다.")

    final_sheet_name = str(sheets.get("final") or "final").strip()
    final_df = read_worksheet_as_df(spreadsheet, final_sheet_name)

    return final_df


# -------------------------------------------------
# 4. 컬럼 자동 탐지
# -------------------------------------------------
def find_first_matching_column(df: pd.DataFrame, candidates):
    lower_map = {str(col).strip().lower(): col for col in df.columns}
    for cand in candidates:
        if cand.lower() in lower_map:
            return lower_map[cand.lower()]
    return None


def get_required_columns(df: pd.DataFrame):
    date_col = find_first_matching_column(df, ["날짜", "date", "기준일", "판매일"])
    qty_col = find_first_matching_column(df, ["판매량", "sales_qty", "qty", "수량"])
    sku_name_col = find_first_matching_column(df, ["sku_name", "상품명", "품명", "아이템", "item_name"])
    sku_col = find_first_matching_column(df, ["sku", "품번", "스타일", "style"])

    missing = []
    if date_col is None:
        missing.append("날짜")
    if qty_col is None:
        missing.append("판매량")
    if sku_name_col is None:
        missing.append("sku_name 또는 아이템명")

    if missing:
        raise ValueError(f"필수 컬럼을 찾지 못했습니다: {missing}")

    return {
        "date_col": date_col,
        "qty_col": qty_col,
        "sku_name_col": sku_name_col,
        "sku_col": sku_col,
    }


# -------------------------------------------------
# 5. 아이템명 정리
# -------------------------------------------------
def normalize_item_name(name: str) -> str:
    """
    예:
    [코튼스판] 캡소매 절개 반팔티, (19)Black, S
    -> [코튼스판] 캡소매 절개 반팔티
    """
    if pd.isna(name):
        return ""

    text = str(name).strip()

    # 첫 번째 쉼표 앞까지만 아이템명으로 사용
    text = text.split(",")[0].strip()

    # 공백 정리
    text = re.sub(r"\s+", " ", text)
    return text


# -------------------------------------------------
# 6. 전처리
# -------------------------------------------------
def preprocess_final_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = _normalize_columns(df)
    col_map = get_required_columns(df)

    date_col = col_map["date_col"]
    qty_col = col_map["qty_col"]
    sku_name_col = col_map["sku_name_col"]

    # 날짜 변환
    df["date"] = pd.to_datetime(df[date_col], errors="coerce")
    if df["date"].isna().all():
        # 예: 2026. 2. 25 형식 대응
        df["date"] = pd.to_datetime(df[date_col].astype(str).str.replace(".", "-", regex=False), errors="coerce")

    # 판매량 변환
    df["sales_qty"] = (
        df[qty_col]
        .astype(str)
        .str.replace(",", "", regex=False)
        .str.strip()
        .replace("", "0")
    )
    df["sales_qty"] = pd.to_numeric(df["sales_qty"], errors="coerce").fillna(0)

    # 아이템명 정리
    df["item_name"] = df[sku_name_col].astype(str).apply(normalize_item_name)

    # 유효 데이터만 남기기
    df = df.dropna(subset=["date"])
    df = df[df["item_name"].astype(str).str.strip() != ""].copy()

    # 주차 계산
    iso = df["date"].dt.isocalendar()
    df["year"] = iso["year"].astype(int)
    df["week"] = iso["week"].astype(int)
    df["year_week"] = df["year"].astype(str) + "-" + df["week"].astype(str).str.zfill(2)

    return df


# -------------------------------------------------
# 7. 시즌 분류
# -------------------------------------------------
def get_season_from_week(week: int) -> str:
    if 9 <= week <= 18:
        return "SPRING"
    elif 19 <= week <= 30:
        return "SUMMER"
    elif 31 <= week <= 40:
        return "FALL"
    else:
        return "WINTER"


def classify_item_season(item_weekly_df: pd.DataFrame) -> str:
    temp = item_weekly_df.copy()
    temp["season"] = temp["week"].apply(get_season_from_week)

    season_sum = temp.groupby("season", as_index=False)["sales_qty"].sum()
    season_map = dict(zip(season_sum["season"], season_sum["sales_qty"]))

    spring = float(season_map.get("SPRING", 0))
    summer = float(season_map.get("SUMMER", 0))
    fall = float(season_map.get("FALL", 0))
    winter = float(season_map.get("WINTER", 0))

    total = spring + summer + fall + winter
    if total <= 0:
        return "ALL_SEASON"

    spring_ratio = spring / total
    summer_ratio = summer / total
    fall_ratio = fall / total
    winter_ratio = winter / total

    if summer_ratio >= 0.40:
        return "SUMMER_PEAK"
    if winter_ratio >= 0.40:
        return "WINTER_PEAK"
    if (spring_ratio + fall_ratio) >= 0.60:
        return "SPRING_FALL_PEAK"
    if spring_ratio >= 0.35:
        return "SPRING_PEAK"
    if fall_ratio >= 0.35:
        return "FALL_PEAK"
    return "ALL_SEASON"


# -------------------------------------------------
# 8. PLC 분류
# -------------------------------------------------
def classify_item_plc(item_weekly_df: pd.DataFrame) -> str:
    temp = item_weekly_df.sort_values(["year", "week"]).reset_index(drop=True).copy()
    y = temp["sales_qty"].values.astype(float)

    if len(y) < 4:
        return "판단불가"

    peak_idx = int(np.argmax(y))
    last_idx = len(y) - 1

    # 최근 3주 평균과 직전 3주 평균 비교
    recent_n = min(3, len(y))
    prev_n = min(3, max(len(y) - recent_n, 1))

    recent_mean = float(np.mean(y[-recent_n:]))
    prev_mean = float(np.mean(y[-(recent_n + prev_n):-recent_n])) if len(y) > recent_n else recent_mean

    # 최대값 대비 최근 수준
    peak_value = float(np.max(y)) if np.max(y) > 0 else 1.0
    recent_ratio_to_peak = recent_mean / peak_value

    # 간단한 PLC 규칙
    if peak_idx >= len(y) - 2:
        if recent_mean >= prev_mean:
            return "성장"
        return "성숙"

    if peak_idx <= 1 and recent_ratio_to_peak < 0.5 and recent_mean < prev_mean:
        return "쇠퇴"

    if last_idx <= 2:
        return "도입"

    if recent_mean > prev_mean * 1.1 and peak_idx >= len(y) - 3:
        return "성장"

    if recent_ratio_to_peak >= 0.75:
        return "성숙"

    if recent_mean < prev_mean * 0.9:
        return "쇠퇴"

    return "성숙"


# -------------------------------------------------
# 9. 주차별 집계
# -------------------------------------------------
def make_weekly_item_sales(df: pd.DataFrame) -> pd.DataFrame:
    weekly = (
        df.groupby(["item_name", "year", "week", "year_week"], as_index=False)["sales_qty"]
        .sum()
        .sort_values(["item_name", "year", "week"])
        .reset_index(drop=True)
    )
    return weekly


# -------------------------------------------------
# 10. 예측 로직
# -------------------------------------------------
def season_factor_for_next_week(item_season: str, next_week: int) -> float:
    next_season = get_season_from_week(next_week)

    if item_season == "SUMMER_PEAK":
        return 1.15 if next_season == "SUMMER" else 0.92
    if item_season == "WINTER_PEAK":
        return 1.15 if next_season == "WINTER" else 0.92
    if item_season == "SPRING_PEAK":
        return 1.12 if next_season == "SPRING" else 0.94
    if item_season == "FALL_PEAK":
        return 1.12 if next_season == "FALL" else 0.94
    if item_season == "SPRING_FALL_PEAK":
        return 1.10 if next_season in ["SPRING", "FALL"] else 0.95
    return 1.00


def plc_factor(item_plc: str) -> float:
    if item_plc == "도입":
        return 1.05
    if item_plc == "성장":
        return 1.12
    if item_plc == "성숙":
        return 1.00
    if item_plc == "쇠퇴":
        return 0.88
    return 1.00


def predict_next_week_sales(item_weekly_df: pd.DataFrame, item_season: str, item_plc: str):
    temp = item_weekly_df.sort_values(["year", "week"]).reset_index(drop=True).copy()
    y = temp["sales_qty"].values.astype(float)

    if len(y) == 0:
        return 0.0, None

    if len(y) == 1:
        base = y[-1]
    elif len(y) == 2:
        base = y[-1] * 0.7 + y[-2] * 0.3
    else:
        # 최근값 가중 평균
        recent = y[-3:]
        weights = np.array([0.2, 0.3, 0.5])
        base = float(np.sum(recent * weights))

    # 최근 추세 반영
    if len(y) >= 2 and y[-2] > 0:
        trend_ratio = y[-1] / y[-2]
        trend_ratio = float(np.clip(trend_ratio, 0.7, 1.3))
    else:
        trend_ratio = 1.0

    last_year = int(temp.iloc[-1]["year"])
    last_week = int(temp.iloc[-1]["week"])

    next_week = last_week + 1
    next_year = last_year
    if next_week > 52:
        next_week = 1
        next_year += 1

    s_factor = season_factor_for_next_week(item_season, next_week)
    p_factor = plc_factor(item_plc)

    pred = base * trend_ratio * s_factor * p_factor
    pred = max(0, round(pred, 1))

    next_year_week = f"{next_year}-{str(next_week).zfill(2)}"
    return pred, next_year_week


# -------------------------------------------------
# 11. 아이템 요약 테이블 생성
# -------------------------------------------------
def make_item_summary(weekly_df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for item_name, g in weekly_df.groupby("item_name"):
        g = g.sort_values(["year", "week"]).reset_index(drop=True)
        item_season = classify_item_season(g)
        item_plc = classify_item_plc(g)
        pred, next_yw = predict_next_week_sales(g, item_season, item_plc)

        rows.append({
            "item_name": item_name,
            "season_type": item_season,
            "plc_type": item_plc,
            "last_week_sales": float(g.iloc[-1]["sales_qty"]),
            "pred_next_week_sales": pred,
            "pred_next_year_week": next_yw,
            "weeks_count": len(g),
        })

    summary = pd.DataFrame(rows)
    if not summary.empty:
        summary = summary.sort_values(["pred_next_week_sales", "item_name"], ascending=[False, True]).reset_index(drop=True)
    return summary


# -------------------------------------------------
# 12. 그래프
# -------------------------------------------------
def draw_item_chart(item_weekly_df: pd.DataFrame, pred_value: float, pred_yw: str):
    temp = item_weekly_df.sort_values(["year", "week"]).reset_index(drop=True).copy()

    x_actual = temp["year_week"].tolist()
    y_actual = temp["sales_qty"].tolist()

    x_all = x_actual + [pred_yw]
    y_all = y_actual + [pred_value]

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=x_actual,
            y=y_actual,
            mode="lines+markers",
            name="실제 판매량"
        )
    )

    if len(x_actual) > 0:
        fig.add_trace(
            go.Scatter(
                x=[x_actual[-1], pred_yw],
                y=[y_actual[-1], pred_value],
                mode="lines+markers",
                name="다음 주 예측"
            )
        )

    fig.update_layout(
        title="주차별 판매량 및 다음 주 예측",
        xaxis_title="연도-주차",
        yaxis_title="판매량",
        hovermode="x unified",
        height=500
    )
    return fig


# -------------------------------------------------
# 13. 실행
# -------------------------------------------------
try:
    with st.spinner("final 시트 데이터를 불러오는 중..."):
        raw_df = load_data_from_gsheet()
        df = preprocess_final_data(raw_df)
        weekly_df = make_weekly_item_sales(df)
        summary_df = make_item_summary(weekly_df)

    st.success("데이터를 정상적으로 불러왔습니다.")

    if weekly_df.empty:
        st.warning("표시할 데이터가 없습니다.")
        st.stop()

    st.subheader("아이템 예측 요약")
    st.dataframe(summary_df, use_container_width=True)

    item_list = summary_df["item_name"].dropna().unique().tolist()
    selected_item = st.selectbox("아이템 선택", item_list)

    selected_weekly = weekly_df[weekly_df["item_name"] == selected_item].copy()
    selected_summary = summary_df[summary_df["item_name"] == selected_item].iloc[0]

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("시즌 분류", selected_summary["season_type"])
    col2.metric("PLC 분류", selected_summary["plc_type"])
    col3.metric("최근 주 판매량", f"{selected_summary['last_week_sales']:.1f}")
    col4.metric(
        f"예측 판매량 ({selected_summary['pred_next_year_week']})",
        f"{selected_summary['pred_next_week_sales']:.1f}"
    )

    st.subheader("선택 아이템 주차별 판매 흐름")
    chart = draw_item_chart(
        selected_weekly,
        selected_summary["pred_next_week_sales"],
        selected_summary["pred_next_year_week"]
    )
    st.plotly_chart(chart, use_container_width=True)

    st.subheader("선택 아이템 주차 데이터")
    st.dataframe(selected_weekly, use_container_width=True)

except Exception as e:
    st.error(f"실행 중 오류가 발생했습니다: {e}")
