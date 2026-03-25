import re
from datetime import datetime
from typing import Optional, Tuple, List

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import streamlit as st
import gspread
from google.oauth2.service_account import Credentials


# -------------------------------------------------
# 1. 기본 설정
# -------------------------------------------------
st.set_page_config(page_title="시즌 분류 + 판매 예측", layout="wide")
st.title("아이템 시즌 분류 + 주차별 판매 예측")


# -------------------------------------------------
# 2. 시즌 정의
# -------------------------------------------------
def get_season(week: int) -> str:
    if 9 <= week <= 18:
        return "SPRING"
    elif 19 <= week <= 30:
        return "SUMMER"
    elif 31 <= week <= 40:
        return "FALL"
    else:
        return "WINTER"


def classify_item(row: pd.Series) -> str:
    spring_ratio = row["SPRING_RATIO"]
    summer_ratio = row["SUMMER_RATIO"]
    fall_ratio = row["FALL_RATIO"]
    winter_ratio = row["WINTER_RATIO"]

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
# 3. 구글시트 연결
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
    client = gspread.authorize(credentials)
    return client


def open_main_spreadsheet():
    client = get_gsheet_client()
    sheet_url = st.secrets["sheets"]["SHEET_URL"]
    return client.open_by_url(sheet_url)


# -------------------------------------------------
# 4. 기존 시즌 분류용 데이터 읽기
#    - 기존 코드 기준: st.secrets["sheets"]["WORKSHEET_NAME"]
# -------------------------------------------------
@st.cache_data(show_spinner=False)
def load_history_sheet():
    spreadsheet = open_main_spreadsheet()
    worksheet_name = st.secrets["sheets"]["WORKSHEET_NAME"]
    worksheet = spreadsheet.worksheet(worksheet_name)
    values = worksheet.get_all_values()

    if not values or len(values) < 3:
        raise ValueError("기존 시즌 분류 시트 구조를 확인해주세요. 최소 3행 이상 필요합니다.")

    header = values[1]
    data = values[2:]
    df = pd.DataFrame(data, columns=header)

    # 완전히 빈 컬럼 제거
    df = df.loc[:, [str(col).strip() != "" for col in df.columns]]
    return df


# -------------------------------------------------
# 5. final 시트 읽기
#    - 사용자가 말한 st.secrets["final"] 사용
#    - final 값은 worksheet 이름이라고 가정
# -------------------------------------------------
@st.cache_data(show_spinner=False)
def load_final_sheet_raw():
    spreadsheet = open_main_spreadsheet()

    final_sheet_name = st.secrets["final"]
    worksheet = spreadsheet.worksheet(final_sheet_name)
    values = worksheet.get_all_values()

    if not values or len(values) < 3:
        raise ValueError("final 시트 구조를 확인해주세요. 최소 3행 이상 필요합니다.")

    return values


def make_final_display_df(values: List[List[str]]) -> pd.DataFrame:
    """
    final 시트는 보통 2줄 헤더 구조라고 보고 처리
    0행: 날짜
    1행: 세부항목(기초재고, 판매량, 분배량 ...)
    2행부터: 데이터
    """
    header_row_1 = values[0]
    header_row_2 = values[1]
    data = values[2:]

    max_len = max(len(r) for r in values)

    def pad_row(row, n):
        return row + [""] * (n - len(row))

    header_row_1 = pad_row(header_row_1, max_len)
    header_row_2 = pad_row(header_row_2, max_len)
    data = [pad_row(r, max_len) for r in data]

    merged_columns = []
    current_date = ""

    for top, bottom in zip(header_row_1, header_row_2):
        top = str(top).strip()
        bottom = str(bottom).strip()

        if top:
            current_date = top

        if current_date and bottom:
            merged_columns.append(f"{current_date}|{bottom}")
        elif bottom:
            merged_columns.append(bottom)
        elif current_date:
            merged_columns.append(current_date)
        else:
            merged_columns.append("")

    df = pd.DataFrame(data, columns=merged_columns)

    # 완전히 빈 컬럼 제거
    df = df.loc[:, [str(c).strip() != "" for c in df.columns]]
    return df


# -------------------------------------------------
# 6. 기존 시즌 분류용 데이터 전처리
# -------------------------------------------------
def convert_wide_to_long(df_wide: pd.DataFrame) -> pd.DataFrame:
    df_wide = df_wide.copy()
    df_wide.columns = [str(c).strip() for c in df_wide.columns]

    first_col = df_wide.columns[0]

    df_long = df_wide.melt(
        id_vars=[first_col],
        var_name="item_name",
        value_name="sales_qty"
    ).rename(columns={first_col: "year_week"})

    df_long["item_name"] = df_long["item_name"].astype(str).str.strip()
    df_long = df_long[df_long["item_name"] != ""]
    return df_long


def preprocess_history_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]

    if "year_week" not in df.columns and "item_name" not in df.columns and "sales_qty" not in df.columns:
        df = convert_wide_to_long(df)

    rename_map = {}
    for col in df.columns:
        if col == "연도/주":
            rename_map[col] = "year_week"
        elif col == "아이템":
            rename_map[col] = "item_name"
        elif col in ["판매수량", "판매수량의 SUM"]:
            rename_map[col] = "sales_qty"

    df = df.rename(columns=rename_map)

    required_cols = {"year_week", "item_name", "sales_qty"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"기존 시즌 분류 시트에 필수 컬럼이 없습니다: {missing}")

    df["year_week"] = df["year_week"].astype(str).str.strip()
    df["item_name"] = df["item_name"].astype(str).str.strip()

    df["sales_qty"] = (
        df["sales_qty"]
        .astype(str)
        .str.replace(",", "", regex=False)
        .str.strip()
        .replace("", "0")
    )
    df["sales_qty"] = pd.to_numeric(df["sales_qty"], errors="coerce").fillna(0)

    extracted = df["year_week"].str.extract(r"(?P<year>\d{4})-(?P<week>\d{1,2})")
    df["year"] = pd.to_numeric(extracted["year"], errors="coerce")
    df["week"] = pd.to_numeric(extracted["week"], errors="coerce")

    df = df.dropna(subset=["year", "week"])
    df["year"] = df["year"].astype(int)
    df["week"] = df["week"].astype(int)
    df["season"] = df["week"].apply(get_season)

    # ISO week 기준 날짜 생성
    df["week_start_date"] = df.apply(
        lambda x: datetime.fromisocalendar(int(x["year"]), int(x["week"]), 1),
        axis=1
    )

    return df


def make_classification_table(df: pd.DataFrame) -> pd.DataFrame:
    season_sum = (
        df.groupby(["item_name", "season"], as_index=False)["sales_qty"]
        .sum()
    )

    pivot = (
        season_sum.pivot(index="item_name", columns="season", values="sales_qty")
        .fillna(0)
        .reset_index()
    )

    for col in ["SPRING", "SUMMER", "FALL", "WINTER"]:
        if col not in pivot.columns:
            pivot[col] = 0

    pivot["TOTAL_QTY"] = (
        pivot["SPRING"] + pivot["SUMMER"] + pivot["FALL"] + pivot["WINTER"]
    )

    total_nonzero = pivot["TOTAL_QTY"].replace(0, pd.NA)
    pivot["SPRING_RATIO"] = pivot["SPRING"] / total_nonzero
    pivot["SUMMER_RATIO"] = pivot["SUMMER"] / total_nonzero
    pivot["FALL_RATIO"] = pivot["FALL"] / total_nonzero
    pivot["WINTER_RATIO"] = pivot["WINTER"] / total_nonzero
    pivot = pivot.fillna(0)

    pivot["CATEGORY"] = pivot.apply(classify_item, axis=1)

    result = pivot[
        [
            "item_name",
            "SPRING",
            "SUMMER",
            "FALL",
            "WINTER",
            "TOTAL_QTY",
            "SPRING_RATIO",
            "SUMMER_RATIO",
            "FALL_RATIO",
            "WINTER_RATIO",
            "CATEGORY",
        ]
    ].copy()

    for col in ["SPRING_RATIO", "SUMMER_RATIO", "FALL_RATIO", "WINTER_RATIO"]:
        result[col] = (result[col] * 100).round(1)

    result = result.sort_values(["CATEGORY", "TOTAL_QTY"], ascending=[True, False]).reset_index(drop=True)
    return result


# -------------------------------------------------
# 7. 상품명에서 아이템 종류 추출
# -------------------------------------------------
ITEM_KEYWORDS = [
    "반팔티", "긴팔티", "민소매티", "셔츠", "블라우스", "가디건", "스웨터", "맨투맨",
    "원피스", "스커트", "반바지", "면바지", "기획바지", "변형바지", "점퍼", "일반점퍼",
    "데님자켓", "모직코트", "트렌치코트", "다운", "가방", "모자", "부츠", "운동화",
    "목도리", "머플러", "수영복", "슬리퍼", "양말"
]


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", "", str(text).strip().lower())


def extract_base_item_name(product_name: str) -> str:
    """
    예:
    [코튼스판] 캡소매 절개 반팔티, (19)Black, S
    -> 반팔티
    """
    text = str(product_name)

    # 대괄호 제거
    text = re.sub(r"\[.*?\]", "", text).strip()

    # 첫 번째 쉼표 앞까지만 사용
    text = text.split(",")[0].strip()

    # 등록된 키워드 우선 탐색
    for keyword in ITEM_KEYWORDS:
        if keyword in text:
            return keyword

    # 못 찾으면 마지막 단어 비슷하게 반환
    parts = text.split()
    if parts:
        return parts[-1].strip()

    return text.strip()


def find_best_history_item(product_name: str, classification_df: pd.DataFrame) -> Optional[str]:
    extracted = extract_base_item_name(product_name)
    extracted_norm = normalize_text(extracted)

    item_names = classification_df["item_name"].dropna().astype(str).tolist()

    # 1차: 완전일치
    for item in item_names:
        if normalize_text(item) == extracted_norm:
            return item

    # 2차: 포함일치
    for item in item_names:
        item_norm = normalize_text(item)
        if extracted_norm in item_norm or item_norm in extracted_norm:
            return item

    return None


# -------------------------------------------------
# 8. 그래프용 데이터 생성
# -------------------------------------------------
def make_item_weekly_series(history_df: pd.DataFrame, item_name: str) -> pd.DataFrame:
    item_df = history_df[history_df["item_name"] == item_name].copy()
    item_df = item_df.sort_values("week_start_date")

    weekly = (
        item_df.groupby(["year_week", "week_start_date"], as_index=False)["sales_qty"]
        .sum()
        .sort_values("week_start_date")
        .reset_index(drop=True)
    )
    return weekly


def make_item_monthly_series(history_df: pd.DataFrame, item_name: str) -> pd.DataFrame:
    item_df = history_df[history_df["item_name"] == item_name].copy()
    item_df["year_month"] = pd.to_datetime(item_df["week_start_date"]).dt.to_period("M").astype(str)

    monthly = (
        item_df.groupby("year_month", as_index=False)["sales_qty"]
        .sum()
        .rename(columns={"sales_qty": "monthly_sales_qty"})
    )
    return monthly


def draw_weekly_monthly_chart(weekly_df: pd.DataFrame, monthly_df: pd.DataFrame, item_name: str):
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=weekly_df["year_week"],
            y=weekly_df["sales_qty"],
            mode="lines+markers",
            name="주차별 판매량"
        )
    )

    # 월별 선도 같이 표시
    # x축 라벨 길이가 달라서 월별은 별도 축 대신 같은 축에 간단히 뒤쪽에 표시하기보다
    # 사용자 요청대로 한 화면에 2개 선을 보여주려면 x축 기준을 날짜로 맞추는 게 가장 안전함
    if not monthly_df.empty:
        monthly_plot = monthly_df.copy()
        monthly_plot["month_date"] = pd.to_datetime(monthly_plot["year_month"] + "-01")

        fig.add_trace(
            go.Scatter(
                x=monthly_plot["month_date"],
                y=monthly_plot["monthly_sales_qty"],
                mode="lines+markers",
                name="월별 판매량"
            )
        )

    fig.update_layout(
        title=f"{item_name} 월별/주차별 판매 추이",
        xaxis_title="기간",
        yaxis_title="판매량",
        hovermode="x unified",
        height=500,
    )

    st.plotly_chart(fig, use_container_width=True)


# -------------------------------------------------
# 9. final 시트 전처리
# -------------------------------------------------
def get_first_existing_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def parse_final_to_long(df: pd.DataFrame) -> pd.DataFrame:
    """
    final 시트 구조 예시:
    상품코드 / 상품명 / 02월25일|기초재고 / 02월25일|판매량 / ... / 03월25일|예측판매량 ...
    """
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]

    sku_col = get_first_existing_col(df, ["품번", "상품코드", "스타일", "style", "STYLE"])
    name_col = get_first_existing_col(df, ["상품명", "아이템명", "품명", "name", "NAME"])

    if sku_col is None:
        sku_col = df.columns[0]
    if name_col is None:
        name_col = df.columns[1]

    meta_cols = [sku_col, name_col]
    date_metric_cols = [c for c in df.columns if "|" in str(c)]

    long_rows = []

    for _, row in df.iterrows():
        sku = row[sku_col]
        product_name = row[name_col]

        grouped = {}
        for col in date_metric_cols:
            date_str, metric = col.split("|", 1)
            grouped.setdefault(date_str, {})
            grouped[date_str][metric] = row[col]

        for date_str, metrics in grouped.items():
            one = {
                "sku": sku,
                "product_name": product_name,
                "date_label": date_str.strip(),
            }
            one.update(metrics)
            long_rows.append(one)

    long_df = pd.DataFrame(long_rows)

    # 숫자형 변환
    numeric_cols = ["기초재고", "판매량", "예측판매량", "분배량", "출고량(회전 등)", "로스", "보조지표"]
    for c in numeric_cols:
        if c not in long_df.columns:
            long_df[c] = 0

        long_df[c] = (
            long_df[c]
            .astype(str)
            .str.replace(",", "", regex=False)
            .str.strip()
            .replace("", "0")
        )
        long_df[c] = pd.to_numeric(long_df[c], errors="coerce").fillna(0)

    # 날짜 파싱
    long_df["parsed_date"] = pd.to_datetime(long_df["date_label"], format="%m월 %d일", errors="coerce")

    # 연도는 현재 연도 기준으로 넣음
    current_year = datetime.now().year
    long_df["parsed_date"] = long_df["parsed_date"].apply(
        lambda x: x.replace(year=current_year) if pd.notnull(x) else pd.NaT
    )

    long_df = long_df.sort_values(["product_name", "parsed_date"]).reset_index(drop=True)
    return long_df


# -------------------------------------------------
# 10. 주차별 판매 예측
# -------------------------------------------------
def compute_seasonal_factor(history_weekly_df: pd.DataFrame, target_week: int) -> float:
    """
    과거 동일 주차 평균 / 전체 주차 평균
    """
    if history_weekly_df.empty:
        return 1.0

    temp = history_weekly_df.copy()
    temp["week_no"] = temp["year_week"].str.extract(r"-(\d{1,2})$")[0]
    temp["week_no"] = pd.to_numeric(temp["week_no"], errors="coerce")

    overall_mean = temp["sales_qty"].mean()
    same_week_mean = temp.loc[temp["week_no"] == target_week, "sales_qty"].mean()

    if pd.isna(overall_mean) or overall_mean <= 0:
        return 1.0
    if pd.isna(same_week_mean) or same_week_mean <= 0:
        return 1.0

    factor = same_week_mean / overall_mean

    # 너무 과하게 흔들리지 않게 제한
    factor = max(0.7, min(1.3, factor))
    return factor


def predict_next_sales(item_final_df: pd.DataFrame, history_weekly_df: pd.DataFrame) -> pd.DataFrame:
    """
    예측 방식:
    1. 최근 실제 판매량 3주 가중평균
    2. 과거 동일 시즌성 보정
    3. 기초재고보다 많이 예측하지 않도록 제한
    """
    df = item_final_df.copy().sort_values("parsed_date").reset_index(drop=True)

    sales_col = "판매량"
    pred_col = "예측판매량"

    actual_sales = df[sales_col].astype(float).tolist()
    base_stock = df["기초재고"].astype(float).tolist()

    predictions = []

    for i in range(len(df)):
        current_actual = df.loc[i, sales_col]
        current_pred = df.loc[i, pred_col]

        # 이미 예측판매량이 있으면 유지
        if pd.notnull(current_pred) and float(current_pred) > 0:
            predictions.append(float(current_pred))
            continue

        # 최근 실제 판매량 확보
        history_actual = [x for x in actual_sales[:i] if pd.notnull(x)]
        recent = history_actual[-3:]

        if len(recent) == 0:
            base_pred = 0.0
        elif len(recent) == 1:
            base_pred = recent[-1]
        elif len(recent) == 2:
            base_pred = recent[-1] * 0.6 + recent[-2] * 0.4
        else:
            base_pred = recent[-1] * 0.5 + recent[-2] * 0.3 + recent[-3] * 0.2

        # 목표 주차 계산
        target_date = df.loc[i, "parsed_date"]
        if pd.notnull(target_date):
            target_week = int(target_date.isocalendar().week)
        else:
            target_week = None

        seasonal_factor = 1.0
        if target_week is not None and not history_weekly_df.empty:
            seasonal_factor = compute_seasonal_factor(history_weekly_df, target_week)

        pred = base_pred * seasonal_factor

        # 음수 방지
        pred = max(0.0, pred)

        # 재고 제한
        stock = float(base_stock[i]) if pd.notnull(base_stock[i]) else 0.0
        pred = min(pred, stock)

        predictions.append(round(pred))

    df["최종예측판매량"] = predictions
    return df


# -------------------------------------------------
# 11. 화면 실행
# -------------------------------------------------
if st.button("데이터 불러오기"):
    try:
        # 1) 기존 시즌 분류용 데이터
        raw_history_df = load_history_sheet()
        history_df = preprocess_history_data(raw_history_df)
        classification_df = make_classification_table(history_df)

        # 2) final 시트
        final_values = load_final_sheet_raw()
        final_display_df = make_final_display_df(final_values)
        final_long_df = parse_final_to_long(final_display_df)

        st.success("데이터를 불러왔습니다.")

        # -------------------------------------------------
        # A. 기존 시즌 분류 결과
        # -------------------------------------------------
        st.subheader("시즌 분류 결과")
        st.dataframe(classification_df, use_container_width=True)

        # -------------------------------------------------
        # B. final 시트 원본 표
        # -------------------------------------------------
        st.subheader("final 시트 원본 표")
        st.dataframe(final_display_df, use_container_width=True)

        # -------------------------------------------------
        # C. 상품 선택
        # -------------------------------------------------
        st.subheader("상품 상세 조회")
        product_list = final_long_df["product_name"].dropna().astype(str).unique().tolist()
        selected_product = st.selectbox("상품명 선택", product_list)

        product_final_df = final_long_df[final_long_df["product_name"] == selected_product].copy()
        product_final_df = product_final_df.sort_values("parsed_date").reset_index(drop=True)

        st.write("선택 상품 데이터")
        st.dataframe(product_final_df, use_container_width=True)

        # -------------------------------------------------
        # D. 상품명 -> 시즌 아이템 매칭
        # -------------------------------------------------
        extracted_item = extract_base_item_name(selected_product)
        matched_history_item = find_best_history_item(selected_product, classification_df)

        col1, col2, col3 = st.columns(3)
        col1.metric("상품명에서 읽은 아이템", extracted_item)
        col2.metric("매칭된 기존 아이템", matched_history_item if matched_history_item else "매칭 실패")

        if matched_history_item:
            matched_row = classification_df[classification_df["item_name"] == matched_history_item].iloc[0]
            col3.metric("시즌 분류", matched_row["CATEGORY"])

            st.write("시즌 분류 상세")
            st.dataframe(
                pd.DataFrame([matched_row]),
                use_container_width=True
            )
        else:
            st.warning("상품명에서 기존 시즌 분류 아이템을 찾지 못했습니다. 키워드 사전을 추가하면 더 정확해집니다.")

        # -------------------------------------------------
        # E. 월별 / 주차별 그래프
        # -------------------------------------------------
        if matched_history_item:
            weekly_series = make_item_weekly_series(history_df, matched_history_item)
            monthly_series = make_item_monthly_series(history_df, matched_history_item)

            st.subheader("월별 / 주차별 판매 그래프")
            draw_weekly_monthly_chart(weekly_series, monthly_series, matched_history_item)

            st.write("주차별 판매 데이터")
            st.dataframe(weekly_series, use_container_width=True)

            st.write("월별 판매 데이터")
            st.dataframe(monthly_series, use_container_width=True)

        # -------------------------------------------------
        # F. 다음 주 판매 예측
        # -------------------------------------------------
        st.subheader("주차별 판매 예측")

        if matched_history_item:
            weekly_series_for_pred = make_item_weekly_series(history_df, matched_history_item)
        else:
            weekly_series_for_pred = pd.DataFrame(columns=["year_week", "week_start_date", "sales_qty"])

        predicted_df = predict_next_sales(product_final_df, weekly_series_for_pred)

        st.dataframe(predicted_df, use_container_width=True)

        # 예측 그래프
        fig_pred = go.Figure()

        fig_pred.add_trace(
            go.Scatter(
                x=predicted_df["date_label"],
                y=predicted_df["판매량"],
                mode="lines+markers",
                name="실판매량"
            )
        )

        fig_pred.add_trace(
            go.Scatter(
                x=predicted_df["date_label"],
                y=predicted_df["최종예측판매량"],
                mode="lines+markers",
                name="예측판매량"
            )
        )

        fig_pred.update_layout(
            title=f"{selected_product} 주차별 판매 예측",
            xaxis_title="기준일",
            yaxis_title="판매량",
            hovermode="x unified",
            height=500
        )

        st.plotly_chart(fig_pred, use_container_width=True)

    except Exception as e:
        st.error(f"오류가 발생했습니다: {e}")
