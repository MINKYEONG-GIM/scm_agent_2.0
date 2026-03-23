import re
import pandas as pd
import streamlit as st
from sqlalchemy import create_engine

# -------------------------------------------------
# 1. 기본 설정
# -------------------------------------------------
st.set_page_config(page_title="아이템 시즌 분류", layout="wide")
st.title("아이템 시즌 분류 화면")

st.markdown("""
주차별 판매량 데이터를 기준으로 각 아이템을 아래 규칙으로 분류합니다.

- ALL_SEASON: 모든 시즌 비중 15% 이상
- SUMMER_PEAK: 여름 비중 40% 이상
- WINTER_PEAK: 겨울 비중 40% 이상
- SPRING_PEAK: 봄 비중 35% 이상
- FALL_PEAK: 가을 비중 35% 이상
- SPRING_FALL_PEAK: 봄+가을 비중 50% 이상
""")

# -------------------------------------------------
# 2. 시즌 정의
# -------------------------------------------------
# 기준:
# 봄   = 9~18주
# 여름 = 19~30주
# 가을 = 31~40주
# 겨울 = 41~52주 + 1~8주

def get_season(week: int) -> str:
    if 9 <= week <= 18:
        return "SPRING"
    elif 19 <= week <= 30:
        return "SUMMER"
    elif 31 <= week <= 40:
        return "FALL"
    else:
        return "WINTER"


# -------------------------------------------------
# 3. 분류 함수
# -------------------------------------------------
def classify_item(row: pd.Series) -> str:
    spring_ratio = row["SPRING_RATIO"]
    summer_ratio = row["SUMMER_RATIO"]
    fall_ratio = row["FALL_RATIO"]
    winter_ratio = row["WINTER_RATIO"]

    # 우선순위 중요
    if (
        spring_ratio >= 0.15
        and summer_ratio >= 0.15
        and fall_ratio >= 0.15
        and winter_ratio >= 0.15
    ):
        return "ALL_SEASON"

    if summer_ratio >= 0.40:
        return "SUMMER_PEAK"

    if winter_ratio >= 0.40:
        return "WINTER_PEAK"

    if (spring_ratio + fall_ratio) >= 0.50:
        return "SPRING_FALL_PEAK"

    if spring_ratio >= 0.35:
        return "SPRING_PEAK"

    if fall_ratio >= 0.35:
        return "FALL_PEAK"

    return "UNCLASSIFIED"


# -------------------------------------------------
# 4. DB 조회 함수
# -------------------------------------------------
@st.cache_data(show_spinner=False)
def load_data_from_db():
    """
    예시:
    DB에서 아래 형태의 결과가 나오도록 조회한다고 가정

    컬럼:
    - year_week : '2025-01'
    - item_name : '가디건'
    - sales_qty : 4035

    실제 테이블명/컬럼명에 맞게 SQL만 수정하면 됨
    """

    # ---------------------------------------------
    # DB 접속 정보 예시
    # 본인 환경에 맞게 수정
    # ---------------------------------------------
    DB_USER = "your_id"
    DB_PASSWORD = "your_pw"
    DB_HOST = "your_host"
    DB_PORT = "5432"
    DB_NAME = "your_db"

    # PostgreSQL 예시
    engine = create_engine(
        f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
    )

    sql = """
    SELECT
        year_week,
        item_name,
        sales_qty
    FROM item_weekly_sales
    WHERE year_week IS NOT NULL
      AND item_name IS NOT NULL
    """

    df = pd.read_sql(sql, engine)
    return df


# -------------------------------------------------
# 5. 업로드 파일 형태(가로형)도 처리 가능하도록 변환 함수
# -------------------------------------------------
def convert_wide_to_long(df_wide: pd.DataFrame) -> pd.DataFrame:
    """
    현재 업로드한 텍스트처럼
    행 = 연도/주
    열 = 아이템명
    값 = 판매수량
    형태일 때 long 형태로 바꿔줌
    """

    first_col = df_wide.columns[0]

    df_long = df_wide.melt(
        id_vars=[first_col],
        var_name="item_name",
        value_name="sales_qty"
    ).rename(columns={first_col: "year_week"})

    return df_long


# -------------------------------------------------
# 6. 데이터 전처리
# -------------------------------------------------
def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # 컬럼명 정리
    df.columns = [str(c).strip() for c in df.columns]

    # 필수 컬럼 확인
    required_cols = {"year_week", "item_name", "sales_qty"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"필수 컬럼이 없습니다: {missing}")

    # 값 정리
    df["year_week"] = df["year_week"].astype(str).str.strip()
    df["item_name"] = df["item_name"].astype(str).str.strip()

    # 숫자 변환
    df["sales_qty"] = (
        df["sales_qty"]
        .astype(str)
        .str.replace(",", "", regex=False)
        .str.strip()
        .replace("", "0")
    )
    df["sales_qty"] = pd.to_numeric(df["sales_qty"], errors="coerce").fillna(0)

    # year / week 분리
    extracted = df["year_week"].str.extract(r"(?P<year>\d{4})-(?P<week>\d{1,2})")
    df["year"] = pd.to_numeric(extracted["year"], errors="coerce")
    df["week"] = pd.to_numeric(extracted["week"], errors="coerce")

    df = df.dropna(subset=["year", "week"])
    df["year"] = df["year"].astype(int)
    df["week"] = df["week"].astype(int)

    # 시즌 부여
    df["season"] = df["week"].apply(get_season)

    return df


# -------------------------------------------------
# 7. 시즌 비중 계산 및 분류
# -------------------------------------------------
def make_classification_table(df: pd.DataFrame) -> pd.DataFrame:
    # 시즌별 판매량 합계
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

    # 0으로 나누기 방지
    pivot["SPRING_RATIO"] = pivot["SPRING"] / pivot["TOTAL_QTY"].replace(0, pd.NA)
    pivot["SUMMER_RATIO"] = pivot["SUMMER"] / pivot["TOTAL_QTY"].replace(0, pd.NA)
    pivot["FALL_RATIO"] = pivot["FALL"] / pivot["TOTAL_QTY"].replace(0, pd.NA)
    pivot["WINTER_RATIO"] = pivot["WINTER"] / pivot["TOTAL_QTY"].replace(0, pd.NA)

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

    # 퍼센트 보기 좋게
    for col in ["SPRING_RATIO", "SUMMER_RATIO", "FALL_RATIO", "WINTER_RATIO"]:
        result[col] = (result[col] * 100).round(1)

    result = result.sort_values(["CATEGORY", "TOTAL_QTY"], ascending=[True, False])

    return result


# -------------------------------------------------
# 8. 실행 영역
# -------------------------------------------------
data_source = st.radio(
    "데이터 소스 선택",
    ["DB에서 조회", "CSV 업로드"],
    horizontal=True
)

raw_df = None

if data_source == "DB에서 조회":
    if st.button("DB 데이터 가져오기"):
        try:
            raw_df = load_data_from_db()
            st.success("DB 데이터를 불러왔습니다.")
        except Exception as e:
            st.error(f"DB 조회 중 오류가 발생했습니다: {e}")

else:
    uploaded_file = st.file_uploader("CSV 파일 업로드", type=["csv"])
    if uploaded_file is not None:
        try:
            temp_df = pd.read_csv(uploaded_file)

            # CSV가 가로형이면 변환
            if temp_df.shape[1] > 3 and "year_week" not in temp_df.columns:
                temp_df = convert_wide_to_long(temp_df)

            raw_df = temp_df
            st.success("파일을 불러왔습니다.")
        except Exception as e:
            st.error(f"파일 처리 중 오류가 발생했습니다: {e}")

if raw_df is not None:
    try:
        df = preprocess_data(raw_df)
        result_df = make_classification_table(df)

        st.subheader("분류 결과")
        st.dataframe(result_df, use_container_width=True)

        st.subheader("분류별 건수")
        summary_df = (
            result_df.groupby("CATEGORY", as_index=False)
            .agg(
                item_count=("item_name", "count"),
                total_qty=("TOTAL_QTY", "sum")
            )
            .sort_values("item_count", ascending=False)
        )
        st.dataframe(summary_df, use_container_width=True)

        # 상세 조회
        st.subheader("아이템 상세 조회")
        item_list = result_df["item_name"].dropna().unique().tolist()
        selected_item = st.selectbox("아이템 선택", item_list)

        item_detail = result_df[result_df["item_name"] == selected_item]
        st.dataframe(item_detail, use_container_width=True)

    except Exception as e:
        st.error(f"분류 처리 중 오류가 발생했습니다: {e}")
