import pandas as pd
import streamlit as st
import gspread
from google.oauth2.service_account import Credentials

# -------------------------------------------------
# 1. 기본 설정
# -------------------------------------------------
st.set_page_config(page_title="아이템 시즌 분류", layout="wide")
st.title("아이템 시즌 분류 화면")

st.markdown("""
주차별 판매량 데이터를 기준으로 각 아이템을 아래 규칙으로 분류합니다.


- SUMMER_PEAK: 여름 비중 40% 이상
- WINTER_PEAK: 겨울 비중 40% 이상
- SPRING_FALL_PEAK: 봄+가을 비중 60% 이상
- SPRING_PEAK: 봄 비중 35% 이상
- FALL_PEAK: 가을 비중 35% 이상
- ALL_SEASON: 기타

""")

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

# -------------------------------------------------
# 3. 분류 함수
# -------------------------------------------------
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
# 4. 구글시트 연결
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

# -------------------------------------------------
# 5. 구글시트 데이터 읽기
# -------------------------------------------------
def _sheets_section():
    if "sheets" not in st.secrets:
        raise ValueError("secrets에 [sheets] 섹션이 없습니다. secrets.toml을 확인하세요.")
    return st.secrets["sheets"]


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
        raise ValueError("[sheets]에 SHEET_URL 또는 sheet_id 중 하나는 필요합니다.")

    # secrets.toml에서 공백 포함 키는 반드시 따옴표: "PLC DB" = "plc db"
    plc_db_sheet = sheets.get("PLC DB") or sheets.get("WORKSHEET_NAME") or "plc db"
    season_db_sheet = sheets.get("SEASON DB") or "season db"
    final_sheet = sheets.get("final") or "final"

    plc_worksheet = spreadsheet.worksheet(plc_db_sheet)
    season_worksheet = spreadsheet.worksheet(season_db_sheet)
    final_worksheet = spreadsheet.worksheet(final_sheet)

    # PLC DB: 가로형 판매(0행 제목, 1행 헤더, 2행~ 데이터)
    values = plc_worksheet.get_all_values()

    if not values or len(values) < 3:
        raise ValueError("구글시트 데이터 구조를 확인해주세요. 최소 3행 이상 필요합니다.")

    # 0행: 상단 제목(판매수량의 SUM, 아이템 등)
    # 1행: 실제 헤더(연도/주, 가디건, 가방, ...)
    # 2행부터: 실제 데이터
    header = values[1]
    data = values[2:]

    df = pd.DataFrame(data, columns=header)

    # 완전히 빈 컬럼 제거
    df = df.loc[:, [str(col).strip() != "" for col in df.columns]]

    return df
# -------------------------------------------------
# 6. 가로형 데이터 -> 세로형 변환
# -------------------------------------------------
def convert_wide_to_long(df_wide: pd.DataFrame) -> pd.DataFrame:
    df_wide = df_wide.copy()
    df_wide.columns = [str(c).strip() for c in df_wide.columns]

    # 첫 컬럼명 찾기
    first_col = df_wide.columns[0]

    # 보통 '연도/주' 이지만 혹시 다르면 첫 컬럼을 year_week로 강제 사용
    df_long = df_wide.melt(
        id_vars=[first_col],
        var_name="item_name",
        value_name="sales_qty"
    ).rename(columns={first_col: "year_week"})

    # 빈 아이템 제거
    df_long["item_name"] = df_long["item_name"].astype(str).str.strip()
    df_long = df_long[df_long["item_name"] != ""]

    return df_long

# -------------------------------------------------
# 7. 전처리
# -------------------------------------------------
def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]

    # 가로형이면 자동 변환
    if "year_week" not in df.columns and "item_name" not in df.columns and "sales_qty" not in df.columns:
        df = convert_wide_to_long(df)

    # 한글 헤더 대응
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
        raise ValueError(f"필수 컬럼이 없습니다: {missing}")

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

    return df

# -------------------------------------------------
# 8. 분류 테이블 생성
# -------------------------------------------------
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

    result = result.sort_values(["CATEGORY", "TOTAL_QTY"], ascending=[True, False])
    return result

# -------------------------------------------------
# 9. 실행 (페이지 로드 시 자동 조회)
# -------------------------------------------------
try:
    with st.spinner("구글시트에서 데이터를 불러오는 중..."):
        raw_df = load_data_from_gsheet()
        df = preprocess_data(raw_df)
        result_df = make_classification_table(df)

    st.success("구글시트 데이터를 불러왔습니다.")

    st.subheader("분류 결과")
    st.dataframe(result_df, use_container_width=True)

    st.subheader("분류별 건수")
    summary_df = (
        result_df.groupby("CATEGORY", as_index=False)
        .agg(
            item_count=("item_name", "count"),
            total_qty=("TOTAL_QTY", "sum"),
        )
        .sort_values("item_count", ascending=False)
    )
    st.dataframe(summary_df, use_container_width=True)

    st.subheader("아이템 상세 조회")
    item_list = result_df["item_name"].dropna().unique().tolist()
    selected_item = st.selectbox("아이템 선택", item_list)

    item_detail = result_df[result_df["item_name"] == selected_item]
    st.dataframe(item_detail, use_container_width=True)

except Exception as e:
    st.error(f"구글시트 조회 중 오류가 발생했습니다: {e}")
