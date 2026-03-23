import pandas as pd
import numpy as np

import streamlit as st
from streamlit_gsheets import GSheetsConnection

st.title("구글시트 연결 테스트")

SHEET_ID = "1IlJxe4ocFeNODRxMxpgtHA1xKC-Xn5-YvRsaHciUfLw"
WORKSHEET_NAME = "plc db"

conn = st.connection("gsheets", type=GSheetsConnection)

df = conn.read(
    spreadsheet=SHEET_ID,
    worksheet=WORKSHEET_NAME,
    ttl=0
)

st.dataframe(df, use_container_width=True)


st.set_page_config(page_title="PLC 분석기", layout="wide")
st.title("아이템 PLC 자동 분류")

conn = st.connection("gsheets", type=GSheetsConnection)
df = conn.read(worksheet="plc db", ttl=0)

st.subheader("원본 데이터")
st.dataframe(df, use_container_width=True)

result_df = run_plc_classification(df)

st.subheader("분석 결과")
st.dataframe(
    result_df[["아이템", "연도/주", "판매수량", "판매수량_ma", "할인율", "plc"]],
    use_container_width=True
)

# -------------------------------------------------
# 1. 데이터 불러오기
# -------------------------------------------------
# 예시:
# df = pd.read_csv("sales.csv", encoding="utf-8-sig")

# 이미 DataFrame이 있다면 이 부분은 생략 가능
# 반드시 필요한 컬럼:
# 아이템, 연도/주, 외형매출, 정상가, 판매수량

# -------------------------------------------------
# 2. 전처리 함수
# -------------------------------------------------
def prepare_weekly_item_data(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy()

    required_cols = ["아이템", "연도/주", "외형매출", "정상가", "판매수량"]
    missing_cols = [col for col in required_cols if col not in data.columns]
    if missing_cols:
        raise ValueError(f"필수 컬럼이 없습니다: {missing_cols}")

    # 숫자형 변환
    for col in ["외형매출", "정상가", "판매수량"]:
        data[col] = (
            data[col]
            .astype(str)
            .str.replace(",", "", regex=False)
            .str.strip()
        )
        data[col] = pd.to_numeric(data[col], errors="coerce").fillna(0)

    # 반품/음수 데이터가 있으면 그대로 둘 수도 있고 제외할 수도 있음
    # 여기서는 판매 흐름 분석용이므로 판매수량 0 이하 주차는 제외
    # 필요하면 아래 줄을 주석 처리
    data = data[data["판매수량"] > 0].copy()

    # 연도/주 -> 정렬용 숫자 컬럼 생성
    # 예: 2025-01 -> year=2025, week=1
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
    # 정상가가 0이면 할인율 계산 불가 -> 0 처리
    weekly["할인율"] = np.where(
        weekly["정상가"] > 0,
        1 - (weekly["외형매출"] / weekly["정상가"]),
        0
    )

    # 이상치 방지
    weekly["할인율"] = weekly["할인율"].clip(lower=0, upper=1)

    return weekly


# -------------------------------------------------
# 3. 아이템별 PLC 단계 분류 함수
# -------------------------------------------------
def classify_plc_by_item(item_df: pd.DataFrame,
                         intro_weeks: int = 3,
                         decline_discount_threshold: float = 0.30) -> pd.DataFrame:
    """
    intro_weeks: 판매 시작 후 몇 주까지 도입으로 볼지
    decline_discount_threshold: 쇠퇴로 보는 평균 할인율 기준 (예: 0.30 = 30%)
    """
    g = item_df.sort_values("yearweek_num").copy().reset_index(drop=True)

    # 이동평균 판매량 (노이즈 완화)
    g["판매수량_ma"] = g["판매수량"].rolling(window=3, min_periods=1).mean()

    # 이전/다음 주 값
    g["prev_ma"] = g["판매수량_ma"].shift(1)
    g["next_ma"] = g["판매수량_ma"].shift(-1)

    # 결측 처리
    g["prev_ma"] = g["prev_ma"].fillna(g["판매수량_ma"])
    g["next_ma"] = g["next_ma"].fillna(g["판매수량_ma"])

    # 시작 후 몇 번째 판매 주차인지
    g["판매주차순번"] = np.arange(1, len(g) + 1)

    # 최고점
    peak_idx = g["판매수량_ma"].idxmax()
    peak_value = g.loc[peak_idx, "판매수량_ma"]

    if peak_value == 0:
        g["peak_ratio"] = 0
    else:
        g["peak_ratio"] = g["판매수량_ma"] / peak_value

    # 최근 증감률
    g["growth_rate"] = np.where(
        g["prev_ma"] > 0,
        (g["판매수량_ma"] - g["prev_ma"]) / g["prev_ma"],
        0
    )

    # 최고점 이후 여부
    g["is_after_peak"] = g.index > peak_idx

    # local peak 조건
    g["is_local_peak"] = (
        (g["판매수량_ma"] >= g["prev_ma"]) &
        (g["판매수량_ma"] >= g["next_ma"])
    )

    plc_list = []

    for idx, row in g.iterrows():
        stage = None

        # 1) 변곡점: 전체 최고점 주차를 우선 지정
        if idx == peak_idx:
            stage = "변곡점"

        # 2) 도입
        elif (
            row["판매주차순번"] <= intro_weeks and
            row["peak_ratio"] < 0.35
        ):
            stage = "도입"

        # 3) 쇠퇴
        elif (
            row["is_after_peak"] and
            row["growth_rate"] < -0.05 and
            (
                row["peak_ratio"] < 0.70 or
                row["할인율"] >= decline_discount_threshold
            )
        ):
            stage = "쇠퇴"

        # 4) 성숙
        elif (
            row["peak_ratio"] >= 0.85 and
            abs(row["growth_rate"]) <= 0.10
        ):
            stage = "성숙"

        # 5) 성장
        elif (
            (not row["is_after_peak"]) and
            row["peak_ratio"] >= 0.35 and
            row["growth_rate"] > 0.05
        ):
            stage = "성장"

        # 6) 나머지 보정 규칙
        else:
            # 최고점 전이면 성장 쪽
            if not row["is_after_peak"]:
                if row["peak_ratio"] < 0.35:
                    stage = "도입"
                elif row["peak_ratio"] < 0.85:
                    stage = "성장"
                else:
                    stage = "성숙"
            # 최고점 후면 성숙/쇠퇴 중 선택
            else:
                if row["peak_ratio"] >= 0.70 and row["할인율"] < decline_discount_threshold:
                    stage = "성숙"
                else:
                    stage = "쇠퇴"

        plc_list.append(stage)

    g["plc"] = plc_list
    return g


# -------------------------------------------------
# 4. 전체 아이템에 적용
# -------------------------------------------------
def run_plc_classification(df: pd.DataFrame,
                           intro_weeks: int = 3,
                           decline_discount_threshold: float = 0.30) -> pd.DataFrame:
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


# -------------------------------------------------
# 5. 사용 예시
# -------------------------------------------------
# result_df = run_plc_classification(df)

# 보고 싶은 컬럼만 확인
# print(result_df[[
#     "아이템", "연도/주", "판매수량", "판매수량_ma",
#     "할인율", "peak_ratio", "growth_rate", "plc"
# ]].head(50))

# 파일 저장
# result_df.to_csv("item_plc_result.csv", index=False, encoding="utf-8-sig")
