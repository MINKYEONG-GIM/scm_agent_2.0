#PLC는 N개(6개이상, 도입-성장-성숙-쇠퇴 + 정점, 기울기 꺾이는 변곡점)로 쪼개시고 정점이 언제인지, 그래프가 급격히 꺽이는 시점이 언제인지 등을 통해 구간을 분류해주세요
#app.py 에서 from plc_utils import 함수명 으로 가져다 쓸 수 있음

# 포함할 기능: 주차 집계, 스무딩, slope, accel, peak 찾기, turning point 찾기, stage 분류 (7단계 기준), 시즌종료 (discount 기준 가능하게), 급감, 변곡점 flag

import pandas as pd


# =========================
# 1. week 정렬용 변환
# =========================

def parse_year_week(week_str: str):
    """
    '2024-23' -> (2024, 23)
    """
    try:
        y, w = week_str.split("-")
        return int(y), int(w)
    except:
        return 0, 0


# =========================
# 2. 주차별 집계
# =========================

def build_plc_weekly_series(
    base_df: pd.DataFrame,
    week_col: str = "similar_week",
    qty_col: str = "similar_forecast_qty_num"
) -> pd.DataFrame:

    df = (
        base_df.groupby(week_col, as_index=False)[qty_col]
        .sum()
        .rename(columns={
            week_col: "week",
            qty_col: "qty"
        })
    )

    df["sort_key"] = df["week"].apply(parse_year_week)
    df = df.sort_values("sort_key").reset_index(drop=True)

    df["t"] = range(len(df))

    return df


# =========================
# 3. 스무딩 + slope + accel
# =========================

def add_plc_features(
    df: pd.DataFrame,
    window: int = 3
) -> pd.DataFrame:

    result = df.copy()

    # 이동평균
    result["smooth_qty"] = (
        result["qty"]
        .rolling(window=window, center=True, min_periods=1)
        .mean()
    )

    # 기울기
    result["slope"] = result["smooth_qty"].diff()

    # 기울기 변화
    result["accel"] = result["slope"].diff()

    # 최대값
    max_qty = result["smooth_qty"].max()

    if max_qty == 0:
        result["qty_ratio"] = 0
    else:
        result["qty_ratio"] = result["smooth_qty"] / max_qty

    return result


# =========================
# 4. peak 찾기
# =========================

def find_peak_idx(df: pd.DataFrame):

    if df.empty:
        return None

    return df["smooth_qty"].idxmax()


# =========================
# 5. turning point 찾기
# =========================

def find_turning_points(df: pd.DataFrame):

    turning = []

    slope = df["slope"].fillna(0)

    for i in range(1, len(df)):

        prev = slope.iloc[i - 1]
        curr = slope.iloc[i]

        if prev > 0 and curr < 0:
            turning.append(i)

    return turning


# =========================
# 6. 급감 찾기
# =========================

def find_fast_drop(df: pd.DataFrame):

    fast = []

    for i in range(1, len(df)):

        prev = df["smooth_qty"].iloc[i - 1]
        curr = df["smooth_qty"].iloc[i]

        if prev == 0:
            continue

        if curr < prev * 0.8:
            fast.append(i)

    return fast


# =========================
# 7. stage 분류
# =========================

def classify_plc_stage(
    df: pd.DataFrame,
    discount_series: pd.Series | None = None
) -> pd.DataFrame:

    result = df.copy()

    result["stage"] = "도입"
    result["is_peak"] = False
    result["is_turning"] = False
    result["is_fast_drop"] = False
    result["is_season_end"] = False

    if result.empty:
        return result

    max_qty = result["smooth_qty"].max()

    peak_idx = find_peak_idx(result)
    turning = find_turning_points(result)
    fast = find_fast_drop(result)

    # peak
    if peak_idx is not None:
        result.loc[peak_idx, "stage"] = "정점"
        result.loc[peak_idx, "is_peak"] = True

    # turning
    for i in turning:
        result.loc[i, "is_turning"] = True

    # fast drop
    for i in fast:
        result.loc[i, "is_fast_drop"] = True

    for i in result.index:

        if i == peak_idx:
            continue

        qty_ratio = result.loc[i, "qty_ratio"]
        slope = result.loc[i, "slope"]
        accel = result.loc[i, "accel"]

        # 시즌종료
        if discount_series is not None:

            if i < len(discount_series):

                discount = discount_series.iloc[i]

                if discount >= 0.5 and qty_ratio < 0.4:
                    result.loc[i, "stage"] = "시즌종료"
                    result.loc[i, "is_season_end"] = True
                    continue

        # 급감
        if slope < 0 and accel < 0 and qty_ratio < 0.7:
            result.loc[i, "stage"] = "급감시작"
            continue

        # 쇠퇴
        if qty_ratio < 0.3 and slope <= 0:
            result.loc[i, "stage"] = "쇠퇴"
            continue

        # 성숙
        if qty_ratio >= 0.7 and abs(slope) < max_qty * 0.05:
            result.loc[i, "stage"] = "성숙"
            continue

        # 성장
        if slope > 0 and qty_ratio >= 0.3:
            result.loc[i, "stage"] = "성장"
            continue

        # 도입
        result.loc[i, "stage"] = "도입"

    return result


# =========================
# 8. 전체 파이프라인
# =========================

def run_plc_pipeline(
    base_df: pd.DataFrame,
    week_col="similar_week",
    qty_col="similar_forecast_qty_num",
    discount_series=None
):

    df = build_plc_weekly_series(
        base_df,
        week_col=week_col,
        qty_col=qty_col
    )

    df = add_plc_features(df)

    df = classify_plc_stage(
        df,
        discount_series=discount_series
    )

    return df


plc_df = run_plc_pipeline(
    base_df,
    discount_series=discount_series
)
