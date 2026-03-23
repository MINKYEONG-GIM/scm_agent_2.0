import os
import json
import re
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import streamlit as st
import gspread
from google.oauth2.service_account import Credentials
from streamlit_gsheets import GSheetsConnection

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

def mark_off_season_stage(df: pd.DataFrame) -> pd.DataFrame:
    """
    성장 중간의 낮은 판매 구간을 비시즌으로 표시한다.

    조건:
    - peak 이전 구간
    - 최고점 대비 OFF_SEASON_RATIO 이하
    - 최소 OFF_SEASON_MIN_WEEKS주 이상 연속
    - 이후 다시 OFF_SEASON_RECOVERY_RATIO 이상으로 회복
    """
    out = df.copy().reset_index(drop=True)

    if out.empty:
        return out

    peak_idx = int(out["qty"].fillna(0).values.argmax())
    n = len(out)

    if "plc_stage_final" not in out.columns:
        out["plc_stage_final"] = out["plc_stage_raw"]

    candidate_idx = []
    for i in range(n):
        ratio = out.loc[i, "ratio_to_peak"]

        if pd.isna(ratio):
            continue

        # peak 이전만 비시즌 후보
        if i >= peak_idx:
            continue

        # 도입 직후 너무 초반 구간 제외
        if i <= INTRO_WEEKS:
            continue

        if ratio <= OFF_SEASON_RATIO:
            candidate_idx.append(i)

    if not candidate_idx:
        return out

    # 연속 구간 묶기
    groups = []
    current = [candidate_idx[0]]

    for idx in candidate_idx[1:]:
        if idx == current[-1] + 1:
            current.append(idx)
        else:
            groups.append(current)
            current = [idx]
    groups.append(current)

    valid_groups = []
    for g in groups:
        if len(g) < OFF_SEASON_MIN_WEEKS:
            continue

        # 이후에 다시 회복되는지 확인
        after_end = g[-1] + 1
        if after_end >= n:
            continue

        future_ratios = out.loc[after_end:peak_idx, "ratio_to_peak"]
        if (future_ratios >= OFF_SEASON_RECOVERY_RATIO).any():
            valid_groups.append(g)

    if not valid_groups:
        return out

    # peak에 가장 가까운 비시즌 구간 1개 선택
    best_group = min(valid_groups, key=lambda g: abs(peak_idx - g[-1]))

    for i in best_group:
        out.loc[i, "plc_stage_final"] = "비시즌"

    return out




# =========================
# 1) 기본 설정
# =========================
st.set_page_config(
    page_title="아이템별 PLC 포인트 분석",
    layout="wide",
)

st.title("아이템별 PLC 포인트 분석")
st.caption("구글시트 기반으로 아이템별 주차 판매 흐름을 분석하고 PLC 단계를 자동 분류합니다.")


# =========================
# 2) 설정값
# =========================
INTRO_WEEKS = 3
GROWTH_MIN_RATIO = 0.35
GROWTH_MAX_RATIO = 0.85
MATURE_RATIO = 0.85
DECLINE_RATIO = 0.70
HIGH_DISCOUNT_RATIO = 0.30
#비시즌 관련 수치
OFF_SEASON_RATIO = 0.45          # 최고점 대비 45% 이하
OFF_SEASON_MIN_WEEKS = 3         # 최소 3주 이상 지속
OFF_SEASON_RECOVERY_RATIO = 0.60 # 이후 다시 60% 이상 회복되면 비시즌으로 인정

ROLLING_WINDOW = 3  # 최근 흐름 판단용
LOCAL_PEAK_WINDOW = 1  # 앞/뒤 1주 비교


# =========================
# 3) 유틸 함수
# =========================
def parse_year_week(value: str) -> Tuple[Optional[int], Optional[int]]:
    """
    '2025-01', '2025/01', '202501' 같은 값을 year, week로 변환
    """
    if pd.isna(value):
        return None, None

    text = str(value).strip()

    # 2025-01 / 2025/01 / 2025_01
    m = re.match(r"^(\d{4})[-/_]?(\d{1,2})$", text)
    if m:
        return int(m.group(1)), int(m.group(2))

    # 혹시 숫자만 있는 경우
    m = re.match(r"^(\d{4})(\d{2})$", text)
    if m:
        return int(m.group(1)), int(m.group(2))

    return None, None


def yearweek_to_sort_key(year_week: str) -> int:
    year, week = parse_year_week(year_week)
    if year is None or week is None:
        return 99999999
    return year * 100 + week


def safe_to_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(
        series.astype(str).str.replace(",", "", regex=False).str.strip(),
        errors="coerce"
    )


def calc_discount_rate(gross_sales: pd.Series, full_price_sales: pd.Series) -> pd.Series:
    """
    할인율 = 1 - (외형매출 / 정상가)
    정상가가 0이거나 없으면 NaN 처리
    """
    denom = full_price_sales.replace(0, np.nan)
    discount = 1 - (gross_sales / denom)
    return discount.clip(lower=0, upper=1)


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    컬럼명 앞뒤 공백 제거
    """
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    return df


def infer_sheet_format(df: pd.DataFrame) -> str:
    """
    wide:
      연도/주, 가디건, 가방, ...
    long:
      item, 연도/주, 판매수량, 외형매출, 정상가 등
    """
    cols = set(df.columns)

    if "연도/주" in cols and len(cols) >= 2:
        # long 판단용 후보
        long_keywords = {"아이템", "item", "판매수량", "외형매출", "정상가"}
        if len(cols.intersection(long_keywords)) >= 2:
            return "long"
        return "wide"

    return "unknown"


def convert_wide_to_long(df: pd.DataFrame) -> pd.DataFrame:
    """
    예시처럼 가로형 데이터:
    연도/주 | 가디건 | 가방 | ...
    를
    연도/주 | item | 판매수량
    형태로 변환
    """
    if "연도/주" not in df.columns:
        raise ValueError("wide 형식 시트에는 '연도/주' 컬럼이 있어야 합니다.")

    value_cols = [c for c in df.columns if c != "연도/주"]

    long_df = df.melt(
        id_vars=["연도/주"],
        value_vars=value_cols,
        var_name="item",
        value_name="판매수량"
    )

    long_df["판매수량"] = safe_to_numeric(long_df["판매수량"])
    long_df = long_df.dropna(subset=["판매수량"])
    long_df = long_df[long_df["item"].astype(str).str.strip() != ""]

    return long_df


def standardize_long_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    long 형식 데이터를 내부 표준 컬럼으로 맞춤
    내부 표준:
    - year_week
    - item
    - qty
    - gross_sales (optional)
    - full_price_sales (optional)
    """
    df = df.copy()

    col_map = {}
    for c in df.columns:
        c_strip = str(c).strip()
        if c_strip == "연도/주":
            col_map[c] = "year_week"
        elif c_strip.lower() == "item" or c_strip == "아이템":
            col_map[c] = "item"
        elif c_strip == "판매수량":
            col_map[c] = "qty"
        elif c_strip == "외형매출":
            col_map[c] = "gross_sales"
        elif c_strip == "정상가":
            col_map[c] = "full_price_sales"

    df = df.rename(columns=col_map)

    required = {"year_week", "item", "qty"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"필수 컬럼이 없습니다: {sorted(list(missing))}")

    df["qty"] = safe_to_numeric(df["qty"])
    if "gross_sales" in df.columns:
        df["gross_sales"] = safe_to_numeric(df["gross_sales"])
    if "full_price_sales" in df.columns:
        df["full_price_sales"] = safe_to_numeric(df["full_price_sales"])

    df = df.dropna(subset=["year_week", "item", "qty"])
    df["item"] = df["item"].astype(str).str.strip()

    year_week_info = df["year_week"].apply(parse_year_week)
    df["year"] = year_week_info.apply(lambda x: x[0])
    df["week"] = year_week_info.apply(lambda x: x[1])
    df["sort_key"] = df["year_week"].astype(str).apply(yearweek_to_sort_key)

    if "gross_sales" in df.columns and "full_price_sales" in df.columns:
        df["discount_rate"] = calc_discount_rate(df["gross_sales"], df["full_price_sales"])
    else:
        df["discount_rate"] = np.nan

    return df


def prepare_data(raw_df: pd.DataFrame) -> pd.DataFrame:
    raw_df = normalize_columns(raw_df)
    sheet_format = infer_sheet_format(raw_df)

    if sheet_format == "wide":
        temp_df = convert_wide_to_long(raw_df)
        temp_df = temp_df.rename(columns={
            "연도/주": "year_week",
            "판매수량": "qty"
        })

        year_week_info = temp_df["year_week"].apply(parse_year_week)
        temp_df["year"] = year_week_info.apply(lambda x: x[0])
        temp_df["week"] = year_week_info.apply(lambda x: x[1])
        temp_df["sort_key"] = temp_df["year_week"].astype(str).apply(yearweek_to_sort_key)
        temp_df["discount_rate"] = np.nan

        return temp_df

    if sheet_format == "long":
        return standardize_long_columns(raw_df)

    raise ValueError(
        "시트 형식을 인식하지 못했습니다. "
        "현재 지원 형식은 1) 연도/주 + 아이템별 컬럼(wide) 또는 "
        "2) 연도/주, 아이템, 판매수량 기반(long) 입니다."
    )


def is_local_peak(series: pd.Series, idx: int, window: int = 1) -> bool:
    """
    주변 주차보다 뚜렷하게 높은 local peak 판단
    """
    current = series.iloc[idx]
    start = max(0, idx - window)
    end = min(len(series), idx + window + 1)

    neighbors = series.iloc[start:end].copy()
    neighbors = neighbors.drop(series.index[idx], errors="ignore")

    if len(neighbors) == 0:
        return False

    return current > neighbors.max()


def get_recent_slope(values: pd.Series, window: int = 3) -> float:
    """
    최근 흐름 단순 기울기
    예: 최근 3개 값 기준으로 증가/감소 판단
    """
    s = values.dropna()
    if len(s) < 2:
        return 0.0

    s = s.iloc[-window:]
    x = np.arange(len(s))
    y = s.values.astype(float)

    if len(y) < 2:
        return 0.0

    slope = np.polyfit(x, y, 1)[0]
    return float(slope)


def classify_plc_stage(
    idx: int,
    qty_series: pd.Series,
    discount_series: Optional[pd.Series] = None
) -> str:
    """
    사용자가 제시한 기준에 맞춰 주차별 PLC 단계 분류
    우선순위:
    1) 변곡점(최고점)
    2) 쇠퇴
    3) 성숙
    4) 성장
    5) 도입
    """

    current_qty = qty_series.iloc[idx]
    if pd.isna(current_qty):
        return "판단불가"

    max_qty = qty_series.max()
    if max_qty <= 0 or pd.isna(max_qty):
        return "판단불가"

    ratio_to_peak = current_qty / max_qty
    peak_idx = int(qty_series.values.argmax())
    recent_series = qty_series.iloc[max(0, idx - ROLLING_WINDOW + 1): idx + 1]
    recent_slope = get_recent_slope(recent_series, window=ROLLING_WINDOW)

    discount = np.nan
    if discount_series is not None and len(discount_series) > idx:
        discount = discount_series.iloc[idx]

    # 판매 시작 후 첫 3주 계산
    non_zero_idx = np.where(qty_series.fillna(0).values > 0)[0]
    if len(non_zero_idx) > 0:
        start_idx = non_zero_idx[0]
        weeks_since_start = idx - start_idx + 1
    else:
        start_idx = idx
        weeks_since_start = 999

    # local peak 여부
    local_peak = is_local_peak(qty_series.fillna(0), idx, window=LOCAL_PEAK_WINDOW)

    # 1) 최고점은 peak_idx 1개만 별도 표시
    if idx == peak_idx:
        return "변곡점(최고점)"

    # 2) 쇠퇴
    # - 최고점 이후
    # - 최고점 대비 70% 미만
    # - 최근 기울기 음수
    # - 평균 할인율 30% 이상이면서 하락 중
    if idx > peak_idx:
        if (ratio_to_peak < DECLINE_RATIO and recent_slope < 0):
            return "쇠퇴"

        if (not pd.isna(discount)) and (discount >= HIGH_DISCOUNT_RATIO) and (recent_slope < 0):
            return "쇠퇴"

    # 3) 성숙
    # - 최고점 부근
    # - 최고 판매량 대비 85% 이상
    # - 최근 증감률이 크지 않음
    # 최근 기울기가 거의 0에 가까우면 유지라고 판단
    if ratio_to_peak >= MATURE_RATIO:
        if abs(recent_slope) <= max_qty * 0.05:
            return "성숙"
        # 피크 부근이면 기울기가 조금 있어도 성숙 처리 가능
        return "성숙"

    # 4) 성장
    # - 최고점 이전
    # - 35% 이상 ~ 85% 미만
    # - 증가 중
    if idx < peak_idx:
        if (GROWTH_MIN_RATIO <= ratio_to_peak < GROWTH_MAX_RATIO) and (recent_slope > 0):
            return "성장"

    # 5) 도입
    # - 판매 시작 후 첫 3주
    # - 최고 판매량 대비 35% 미만
    # - 완만 증가
    if weeks_since_start <= INTRO_WEEKS:
        return "도입"

    if ratio_to_peak < GROWTH_MIN_RATIO and recent_slope >= 0:
        return "도입"

    # 남는 경우 보정
    # 피크 이전이면 성장, 이후면 쇠퇴 쪽으로 보정
    if idx < peak_idx:
        return "성장"
    elif idx > peak_idx:
        return "쇠퇴"
    else:
        return "성숙"

def enforce_single_intro_decline(item_df: pd.DataFrame) -> pd.DataFrame:
    """
    PLC 단계 흐름을 보정한다.

    기본 흐름:
    도입 -> 성장 -> 비시즌 -> 성장 -> 성숙 -> 쇠퇴

    규칙:
    - 도입은 맨 앞 최대 INTRO_WEEKS까지만 허용
    - 쇠퇴는 맨 뒤 1개 연속 구간만 허용
    - 비시즌은 peak 이전의 낮은 판매 구간 중 회복이 있는 경우만 허용
    """
    df = item_df.copy().reset_index(drop=True)

    if df.empty:
        return df

    peak_idx = int(df["qty"].fillna(0).values.argmax())
    n = len(df)

    # -------------------------
    # 1. 도입 구간 확정
    # -------------------------
    intro_end = -1
    for i in range(n):
        if df.loc[i, "plc_stage_raw"] == "도입":
            intro_end = i
        else:
            break

    intro_end = min(intro_end, INTRO_WEEKS - 1)

    for i in range(intro_end + 1, n):
        if df.loc[i, "plc_stage_raw"] == "도입":
            df.loc[i, "plc_stage_raw"] = "성장" if i < peak_idx else "성숙"

    # -------------------------
    # 2. peak 이전 쇠퇴 금지
    # -------------------------
    for i in range(0, peak_idx):
        if df.loc[i, "plc_stage_raw"] == "쇠퇴":
            df.loc[i, "plc_stage_raw"] = "성장"

    # -------------------------
    # 3. 끝 연속 쇠퇴만 인정
    # -------------------------
    decline_start = n
    for i in range(n - 1, -1, -1):
        if df.loc[i, "plc_stage_raw"] == "쇠퇴":
            decline_start = i
        else:
            break

    for i in range(0, decline_start):
        if df.loc[i, "plc_stage_raw"] == "쇠퇴":
            df.loc[i, "plc_stage_raw"] = "성숙" if i >= peak_idx else "성장"

    # -------------------------
    # 4. peak 전후 기본 정리
    # -------------------------
    for i in range(n):
        stage = df.loc[i, "plc_stage_raw"]

        if i < peak_idx:
            if stage == "쇠퇴":
                df.loc[i, "plc_stage_raw"] = "성장"
        elif i > peak_idx:
            if stage == "도입":
                df.loc[i, "plc_stage_raw"] = "성숙"

    # -------------------------
    # 5. 최종 컬럼 준비
    # -------------------------
    df["plc_stage_final"] = df["plc_stage_raw"]

    # -------------------------
    # 6. 비시즌 표시
    # -------------------------
    df = mark_off_season_stage(df)

    # -------------------------
    # 7. 최종 흐름 정리
    # -------------------------
    off_idx = df.index[df["plc_stage_final"] == "비시즌"].tolist()

    if off_idx:
        first_off = min(off_idx)
        last_off = max(off_idx)

        # 도입 뒤 ~ 비시즌 전 = 성장
        for i in range(intro_end + 1, first_off):
            if df.loc[i, "plc_stage_final"] not in ["도입", "비시즌"]:
                df.loc[i, "plc_stage_final"] = "성장"

        # 비시즌 뒤 ~ peak 전 = 성장
        for i in range(last_off + 1, peak_idx):
            if df.loc[i, "plc_stage_final"] != "비시즌":
                df.loc[i, "plc_stage_final"] = "성장"

        # peak 이후 ~ 쇠퇴 전 = 성숙
        for i in range(peak_idx + 1, decline_start):
            if df.loc[i, "plc_stage_final"] != "쇠퇴":
                df.loc[i, "plc_stage_final"] = "성숙"

    else:
        # 비시즌 없으면 기존 구조 유지
        for i in range(intro_end + 1, peak_idx):
            df.loc[i, "plc_stage_final"] = "성장"

        for i in range(peak_idx + 1, decline_start):
            df.loc[i, "plc_stage_final"] = "성숙"

    # peak 주차는 성숙으로 두고, 마커만 별도로 표시
    df.loc[peak_idx, "plc_stage_final"] = "성숙"

    # 마지막 쇠퇴 구간 유지
    for i in range(decline_start, n):
        df.loc[i, "plc_stage_final"] = "쇠퇴"

    df["plc_stage"] = df["plc_stage_final"]

    return df




def build_item_plc(item_df: pd.DataFrame) -> pd.DataFrame:
    item_df = item_df.sort_values("sort_key").reset_index(drop=True).copy()

    qty_series = item_df["qty"].fillna(0)
    discount_series = item_df["discount_rate"] if "discount_rate" in item_df.columns else pd.Series([np.nan] * len(item_df))

    item_df["peak_qty"] = qty_series.max()
    item_df["ratio_to_peak"] = np.where(item_df["peak_qty"] > 0, item_df["qty"] / item_df["peak_qty"], np.nan)

    item_df["recent_slope"] = [
        get_recent_slope(qty_series.iloc[max(0, i - ROLLING_WINDOW + 1): i + 1], window=ROLLING_WINDOW)
        for i in range(len(item_df))
    ]

    item_df["plc_stage_raw"] = [
    classify_plc_stage(i, qty_series, discount_series)
    for i in range(len(item_df))
    ]

    peak_idx = int(qty_series.values.argmax()) if len(qty_series) > 0 else None
    item_df["is_peak_week"] = False
    if peak_idx is not None:
        item_df.loc[peak_idx, "is_peak_week"] = True
    
    # 변곡점(최고점)은 마커로만 표시하고,
    # plc_stage는 4단계 체계로 정리
    item_df["plc_stage_raw"] = item_df["plc_stage_raw"].replace("변곡점(최고점)", "성숙")
    
    item_df = enforce_single_intro_decline(item_df)
    
    return item_df
    
    
    
   


def summarize_latest_status(all_plc_df: pd.DataFrame) -> pd.DataFrame:
    latest_df = (
        all_plc_df.sort_values(["item", "sort_key"])
        .groupby("item", as_index=False)
        .tail(1)
        .copy()
    )

    latest_df = latest_df.rename(columns={
        "year_week": "최신주차",
        "qty": "최신판매수량",
        "peak_qty": "최고판매수량",
        "ratio_to_peak": "최고점대비비율",
        "discount_rate": "최신할인율",
        "plc_stage": "현재PLC"
    })

    cols = ["item", "최신주차", "최신판매수량", "최고판매수량", "최고점대비비율", "최신할인율", "현재PLC"]
    existing_cols = [c for c in cols if c in latest_df.columns]
    latest_df = latest_df[existing_cols].sort_values("item").reset_index(drop=True)

    return latest_df


def make_stage_reason(row: pd.Series) -> str:
    stage = row.get("plc_stage", "")
    qty = row.get("qty", np.nan)
    peak = row.get("peak_qty", np.nan)
    ratio = row.get("ratio_to_peak", np.nan)
    slope = row.get("recent_slope", np.nan)
    discount = row.get("discount_rate", np.nan)

    ratio_pct = f"{ratio * 100:.1f}%" if pd.notna(ratio) else "-"
    discount_pct = f"{discount * 100:.1f}%" if pd.notna(discount) else "-"

    if stage == "도입":
        return (
            f"판매 시작 초기 구간으로 판단했습니다. "
            f"현재 판매수량은 {qty:.0f}, 최고점 대비 {ratio_pct} 수준입니다. "
            f"최근 흐름은 완만한 증가로 보고 도입 단계로 분류했습니다."
        )
    elif stage == "성장":
        return (
            f"최고점 이전 상승 구간으로 판단했습니다. "
            f"현재 판매수량은 {qty:.0f}, 최고점 대비 {ratio_pct} 수준이며 "
            f"최근 흐름이 증가 중이어서 성장 단계로 분류했습니다."
        )
    elif stage == "성숙":
        return (
            f"최고점 부근의 유지 구간으로 판단했습니다. "
            f"현재 판매수량은 {qty:.0f}, 최고점 대비 {ratio_pct} 수준입니다. "
            f"잘 팔리는 수준이 유지되고 있어 성숙 단계로 분류했습니다."
        )
    elif stage == "변곡점(최고점)":
        return (
            f"해당 주차가 전체 기간 중 최고 판매량이거나 주변 주차보다 뚜렷하게 높은 구간입니다. "
            f"현재 판매수량은 {qty:.0f}이며 정점 주차로 판단했습니다."
        )
    elif stage == "쇠퇴":
        return (
            f"최고점 이후 하락 구간으로 판단했습니다. "
            f"현재 판매수량은 {qty:.0f}, 최고점 대비 {ratio_pct} 수준입니다. "
            f"최근 흐름이 하락 중이며 할인율은 {discount_pct}입니다."
        )
    else:
        return "현재 데이터만으로 단계 판단이 어렵습니다."


PLC_LINE_COLOR_MAP = {
    "도입": "#1f77b4",          # 파랑
    "성장": "#2ca02c",          # 초록
    "성숙": "#ff7f0e",          # 주황
    "쇠퇴": "#d62728",          # 빨강
    "변곡점(최고점)": "#8c564b",  # 갈색
    "비시즌": "#7f7f7f"  # 회색
}

def draw_item_chart(item_df: pd.DataFrame, item_name: str) -> go.Figure:
    fig = go.Figure()

    plot_df = item_df.sort_values("sort_key").reset_index(drop=True).copy()

    # PLC 단계가 바뀌는 지점마다 선을 끊어서 별도 trace로 그림
    segment_start = 0
    segment_id = 0

    for i in range(1, len(plot_df)):
        prev_stage = plot_df.loc[i - 1, "plc_stage"]
        curr_stage = plot_df.loc[i, "plc_stage"]

        if prev_stage != curr_stage:
            seg = plot_df.iloc[segment_start:i].copy()
            if len(seg) > 0:
                stage = seg["plc_stage"].iloc[0]
                fig.add_trace(
                    go.Scatter(
                        x=seg["year_week"],
                        y=seg["qty"],
                        mode="lines+markers",
                        name=stage if segment_id == 0 or stage not in [t.name for t in fig.data] else stage,
                        line=dict(
                            color=PLC_LINE_COLOR_MAP.get(stage, "#6b7280"),
                            width=3
                        ),
                        marker=dict(
                            size=8,
                            color=PLC_LINE_COLOR_MAP.get(stage, "#6b7280")
                        ),
                        legendgroup=stage,
                        showlegend=stage not in [t.name for t in fig.data],
                        hovertemplate=(
                            "<b>%{x}</b><br>"
                            "판매수량: %{y:,.0f}<br>"
                            f"PLC: {stage}"
                            "<extra></extra>"
                        )
                    )
                )
                segment_id += 1
            segment_start = i - 1

    # 마지막 구간 추가
    last_seg = plot_df.iloc[segment_start:].copy()
    if len(last_seg) > 0:
        stage = last_seg["plc_stage"].iloc[0]
        fig.add_trace(
            go.Scatter(
                x=last_seg["year_week"],
                y=last_seg["qty"],
                mode="lines+markers",
                name=stage,
                line=dict(
                    color=PLC_LINE_COLOR_MAP.get(stage, "#6b7280"),
                    width=3
                ),
                marker=dict(
                    size=8,
                    color=PLC_LINE_COLOR_MAP.get(stage, "#6b7280")
                ),
                legendgroup=stage,
                showlegend=stage not in [t.name for t in fig.data],
                hovertemplate=(
                    "<b>%{x}</b><br>"
                    "판매수량: %{y:,.0f}<br>"
                    f"PLC: {stage}"
                    "<extra></extra>"
                )
            )
        )

    # 변곡점(최고점) 표시
    peak_rows = plot_df[plot_df["is_peak_week"] == True]
    if len(peak_rows) > 0:
        fig.add_trace(
            go.Scatter(
                x=peak_rows["year_week"],
                y=peak_rows["qty"],
                mode="markers+text",
                name="변곡점(최고점)",
                text=["최고점"] * len(peak_rows),
                textposition="top center",
                marker=dict(
                    size=13,
                    symbol="diamond",
                    color=PLC_LINE_COLOR_MAP.get("변곡점(최고점)", "#8c564b"),
                    line=dict(color="white", width=1)
                ),
                hovertemplate=(
                    "<b>%{x}</b><br>"
                    "판매수량: %{y:,.0f}<br>"
                    "PLC: 변곡점(최고점)"
                    "<extra></extra>"
                )
            )
        )

    fig.update_layout(
        title=f"{item_name} 주차별 판매수량 흐름",
        xaxis_title="연도/주",
        yaxis_title="판매수량",
        hovermode="x unified",
        height=480,
        legend_title="PLC 단계"
    )

    fig.update_xaxes(type="category")

    return fig



# =========================
# 4) 구글시트 읽기
# =========================
@st.cache_data(ttl=300)
def load_sheet_data() -> pd.DataFrame:
    sheets_cfg = get_sheets_config()

    sheet_id = sheets_cfg.get("sheet_id")
    worksheet_name = (
        sheets_cfg.get("WORKSHEET_NAME")
        or sheets_cfg.get("worksheet")
        or sheets_cfg.get("forecast_base_sheet")
        or "plc db"
    )

    if not sheet_id:
        raise ValueError("secrets.toml의 [sheets].sheet_id 가 비어있습니다.")

    return load_sheet_as_df(worksheet_name)



# =========================
# 5) 데이터 처리
# =========================
try:
    raw_df = load_sheet_data()
    st.success("구글시트 데이터를 불러왔습니다.")

    with st.expander("원본 데이터 미리보기", expanded=False):
        st.dataframe(raw_df.head(20), use_container_width=True)

    prepared_df = prepare_data(raw_df)

    # item별 PLC 계산
    item_result_list: List[pd.DataFrame] = []
    for item_name, g in prepared_df.groupby("item"):
        item_plc_df = build_item_plc(g)
        item_result_list.append(item_plc_df)

    if not item_result_list:
        st.error("분석 가능한 아이템 데이터가 없습니다.")
        st.stop()

    plc_df = pd.concat(item_result_list, ignore_index=True)

except Exception as e:
    st.error(f"오류 발생: {e}")
    st.stop()


# =========================
# 6) 상단 요약
# =========================
latest_summary_df = summarize_latest_status(plc_df)

st.subheader("아이템별 현재 PLC 요약")

col1, col2 = st.columns([2, 1])
with col1:
    keyword = st.text_input("아이템 검색", value="")
with col2:
    stage_filter = st.multiselect(
        "현재 PLC 필터",
        options=sorted(latest_summary_df["현재PLC"].dropna().unique().tolist()),
        default=[]
    )

filtered_summary = latest_summary_df.copy()

if keyword.strip():
    filtered_summary = filtered_summary[
        filtered_summary["item"].astype(str).str.contains(keyword.strip(), case=False, na=False)
    ]

if stage_filter:
    filtered_summary = filtered_summary[
        filtered_summary["현재PLC"].isin(stage_filter)
    ]

display_summary = filtered_summary.copy()
if "최고점대비비율" in display_summary.columns:
    display_summary["최고점대비비율"] = (display_summary["최고점대비비율"] * 100).round(1).astype(str) + "%"
if "최신할인율" in display_summary.columns:
    display_summary["최신할인율"] = np.where(
        display_summary["최신할인율"].notna(),
        (display_summary["최신할인율"] * 100).round(1).astype(str) + "%",
        "-"
    )

st.dataframe(display_summary, use_container_width=True, hide_index=True)


# =========================
# 7) 상세 분석
# =========================
st.subheader("아이템 상세 분석")

item_list = sorted(plc_df["item"].dropna().unique().tolist())
selected_item = st.selectbox("아이템 선택", item_list)

item_df = plc_df[plc_df["item"] == selected_item].sort_values("sort_key").reset_index(drop=True)

if item_df.empty:
    st.warning("선택한 아이템 데이터가 없습니다.")
    st.stop()

latest_row = item_df.iloc[-1]
peak_row = item_df.loc[item_df["qty"].idxmax()]

m1, m2, m3, m4 = st.columns(4)
m1.metric("현재 PLC", latest_row["plc_stage"])
m2.metric("최신 판매수량", f"{latest_row['qty']:.0f}")
m3.metric("최고 판매수량", f"{peak_row['qty']:.0f}")
m4.metric("최고점 주차", str(peak_row["year_week"]))

fig = draw_item_chart(item_df, selected_item)
st.plotly_chart(fig, use_container_width=True)

st.markdown("### 최신 주차 판단 근거")
st.write(make_stage_reason(latest_row))

with st.expander("주차별 상세 데이터", expanded=False):
    detail_df = item_df.copy()
    detail_df["ratio_to_peak"] = (detail_df["ratio_to_peak"] * 100).round(1)
    if "discount_rate" in detail_df.columns:
        detail_df["discount_rate"] = np.where(
            detail_df["discount_rate"].notna(),
            (detail_df["discount_rate"] * 100).round(1),
            np.nan
        )

    show_cols = [
        "year_week", "qty", "peak_qty", "ratio_to_peak",
        "recent_slope", "discount_rate", "plc_stage"
    ]
    show_cols = [c for c in show_cols if c in detail_df.columns]
    st.dataframe(detail_df[show_cols], use_container_width=True, hide_index=True)


# =========================
# 8) 기준 설명
# =========================
st.subheader("PLC 분류 기준")

st.markdown(
    """
**도입**
- 판매 시작 후 첫 3주 이내
- 최고 판매량 대비 35% 미만
- 최근 흐름이 완만하게 증가 중

**성장**
- 최고점 이전 구간
- 최고 판매량 대비 35% 이상 ~ 85% 미만
- 최근 판매량이 이전보다 증가 중

**성숙**
- 최고점 부근
- 최고 판매량 대비 85% 이상
- 최근 증감률이 크지 않음

**변곡점(최고점)**
- 전체 기간 중 가장 높은 판매량 주차
- 또는 주변 주차보다 뚜렷하게 높은 local peak

**쇠퇴**
- 최고점 이후 구간
- 최고점 대비 70% 미만
- 최근 증감률 음수
- 평균 할인율 30% 이상이면서 하락 중
"""
)

st.markdown(
    """
**할인율 계산식**
- 할인율 = 1 - (외형매출 / 정상가)

예시:
- 정상가 100,000
- 외형매출 80,000
- 할인율 = 1 - 80,000 / 100,000 = 0.2
- 즉 20% 할인
"""
)
