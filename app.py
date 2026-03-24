import os
import json
import re
import urllib.error
import urllib.request
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import streamlit as st
import gspread
from google.oauth2.service_account import Credentials
from streamlit_gsheets import GSheetsConnection
from typing import List, Optional, Tuple

# =========================
# 1. 설정값 / 상수
# =========================
# 도입: 시작 무조건 도입, 최대 5주, 전주대비 급상승 시 도입 종료
INTRO_MAX_WEEKS = 5
INTRO_MAX_WOW_RATIO = 0.30  # 전주 대비 증가율이 이 값 초과면 도입에서 제외(급상승)

# 성장: 연속 3주 동안 전주대비 증가율이 계속 커지는 패턴(가속)
GROWTH_ACCEL_MIN_STREAK = 3

# 비시즌 (연속 5주 이상, 1구간만 선택)
OFF_SEASON_MIN_WEEKS = 5
OFF_SEASON_RATIO_TO_MEAN = 0.60  # ma5 <= 전체평균 * 이 값
OFF_SEASON_EACH_WEEK_MAX_TO_MEAN = 0.50  # 매주 판매량 <= 전체평균 * 이 값
OFF_SEASON_SLOPE_FRAC = 0.15  # |전주대비 증가량| = |q[i]-q[i-1]| <= 평균 * 이 값 (완만)
MATURE_SLOPE_FRAC = 0.20  # 성숙(plateau): |q[i]-q[i-1]| <= 평균 * 이 값
OFF_SEASON_WINDOW = 5  # ma5 윈도우

NORMALTEST_MIN_N = 8  # 정규성 검정 최소 표본

PLC_LINE_COLOR_MAP = {
    "도입": "#1f77b4",  # 파랑
    "성장": "#2ca02c",  # 초록
    "성숙": "#ff7f0e",  # 주황
    "쇠퇴": "#d62728",  # 빨강
    "변곡점(최고점)": "#8c564b",  # 갈색
    "비시즌": "#7f7f7f",  # 회색
}

OPENAI_CHAT_URL = "https://api.openai.com/v1/chat/completions"


def get_gpt_gpi() -> Optional[str]:
    """
    Streamlit secrets의 gpt_gpi(또는 OPENAI_API_KEY) 또는 환경변수.
    secrets.toml 예: gpt_gpi = "sk-..."
    """
    try:
        if hasattr(st, "secrets"):
            sec = st.secrets
            if "gpt_gpi" in sec:
                v = sec["gpt_gpi"]
                if v is not None and str(v).strip():
                    return str(v).strip()
            if "OPENAI_API_KEY" in sec:
                v = sec["OPENAI_API_KEY"]
                if v is not None and str(v).strip():
                    return str(v).strip()
    except Exception:
        pass
    return (os.getenv("gpt_gpi") or os.getenv("OPENAI_API_KEY") or "").strip() or None


# =========================
# 2. 유틸 함수
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


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    컬럼명 앞뒤 공백 제거
    """
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    return df


def calc_discount_rate(gross_sales: pd.Series, full_price_sales: pd.Series) -> pd.Series:
    """
    할인율 = 1 - (외형매출 / 정상가)
    정상가가 0이거나 없으면 NaN 처리
    """
    denom = full_price_sales.replace(0, np.nan)
    discount = 1 - (gross_sales / denom)
    return discount.clip(lower=0, upper=1)


def make_unique_headers(headers: List[str]) -> List[str]:
    """
    헤더 중복을 방지하기 위해 동일한 이름에 suffix를 붙인다.
    예: 다운, 다운 -> 다운, 다운_2
    """
    counts = {}
    unique = []

    for idx, h in enumerate(headers):
        name = str(h).strip()
        if name == "":
            name = f"unnamed_{idx + 1}"

        if name not in counts:
            counts[name] = 1
            unique.append(name)
        else:
            counts[name] += 1
            unique.append(f"{name}_{counts[name]}")

    return unique


# =========================
# 3. 데이터 전처리 유틸
# =========================
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


# =========================
# 4. 분석 보조 유틸
# =========================
def week_over_week_rate(qty: np.ndarray) -> np.ndarray:
    """
    전주 대비 증가율: (q[t] - q[t-1]) / max(q[t-1], eps)
    첫 주차는 NaN
    """
    q = np.asarray(qty, dtype=float)
    n = len(q)
    w = np.full(n, np.nan)
    eps = 1e-9
    for i in range(1, n):
        prev = q[i - 1]
        if prev <= eps:
            w[i] = np.inf if q[i] > eps else 0.0
        else:
            w[i] = (q[i] - prev) / prev
    return w


def growth_rate_accelerating_mask(wow: np.ndarray) -> np.ndarray:
    """
    연속 GROWTH_ACCEL_MIN_STREAK주 동안 전주대비 증가율이 단조 증가하는 구간 마스크.
    """
    n = len(wow)
    k = GROWTH_ACCEL_MIN_STREAK
    mask = np.zeros(n, dtype=bool)
    if k < 2 or n < k:
        return mask
    for i in range(k - 1, n):
        vals = [wow[i - k + 1 + j] for j in range(k)]
        if any(np.isnan(v) for v in vals):
            continue
        if all(vals[j] < vals[j + 1] for j in range(k - 1)):
            mask[i - k + 1 : i + 1] = True
    return mask


def pick_peak_idx(q: np.ndarray, wow: np.ndarray) -> int:
    """
    피크: 최대 판매량 주차. 동률이면 전주대비 증가율(wow)이 더 큰(급한) 주차.
    """
    n = len(q)
    max_q = float(np.max(q))
    cands = [i for i in range(n) if float(q[i]) >= max_q - 1e-9]
    best = cands[0]
    best_wow = -np.inf
    for c in cands:
        wv = wow[c]
        if np.isnan(wv) or np.isinf(wv):
            wv = -np.inf
        if wv > best_wow:
            best_wow = wv
            best = c
    return best


def sales_look_normal(q: np.ndarray) -> bool:
    """
    정규분포에 가깝다고 보면 비시즌 없음.
    scipy 있으면 Shapiro, 없으면 왜도/첨도 휴리스틱.
    """
    x = np.asarray(q, dtype=float)
    x = x[~np.isnan(x)]
    if len(x) < NORMALTEST_MIN_N:
        return False
    try:
        from scipy.stats import shapiro

        _, p = shapiro(x)
        return bool(p > 0.05)
    except Exception:
        s = float(pd.Series(x).skew())
        k = float(pd.Series(x).kurtosis())
        return abs(s) < 0.75 and abs(k) < 1.2


def find_off_season_segment(
    q: np.ndarray,
    intro_end: int,
    peak_idx: int,
) -> Optional[Tuple[int, int]]:
    """
    비시즌 (선택, 피크 직전 중간 저점 1구간만):
    - 판매량이 정규에 가깝지 않을 때만
    - 연속 OFF_SEASON_MIN_WEEKS주 이상
    - 매주 판매량 <= 전체평균 * OFF_SEASON_EACH_WEEK_MAX_TO_MEAN
    - ma5 <= 전체평균 * OFF_SEASON_RATIO_TO_MEAN
    - |q[i]-q[i-1]| <= 전체평균 * OFF_SEASON_SLOPE_FRAC (완만)
    - 인덱스는 도입 이후 ~ peak_idx-2 까지만 (피크 앞 성장·성숙 공간 확보)
    - 후보가 여러 개면 구간 평균 판매량이 가장 낮은 연속 구간 1개
    """
    n = len(q)
    if n < OFF_SEASON_MIN_WEEKS or peak_idx < 2:
        return None
    if sales_look_normal(q):
        return None
    avg = float(np.mean(q))
    if avg <= 1e-9:
        return None

    ma5 = pd.Series(q).rolling(OFF_SEASON_WINDOW, min_periods=1).mean().values
    cap_q = avg * OFF_SEASON_EACH_WEEK_MAX_TO_MEAN
    cap_ma = avg * OFF_SEASON_RATIO_TO_MEAN
    cap_dq = avg * OFF_SEASON_SLOPE_FRAC

    hi = peak_idx - 1
    if hi <= intro_end + 1:
        return None

    mask = np.zeros(n, dtype=bool)
    for i in range(intro_end + 1, hi):
        if q[i] > cap_q + 1e-9:
            continue
        if ma5[i] > cap_ma + 1e-9:
            continue
        if i > 0:
            if abs(q[i] - q[i - 1]) > cap_dq + 1e-9:
                continue
        mask[i] = True

    runs: List[Tuple[int, int]] = []
    i = 0
    while i < n:
        if not mask[i]:
            i += 1
            continue
        j = i
        while j < n and mask[j]:
            j += 1
        if j - i >= OFF_SEASON_MIN_WEEKS:
            runs.append((i, j - 1))
        i = j

    if not runs:
        return None
    return min(runs, key=lambda lr: float(np.mean(q[lr[0] : lr[1] + 1])))


def _intro_end_from_wow(wow: np.ndarray, n: int) -> int:
    """시작은 도입, 최대 INTRO_MAX_WEEKS주, 급상승(INTRO_MAX_WOW_RATIO) 시 직전 주까지 도입."""
    if n <= 0:
        return -1
    end = 0
    last_intro_idx = min(n - 1, INTRO_MAX_WEEKS - 1)
    for i in range(1, last_intro_idx + 1):
        w = wow[i]
        if np.isfinite(w) and w >= INTRO_MAX_WOW_RATIO:
            break
        end = i
    return end


def _enforce_core_stages(
    stages: List[str],
    q: np.ndarray,
    wow: np.ndarray,
    peak_idx: int,
    intro_end: int,
    n: int,
) -> None:
    """도입·성장·성숙·쇠퇴를 가능한 한 반드시 1주 이상 보장(비시즌은 선택)."""
    if n <= 0:
        return

    # 1) 도입은 시작 고정
    stages[0] = "도입"

    # 2) 성숙은 피크 주차(가능하면 앞뒤 1주 포함)
    peak_lo = max(0, peak_idx - 1)
    peak_hi = min(n - 1, peak_idx + 1)
    for i in range(peak_lo, peak_hi + 1):
        stages[i] = "성숙"

    # 3) 쇠퇴가 없으면 마지막 주를 쇠퇴로 강제
    if n >= 2 and "쇠퇴" not in set(stages):
        stages[n - 1] = "쇠퇴"

    # 4) 성장이 없으면 도입 끝 이후~피크 이전 우선, 없으면 도입/성숙/쇠퇴가 아닌 임의 주차에 배치
    if "성장" not in set(stages):
        growth_idx: Optional[int] = None
        for i in range(max(1, intro_end + 1), peak_idx):
            if stages[i] not in {"도입", "성숙", "쇠퇴"}:
                growth_idx = i
                break
        if growth_idx is None:
            for i in range(1, n):
                if stages[i] not in {"도입", "성숙", "쇠퇴"}:
                    growth_idx = i
                    break
        if growth_idx is None and n >= 3:
            growth_idx = 1 if peak_idx != 1 else 2
        if growth_idx is not None and 0 <= growth_idx < n:
            stages[growth_idx] = "성장"

    # 5) 성숙/쇠퇴 최종 재보장
    if "성숙" not in set(stages):
        stages[min(max(peak_idx, 0), n - 1)] = "성숙"
    if n >= 2 and "쇠퇴" not in set(stages):
        stages[n - 1] = "쇠퇴"


def classify_plc_with_openai(
    item_df: pd.DataFrame,
    api_key: str,
    model: str = "gpt-4o-mini",
) -> Optional[List[str]]:
    """OpenAI Chat Completions로 주차별 PLC 라벨 요청. 성공 시 길이 n 리스트, 실패 시 None."""
    df = item_df.sort_values("sort_key").reset_index(drop=True)
    n = len(df)
    if n == 0:
        return None
    series = [
        {"year_week": str(df.loc[i, "year_week"]), "qty": float(df.loc[i, "qty"] or 0)}
        for i in range(n)
    ]
    rules = (
        "도입: 시계열 시작은 무조건 도입, 최대 5주, 전주대비 증가율 30% 이상이면 급상승으로 도입 종료. "
        "성장: 전주대비 증가율이 3주 이상 커지는 가속 구간, 평균 돌파·급상승 구간. "
        "성숙(plateau): 평균 이상에서 주간 변화가 완만하고 큰 상승·하락 없음; peak 전후 최소 각 1주는 성숙. "
        "쇠퇴: peak 이후(peak 직다음 주는 성숙으로 둘 수 있음) 감소·평균 이하 추세. "
        "비시즌: 연속 5주 이상, 매주<=전체평균의 50%, 5주이동평균<=전체평균의 60%, 변화 완만; "
        "가능하면 피크 전 가장 낮은 바닥 1구간만. "
        "허용 stage 문자열: 도입, 성장, 성숙, 쇠퇴, 비시즌 만 사용."
    )
    payload = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": (
                    "제품생명주기(PLC) 분류기. 입력 series 순서와 동일한 길이의 stages 배열만 JSON으로 반환. "
                    '형식: {"stages":["도입",...]}'
                ),
            },
            {
                "role": "user",
                "content": json.dumps({"rules_ko": rules, "series": series}, ensure_ascii=False),
            },
        ],
        "temperature": 0.2,
    }
    req = urllib.request.Request(
        OPENAI_CHAT_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        text = data["choices"][0]["message"]["content"]
        m = re.search(r"\{[\s\S]*\}", text)
        if not m:
            return None
        obj = json.loads(m.group(0))
        stages = obj.get("stages")
        if not isinstance(stages, list) or len(stages) != n:
            return None
        return [str(s) for s in stages]
    except (urllib.error.URLError, json.JSONDecodeError, KeyError, IndexError, TypeError):
        return None


def classify_plc(
    item_df: pd.DataFrame,
    *,
    use_openai: bool = False,
    openai_model: str = "gpt-4o-mini",
) -> pd.DataFrame:
    """
    PLC 단계 분류 (전주대비 증가율·증가량·이동평균·피크·비시즌 1구간 반영).

    - 도입: 시작 고정, 최대 5주, 급상승(전주대비 증가율 >= INTRO_MAX_WOW_RATIO) 전까지
    - 성장: 가속 마스크(growth_accel) 우선, 그 외 피크 이전·비시즌·도입 제외 구간은 성장 기본
    - 성숙: 평균 이상 + 주간 변화량 <= 평균*MATURE_SLOPE_FRAC; 피크 앞뒤 1주(peak±1)는 항상 성숙
    - 쇠퇴: peak_idx+2 ~ 끝 (peak 직후 1주는 성숙 유지)
    - 비시즌: find_off_season_segment (5주 이상·1구간)
    - use_openai=True 이고 secrets gpt_gpi 가 있으면 API 결과로 덮어쓰기 시도, 실패 시 규칙 기반
    """
    df = item_df.sort_values("sort_key").reset_index(drop=True).copy()
    q = df["qty"].fillna(0).values.astype(float)
    n = len(q)
    wow = week_over_week_rate(q)
    df["recent_slope"] = wow
    df["growth_accel"] = growth_rate_accelerating_mask(wow)

    if n == 0:
        df["plc_stage"] = pd.Series(dtype=object)
        df["peak_qty"] = np.nan
        df["ratio_to_peak"] = np.nan
        df["is_peak_week"] = False
        return df

    peak_qty = float(np.max(q))
    df["peak_qty"] = peak_qty
    df["ratio_to_peak"] = np.where(peak_qty > 0, q / peak_qty, np.nan)
    peak_idx = pick_peak_idx(q, wow)
    avg = float(np.mean(q))

    if use_openai:
        key = get_gpt_gpi()
        if key:
            llm = classify_plc_with_openai(df, key, model=openai_model)
            if llm:
                df["plc_stage"] = llm
                df["is_peak_week"] = False
                df.loc[peak_idx, "is_peak_week"] = True
                allowed = {"도입", "성장", "성숙", "쇠퇴", "비시즌"}
                for i in range(n):
                    if df.loc[i, "plc_stage"] not in allowed:
                        df.loc[i, "plc_stage"] = "성장"
                if str(df.loc[peak_idx, "plc_stage"]) != "성숙":
                    df.loc[peak_idx, "plc_stage"] = "성숙"
                enforced = df["plc_stage"].astype(str).tolist()
                _enforce_core_stages(enforced, q, wow, peak_idx, _intro_end_from_wow(wow, n), n)
                df["plc_stage"] = enforced
                return df

    if n < 4:
        stages = ["도입"] * n
        if n >= 2:
            stages[1] = "성장"
        if n >= 3:
            stages[2] = "성숙"
        df["plc_stage"] = stages
        df["is_peak_week"] = False
        df.loc[peak_idx, "is_peak_week"] = True
        return df

    intro_end = _intro_end_from_wow(wow, n)
    off_seg = find_off_season_segment(q, intro_end, peak_idx)

    slope_abs = np.zeros(n)
    slope_abs[1:] = np.abs(q[1:] - q[:-1])
    mature_cap = avg * MATURE_SLOPE_FRAC

    stages = [""] * n

    for i in range(peak_idx + 2, n):
        stages[i] = "쇠퇴"

    if off_seg is not None:
        s_off, e_off = off_seg
        for j in range(s_off, e_off + 1):
            stages[j] = "비시즌"

    for i in range(0, intro_end + 1):
        stages[i] = "도입"

    for i in range(max(0, peak_idx - 1), min(n, peak_idx + 2)):
        stages[i] = "성숙"

    growth_accel = df["growth_accel"].values

    for i in range(n):
        if stages[i]:
            continue
        if i <= intro_end:
            continue
        if i >= peak_idx + 2:
            continue
        if max(0, peak_idx - 1) <= i <= min(n - 1, peak_idx + 1):
            continue

        if growth_accel[i]:
            stages[i] = "성장"
        elif i > 0 and q[i] >= avg - 1e-9 and slope_abs[i] <= mature_cap + 1e-9:
            stages[i] = "성숙"
        else:
            stages[i] = "성장"

    for i in range(max(0, peak_idx - 1), min(n, peak_idx + 2)):
        stages[i] = "성숙"

    _enforce_core_stages(stages, q, wow, peak_idx, intro_end, n)

    df["plc_stage"] = stages
    df["is_peak_week"] = False
    df.loc[peak_idx, "is_peak_week"] = True
    return df


def assign_plc_stages(item_df: pd.DataFrame) -> pd.DataFrame:
    """세션 옵션에 따라 classify_plc 호출 (Streamlit session_state)."""
    use_gpt = bool(st.session_state.get("use_gpt_plc", False))
    model = str(st.session_state.get("openai_plc_model", "gpt-4o-mini"))
    return classify_plc(item_df, use_openai=use_gpt, openai_model=model)


def build_item_plc(item_df: pd.DataFrame) -> pd.DataFrame:
    item_df = item_df.sort_values("sort_key").reset_index(drop=True).copy()

    qty_series = item_df["qty"].fillna(0)

    item_df["peak_qty"] = qty_series.max()
    item_df["ratio_to_peak"] = np.where(item_df["peak_qty"] > 0, item_df["qty"] / item_df["peak_qty"], np.nan)

    item_df = assign_plc_stages(item_df)

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
    ratio = row.get("ratio_to_peak", np.nan)
    slope = row.get("recent_slope", np.nan)
    accel = row.get("growth_accel", False)

    ratio_pct = f"{ratio * 100:.1f}%" if pd.notna(ratio) else "-"
    slope_pct = f"{slope * 100:.1f}%" if pd.notna(slope) and np.isfinite(slope) else "-"

    if stage == "도입":
        return (
            f"시작 구간은 무조건 도입으로 두었습니다. "
            f"도입은 최대 {INTRO_MAX_WEEKS}주이며, 전주 대비 증가율이 "
            f"{INTRO_MAX_WOW_RATIO:.0%}을 넘는 급상승이 나오면 도입을 종료합니다. "
            f"현재 판매수량 {qty:.0f}, 최고점 대비 {ratio_pct}, 전주 대비 증가율 {slope_pct}입니다."
        )
    elif stage == "성장":
        accel_txt = "연속 3주 동안 전주 대비 증가율이 커지는 가속 패턴이 있습니다. " if accel else ""
        return (
            f"피크 이전 성장 구간입니다. {accel_txt}"
            f"성장기 판단에는 전주 대비 증가율(증가량이 아닌 비율) 추이를 사용합니다. "
            f"현재 판매수량 {qty:.0f}, 최고점 대비 {ratio_pct}, 전주 대비 증가율 {slope_pct}입니다."
        )
    elif stage == "성숙":
        return (
            f"피크를 포함하는 성숙 구간입니다. 피크 전·후 최소 1주씩 포함하고, "
            f"가능하면 구간 내 매주 판매량이 전체 평균보다 높은 후보를 고릅니다. "
            f"현재 판매수량 {qty:.0f}, 최고점 대비 {ratio_pct}입니다."
        )
    elif stage == "비시즌":
        return (
            f"판매량 분포가 정규에 가깝지 않을 때만 비시즌을 둡니다. "
            f"연속 {OFF_SEASON_MIN_WEEKS}주 이상, 매주 판매량이 평균의 {OFF_SEASON_EACH_WEEK_MAX_TO_MEAN:.0%} 이하이고, "
            f"5주 이동평균이 평균의 {OFF_SEASON_RATIO_TO_MEAN:.0%} 이하이며, "
            f"주간 증가량(절대값)이 평균의 {OFF_SEASON_SLOPE_FRAC:.0%} 이하로 완만한 구간 중 "
            f"평균 판매량이 가장 낮은 연속 구간 1개만 지정합니다."
        )
    elif stage == "변곡점(최고점)":
        return (
            f"전체 기간 중 최대 판매량 주차이며, 동률이면 전주 대비 증가율이 더 큰 주를 피크로 잡습니다. "
            f"현재 판매수량은 {qty:.0f}입니다."
        )
    elif stage == "쇠퇴":
        return (
            f"피크 이후 쇠퇴 구간입니다. "
            f"현재 판매수량 {qty:.0f}, 최고점 대비 {ratio_pct}, 전주 대비 증가율 {slope_pct}입니다."
        )
    else:
        return "현재 데이터만으로 단계 판단이 어렵습니다."


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
# 6. 시트 로딩 함수
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

    values = ws.get_all_values()
    if not values:
        return pd.DataFrame()

    raw_headers = values[0]
    headers = make_unique_headers([str(h) for h in raw_headers])

    rows = values[1:] if len(values) > 1 else []
    if not rows:
        return pd.DataFrame(columns=headers)

    max_cols = len(headers)
    normalized_rows = []
    for row in rows:
        row = list(row)
        if len(row) < max_cols:
            row = row + [""] * (max_cols - len(row))
        elif len(row) > max_cols:
            row = row[:max_cols]
        normalized_rows.append(row)

    return pd.DataFrame(normalized_rows, columns=headers)


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
# 7. Streamlit 실행부
# =========================
st.set_page_config(
    page_title="아이템별 PLC 포인트 분석",
    layout="wide",
)

st.title("아이템별 PLC 포인트 분석")
st.caption("구글시트 기반으로 아이템별 주차 판매 흐름을 분석하고 PLC 단계를 자동 분류합니다.")

with st.sidebar:
    st.subheader("PLC 분류 옵션")
    st.checkbox(
        "OpenAI로 PLC 분류 (secrets: gpt_gpi)",
        value=False,
        key="use_gpt_plc",
        help="켜면 classify_plc()가 규칙 대신 API 응답을 우선 사용합니다. 실패 시 규칙 기반으로 폴백합니다.",
    )
    st.text_input("OpenAI 모델명", value="gpt-4o-mini", key="openai_plc_model")
    if get_gpt_gpi():
        st.caption("gpt_gpi 키가 로드되었습니다.")
    else:
        st.caption("gpt_gpi 미설정 시 GPT 옵션은 폴백만 동작합니다.")

try:
    raw_df = load_sheet_data()
    st.success("구글시트 데이터를 불러왔습니다.")

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
# 8) 상단 요약
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
# 9) 상세 분석
# =========================
st.subheader("아이템 상세 분석")

item_list = sorted(plc_df["item"].dropna().unique().tolist())
selected_item = st.selectbox("아이템 선택", item_list)

item_df = plc_df[plc_df["item"] == selected_item].sort_values("sort_key").reset_index(drop=True)

if item_df.empty:
    st.warning("선택한 아이템 데이터가 없습니다.")
    st.stop()

latest_row = item_df.iloc[-1]
_peak_df = item_df[item_df["is_peak_week"] == True]
peak_row = _peak_df.iloc[0] if len(_peak_df) else item_df.loc[item_df["qty"].idxmax()]

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
        "recent_slope", "growth_accel", "discount_rate", "plc_stage"
    ]
    show_cols = [c for c in show_cols if c in detail_df.columns]
    st.dataframe(detail_df[show_cols], use_container_width=True, hide_index=True)


# =========================
# 10) 기준 설명
# =========================
st.subheader("PLC 분류 기준")

st.markdown(
    """
**공통**
- 기울기(변화율)는 **전주 대비 증가율** `(이번주−전주)/전주` 로 계산합니다.
- 초반은 무조건 **도입**, **도입·성장·성숙·쇠퇴**는 모든 아이템에 최소 1주씩 존재하도록 맞춥니다.
- **도입·성장·성숙·쇠퇴**는 데이터가 짧지 않을 때 각각 최소 1주는 존재하도록 후처리 보정합니다.
- **비시즌**은 필수가 아니며, 아래 조건을 만족할 때만 중간에 표시합니다.

**도입**
- 시작 주차는 항상 도입입니다.
- 도입 구간은 **최대 5주**입니다.
- 전주 대비 증가율이 **급상승**이면(상한 초과) 도입을 끝냅니다.

**성장**
- 피크 **이전** 타임라인 구간입니다.
- **연속 3주** 동안 전주 대비 증가율이 **계속 커지는**(가속) 패턴이 있으면 성장기 근거로 활용합니다(증가량이 아니라 **증가율**).

**성숙**
- **4주 이상**, 피크 **앞·뒤 최소 1주씩** 포함하는 구간 후보 중에서 고릅니다.
- 가능하면 구간 안의 **모든 주**가 판매량 **전체 평균보다 높은** 구간을 우선합니다.

**피크(변곡점·최고점)**
- **최대 판매량** 주차입니다. 동률이면 전주 대비 증가율이 **더 큰(더 급한)** 주차를 피크로 씁니다.

**쇠퇴**
- 피크 **이후** 구간입니다.

**비시즌** (선택)
- 판매량이 **정규분포에 가깝다**고 판단되면 비시즌을 두지 않습니다(Shapiro 검정, 없으면 왜도·첨도 휴리스틱).
- **연속 5주 이상** 후보: 5주 이동평균 ≤ 전체평균×60%, **각 주** 판매량 ≤ 전체평균×50%, **주간 증가량(절대값)** ≤ 전체평균×15%(완만).
- 피크 직전(피크·앞뒤 성숙 공간 확보) **중간**에서만 탐색하고, 후보 중 **구간 평균 판매량이 가장 낮은 연속 구간 1개**만 지정합니다.
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
