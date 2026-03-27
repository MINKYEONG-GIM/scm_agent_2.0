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





import os
import re
import json
from typing import List, Optional, Tuple, Dict

import numpy as np
import pandas as pd
import gspread
from google.oauth2.service_account import Credentials
from supabase import create_client, Client


# =========================================================
# 환경설정 읽기
# =========================================================
def load_local_secrets() -> dict:
    """
    streamlit 없이도 돌릴 수 있게 단순 dict 형태로 secrets를 읽는 보조 함수.
    실제로는 .streamlit/secrets.toml 대신 환경변수로만 써도 됨.
    """
    try:
        import tomllib  # py3.11+
        path = ".streamlit/secrets.toml"
        if os.path.exists(path):
            with open(path, "rb") as f:
                return tomllib.load(f)
    except Exception:
        pass
    return {}


SECRETS = load_local_secrets()


def get_secret(path: List[str], default=None):
    cur = SECRETS
    try:
        for key in path:
            cur = cur[key]
        return cur
    except Exception:
        return default


# =========================================================
# 공통 유틸
# =========================================================
def make_unique_headers(headers: List[str]) -> List[str]:
    seen = {}
    result = []
    for h in headers:
        col = str(h).strip()
        if not col:
            col = "unnamed"

        if col not in seen:
            seen[col] = 1
            result.append(col)
        else:
            seen[col] += 1
            result.append(f"{col}_{seen[col]}")
    return result


def clean_number(value):
    if pd.isna(value):
        return np.nan

    s = str(value).strip()
    if s == "":
        return np.nan

    s = s.replace(",", "")
    try:
        return float(s)
    except Exception:
        return np.nan


def parse_yearweek_to_date(yearweek: str) -> pd.Timestamp:
    s = str(yearweek).strip()
    if not re.match(r"^\d{4}-\d{1,2}$", s):
        return pd.NaT

    year_str, week_str = s.split("-")
    year = int(year_str)
    week = int(week_str)

    try:
        return pd.to_datetime(f"{year}-W{week:02d}-1", format="%G-W%V-%u", errors="coerce")
    except Exception:
        return pd.NaT


def extract_item_code_from_sku(sku: str) -> str:
    s = str(sku).strip()
    if len(s) >= 4:
        return s[2:4]
    return ""


def style_code_from_material(material: str) -> str:
    s = str(material).strip()
    return s[:10] if s else ""


# =========================================================
# Google Sheets 연결
# =========================================================
def get_gspread_client():
    scopes = [
        "https://www.googleapis.com/auth/spreadsheets",
        "https://www.googleapis.com/auth/drive.readonly",
    ]

    creds_dict = get_secret(["gcp_service_account"])
    if not creds_dict:
        service_account_json = os.getenv("GOOGLE_SERVICE_ACCOUNT_JSON")
        if service_account_json:
            creds_dict = json.loads(service_account_json)
        else:
            raise ValueError("구글 서비스 계정 정보가 없습니다.")

    credentials = Credentials.from_service_account_info(dict(creds_dict), scopes=scopes)
    return gspread.authorize(credentials)


def get_sheet_id() -> str:
    v = get_secret(["sheets", "sheet_id"]) or os.getenv("GOOGLE_SHEET_ID")
    if not v:
        raise ValueError("sheet_id가 없습니다.")
    return str(v)


def get_sheet_name(key: str, default_name: str) -> str:
    return str(get_secret(["sheets", key], default_name))


def load_sheet_as_df(worksheet_name: str) -> pd.DataFrame:
    client = get_gspread_client()
    sh = client.open_by_key(get_sheet_id())
    ws = sh.worksheet(worksheet_name)

    values = ws.get_all_values()
    if not values:
        return pd.DataFrame()

    headers = make_unique_headers(values[0])
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


# =========================================================
# final 정규화
# =========================================================
def prepare_final_df(final_df: pd.DataFrame) -> pd.DataFrame:
    """
    신버전(final DB) 기준:
    CALDAY, PLANT, MATERIAL, SALE 등
    -> 표준 컬럼으로 변환
    """
    df = final_df.copy()

    new_cols = {"CALDAY", "PLANT", "MATERIAL", "SALE"}
    is_new_schema = all(c in df.columns for c in new_cols)

    if is_new_schema:
        df["sku"] = df["MATERIAL"].astype(str).str.strip()
        df["sku_name"] = df["MATERIAL"].astype(str).str.strip()
        df["plant_name"] = df["PLANT"].astype(str).str.strip().replace("", "전체")

        sale_raw = df["SALE"].apply(clean_number).fillna(0)

        if "SSTOC_TMP_QTY" in df.columns:
            sstoc = df["SSTOC_TMP_QTY"].apply(clean_number)
        else:
            sstoc = pd.Series(np.nan, index=df.index, dtype=float)

        mask_sstoc_neg = sstoc.notna() & (sstoc < 0)

        df["판매량"] = sale_raw.astype(float)
        df.loc[mask_sstoc_neg, "판매량"] = sale_raw.loc[mask_sstoc_neg].abs()

        calday = df["CALDAY"].astype(str).str.strip().str.replace(r"\.0$", "", regex=True)
        df["날짜"] = pd.to_datetime(calday, format="%Y%m%d", errors="coerce")

        if "HSTOC_QTY" in df.columns:
            df["기초재고"] = df["HSTOC_QTY"].apply(clean_number)

        ipgo = (
            df["IPGO_QTY"].apply(clean_number).fillna(0)
            if "IPGO_QTY" in df.columns
            else pd.Series(0.0, index=df.index, dtype=float)
        )
        sstoc_pos = sstoc.fillna(0).clip(lower=0)
        df["분배량"] = ipgo.astype(float) + sstoc_pos.astype(float)

        ordqty = (
            df["ORDQTY"].apply(clean_number).fillna(0)
            if "ORDQTY" in df.columns
            else pd.Series(0.0, index=df.index, dtype=float)
        )
        df["출고량(회전 등)"] = ordqty.astype(float)
        df.loc[mask_sstoc_neg, "출고량(회전 등)"] = (
            sstoc.loc[mask_sstoc_neg].abs() - sale_raw.loc[mask_sstoc_neg].abs()
        ).astype(float)

        df["item_code"] = df["sku"].apply(extract_item_code_from_sku)
        df.loc[df["item_code"].astype(str).str.strip() == "", "item_code"] = df["sku"]
        df["style_code"] = df["sku"].map(style_code_from_material)
        return df

    # 구버전 fallback
    required_cols = ["sku", "sku_name", "날짜", "판매량"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"final 시트 필수 컬럼이 없습니다: {missing}")

    if "plant_name" not in df.columns:
        df["plant_name"] = "전체"

    df["sku"] = df["sku"].astype(str).str.strip()
    df["sku_name"] = df["sku_name"].astype(str).str.strip()
    df["plant_name"] = df["plant_name"].astype(str).str.strip().replace("", "전체")
    df["item_code"] = df["sku"].apply(extract_item_code_from_sku)
    df["판매량"] = df["판매량"].apply(clean_number).fillna(0)

    raw_date = (
        df["날짜"]
        .astype(str)
        .str.strip()
        .str.replace(".", "-", regex=False)
        .str.replace("/", "-", regex=False)
        .str.replace(" ", "", regex=False)
    )
    current_year = pd.Timestamp.today().year
    raw_date = raw_date.str.replace(
        r"^(\d{1,2})월(\d{1,2})일$",
        rf"{current_year}-\1-\2",
        regex=True
    )
    df["날짜"] = pd.to_datetime(raw_date, errors="coerce")
    df["style_code"] = df["sku"].map(style_code_from_material)
    return df


# =========================================================
# plc db -> weekly/monthly 시계열
# =========================================================
def prepare_plc_item_timeseries(plc_df: pd.DataFrame, item_code: str) -> Tuple[str, pd.DataFrame, pd.DataFrame]:
    df = plc_df.copy()

    required_cols = ["아이템명", "아이템코드"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"plc db 필수 컬럼이 없습니다: {missing}")

    df["아이템코드"] = df["아이템코드"].astype(str).str.strip()
    matched = df[df["아이템코드"] == str(item_code).strip()].copy()

    if matched.empty:
        raise ValueError(f"plc db에서 아이템코드 '{item_code}'를 찾지 못했습니다.")

    row = matched.iloc[0]
    item_name = str(row["아이템명"]).strip()

    week_cols = [c for c in df.columns if re.match(r"^\d{4}-\d{1,2}$", str(c).strip())]
    if not week_cols:
        raise ValueError("plc db에 2025-01 형식의 주차 컬럼이 없습니다.")

    records = []
    for col in week_cols:
        sales = clean_number(row[col])
        week_start = parse_yearweek_to_date(col)
        if pd.isna(week_start):
            continue

        records.append({
            "year_week": str(col).strip(),
            "week_start": week_start,
            "sales": 0 if pd.isna(sales) else float(sales),
        })

    weekly_df = pd.DataFrame(records).sort_values("week_start").reset_index(drop=True)
    if weekly_df.empty:
        raise ValueError(f"아이템코드 '{item_code}'의 주차 데이터가 없습니다.")

    weekly_df["month"] = weekly_df["week_start"].dt.to_period("M").dt.to_timestamp()
    monthly_df = (
        weekly_df.groupby("month", as_index=False)["sales"]
        .sum()
        .sort_values("month")
        .reset_index(drop=True)
    )

    return item_name, weekly_df, monthly_df


# =========================================================
# 형태 분류
# =========================================================
def smooth_series(values: np.ndarray, window: int = 2) -> np.ndarray:
    if len(values) < window:
        return values.copy()
    return pd.Series(values).rolling(window=window, center=True, min_periods=1).mean().values


def find_significant_peaks(
    values: np.ndarray,
    min_peak_ratio: float = 0.35,
    min_prominence_ratio: float = 0.10,
    min_distance: int = 1
) -> List[int]:
    if len(values) < 3:
        return []

    max_val = np.max(values)
    if max_val <= 0:
        return []

    candidate_peaks = []
    for i in range(1, len(values) - 1):
        if values[i] > values[i - 1] and values[i] >= values[i + 1]:
            left_base = values[i - 1]
            right_base = values[i + 1]
            base_level = max(left_base, right_base)

            peak_ratio = values[i] / max_val
            prominence = values[i] - base_level
            prominence_ratio = prominence / max_val

            if peak_ratio >= min_peak_ratio and prominence_ratio >= min_prominence_ratio:
                candidate_peaks.append(i)

    if not candidate_peaks:
        return []

    filtered = []
    for idx in candidate_peaks:
        if not filtered:
            filtered.append(idx)
        else:
            prev_idx = filtered[-1]
            if idx - prev_idx <= min_distance:
                if values[idx] > values[prev_idx]:
                    filtered[-1] = idx
            else:
                filtered.append(idx)
    return filtered


def is_double_peak(values: np.ndarray) -> Tuple[bool, List[int]]:
    peaks = find_significant_peaks(values, min_peak_ratio=0.25, min_prominence_ratio=0.05, min_distance=2)
    if len(peaks) < 2:
        return False, peaks

    mx = np.max(values)
    strong = [p for p in peaks if values[p] >= mx * 0.6]
    if len(strong) < 2:
        return False, peaks

    strong = sorted(strong)
    for i in range(len(strong) - 1):
        p1 = strong[i]
        p2 = strong[i + 1]
        if p2 - p1 < 2:
            continue

        valley = np.min(values[p1:p2 + 1])
        lower_peak = min(values[p1], values[p2])
        if lower_peak > 0 and valley / lower_peak <= 0.85:
            return True, [p1, p2]

    return False, peaks


def is_single_peak(values: np.ndarray) -> Tuple[bool, List[int]]:
    peaks = find_significant_peaks(values, min_peak_ratio=0.30, min_prominence_ratio=0.08, min_distance=2)

    if len(peaks) == 1:
        return True, peaks
    if len(peaks) == 0:
        return False, peaks

    mx = np.max(values)
    strong_peaks = [p for p in peaks if values[p] >= mx * 0.60]
    if len(strong_peaks) == 1:
        return True, strong_peaks

    return False, peaks


def is_all_season(values: np.ndarray) -> bool:
    if len(values) < 4:
        return False

    avg = np.mean(values)
    mx = np.max(values)

    if avg <= 0:
        return False
    if mx / avg > 2.0:
        return False

    low = values < avg * 0.5
    if np.sum(low) > len(values) * 0.3:
        return False

    near_avg = (values >= avg * 0.7) & (values <= avg * 1.3)
    if np.sum(near_avg) < len(values) * 0.7:
        return False

    return True


def classify_shape(monthly_df: pd.DataFrame) -> Tuple[str, str]:
    if monthly_df.empty:
        return "판단불가", "월별 데이터 없음"

    y = monthly_df["sales"].values.astype(float)
    if len(y) < 3:
        return "판단불가", "월별 데이터 부족"

    y_smooth = smooth_series(y, window=2)

    is_double, peaks = is_double_peak(y_smooth)
    if is_double:
        return "쌍봉형", f"유의미한 피크 2개 발견: {peaks}"

    is_single, peaks = is_single_peak(y_smooth)
    if is_single:
        return "단봉형", f"유의미한 피크 1개 발견: {peaks}"

    if is_all_season(y_smooth):
        return "올시즌형", "월별 매출이 비교적 고르게 분포"

    return "단봉형", "명확하지 않아 단봉형으로 처리"


# =========================================================
# 주차 단계 분류
# =========================================================
def classify_weekly_stages_by_shape(weekly_df: pd.DataFrame, shape_label: str) -> pd.DataFrame:
    df = weekly_df.copy().reset_index(drop=True)
    y = df["sales"].astype(float).fillna(0).values
    n = len(df)

    if n == 0:
        df["stage"] = []
        return df

    df["stage"] = "성숙"
    smooth = pd.Series(y).rolling(window=3, center=True, min_periods=1).mean().values

    def safe_argmax(arr):
        if len(arr) == 0:
            return 0
        return int(np.argmax(arr))

    if shape_label == "단봉형":
        peak_idx = int(np.argmax(y))

        intro_end = min(3, max(1, peak_idx // 3))
        growth_start = intro_end + 1
        growth_end = max(growth_start, peak_idx - 1)
        peak_start = peak_idx
        peak_end = peak_idx
        maturity_start = min(n - 1, peak_end + 1)
        maturity_end = min(n - 1, maturity_start + 2)
        decline_start = min(n - 1, maturity_end + 1)

        df.loc[:intro_end, "stage"] = "도입"
        if growth_start <= growth_end:
            df.loc[growth_start:growth_end, "stage"] = "성장"
        df.loc[peak_start:peak_end, "stage"] = "피크"
        if maturity_start <= maturity_end:
            df.loc[maturity_start:maturity_end, "stage"] = "성숙"
        if decline_start < n:
            df.loc[decline_start:, "stage"] = "쇠퇴"
        return df

    if shape_label == "쌍봉형":
        peaks = find_significant_peaks(smooth, min_peak_ratio=0.25, min_prominence_ratio=0.05, min_distance=2)

        if len(peaks) >= 2:
            peaks = sorted(peaks, key=lambda i: smooth[i], reverse=True)[:2]
            peaks = sorted(peaks)
            peak1, peak2 = peaks[0], peaks[1]
        else:
            peak1 = safe_argmax(smooth[: max(1, n // 2)])
            peak2 = safe_argmax(smooth[max(peak1 + 1, 1):]) + max(peak1 + 1, 1)
            if peak2 >= n:
                peak2 = n - 1

        if peak2 > peak1 + 1:
            valley_rel = np.argmin(smooth[peak1:peak2 + 1])
            valley_idx = peak1 + valley_rel
        else:
            valley_idx = min(n - 1, peak1 + 1)

        intro_end = min(3, max(1, peak1 // 3))
        growth_start = intro_end + 1
        growth_end = max(growth_start, peak1 - 1)

        maturity1_start = min(n - 1, peak1 + 1)
        maturity1_end = min(n - 1, max(maturity1_start, valley_idx - 2))

        offseason_start = min(n - 1, max(maturity1_end + 1, valley_idx - 1))
        offseason_end = min(n - 1, valley_idx + 1)

        maturity2_start = min(n - 1, offseason_end + 1)
        maturity2_end = min(n - 1, max(maturity2_start, peak2 - 1))

        maturity3_start = min(n - 1, peak2 + 1)
        maturity3_end = min(n - 1, maturity3_start + 1)

        decline_start = min(n - 1, maturity3_end + 1)

        df.loc[:intro_end, "stage"] = "도입"
        if growth_start <= growth_end:
            df.loc[growth_start:growth_end, "stage"] = "성장"
        df.loc[peak1:peak1, "stage"] = "피크"
        if maturity1_start <= maturity1_end:
            df.loc[maturity1_start:maturity1_end, "stage"] = "성숙"
        if offseason_start <= offseason_end:
            df.loc[offseason_start:offseason_end, "stage"] = "비시즌"
        if maturity2_start <= maturity2_end:
            df.loc[maturity2_start:maturity2_end, "stage"] = "성숙"
        df.loc[peak2:peak2, "stage"] = "피크2"
        if maturity3_start <= maturity3_end:
            df.loc[maturity3_start:maturity3_end, "stage"] = "성숙"
        if decline_start < n:
            df.loc[decline_start:, "stage"] = "쇠퇴"
        return df

    intro_end = min(2, n - 1)
    growth_end = min(max(intro_end + 2, n // 4), n - 1)
    decline_start = max(growth_end + 1, n - max(3, n // 5))

    df.loc[:intro_end, "stage"] = "도입"
    if intro_end + 1 <= growth_end:
        df.loc[intro_end + 1:growth_end, "stage"] = "성장"
    if growth_end + 1 <= decline_start - 1:
        df.loc[growth_end + 1:decline_start - 1, "stage"] = "성숙"
    if decline_start < n:
        df.loc[decline_start:, "stage"] = "쇠퇴"

    return df


# =========================================================
# 예측
# =========================================================
def forecast_with_ratio(
    weekly_df: pd.DataFrame,
    final_item_df: pd.DataFrame
) -> pd.DataFrame:
    """
    기존 코드의 핵심 방식:
    - plc db의 작년 주차 비중
    - final의 올해 누적 판매량
    - 남은 주차를 비중대로 배분
    """
    df_last = weekly_df.copy()
    df_last["week_no"] = df_last["week_start"].dt.isocalendar().week.astype(int)
    df_last["sales"] = pd.to_numeric(df_last["sales"], errors="coerce").fillna(0.0)

    last_total = float(df_last["sales"].sum())
    if last_total <= 0:
        return pd.DataFrame(columns=["날짜", "forecast"])

    df_last["ratio"] = df_last["sales"] / last_total
    ratio_by_week = df_last.groupby("week_no")["ratio"].sum().to_dict()
    last_sales_by_week = df_last.groupby("week_no")["sales"].sum().to_dict()

    df_this = final_item_df.dropna(subset=["날짜"]).copy()
    if df_this.empty:
        this_sales_by_week = {}
    else:
        df_this["iso_year"] = df_this["날짜"].dt.isocalendar().year.astype(int)
        df_this["week_no"] = df_this["날짜"].dt.isocalendar().week.astype(int)
        df_this["판매량"] = pd.to_numeric(df_this["판매량"], errors="coerce").fillna(0.0)
        this_year = int(pd.Timestamp.today().year)
        df_this = df_this[df_this["iso_year"] == this_year].copy()
        this_sales_by_week = df_this.groupby("week_no")["판매량"].sum().to_dict()

    this_year = int(pd.Timestamp.today().year)
    current_week_no = int(pd.Timestamp.today().isocalendar().week)

    this_to_date = float(sum(v for w, v in this_sales_by_week.items() if int(w) <= current_week_no))
    last_to_date = float(sum(v for w, v in last_sales_by_week.items() if int(w) <= current_week_no))

    if last_to_date <= 0:
        return pd.DataFrame(columns=["날짜", "forecast"])

    scale = this_to_date / last_to_date if last_to_date > 0 else 0
    expected_total = last_total * scale
    remaining_total = max(0.0, expected_total - this_to_date)

    remaining_weeks = sorted([int(w) for w in ratio_by_week.keys() if int(w) > current_week_no])
    if not remaining_weeks:
        return pd.DataFrame(columns=["날짜", "forecast"])

    remaining_ratio_sum = float(sum(ratio_by_week.get(w, 0.0) for w in remaining_weeks))
    if remaining_ratio_sum <= 0:
        return pd.DataFrame(columns=["날짜", "forecast"])

    forecast_rows = []
    for w in remaining_weeks:
        ratio = float(ratio_by_week.get(w, 0.0)) / remaining_ratio_sum
        qty = int(round(remaining_total * ratio))
        date = pd.to_datetime(f"{this_year}-W{w:02d}-1", format="%G-W%V-%u", errors="coerce")
        if pd.notna(date):
            forecast_rows.append({
                "날짜": date,
                "forecast": max(0, qty),
            })

    return pd.DataFrame(forecast_rows)


# =========================================================
# season / peak 계산
# =========================================================
def get_peak_week(weekly_df: pd.DataFrame) -> Optional[int]:
    if weekly_df.empty:
        return None
    tmp = weekly_df.copy()
    tmp["week_no"] = tmp["week_start"].dt.isocalendar().week.astype(int)
    top = tmp.sort_values(["sales", "week_no"], ascending=[False, True]).iloc[0]
    return int(top["week_no"])


def get_peak_month(monthly_df: pd.DataFrame) -> Optional[int]:
    if monthly_df.empty:
        return None
    tmp = monthly_df.copy()
    tmp["month_no"] = pd.to_datetime(tmp["month"]).dt.month.astype(int)
    top = tmp.sort_values(["sales", "month_no"], ascending=[False, True]).iloc[0]
    return int(top["month_no"])


def get_season_start_end_week(weekly_df: pd.DataFrame) -> Tuple[Optional[int], Optional[int]]:
    if weekly_df.empty:
        return None, None

    tmp = weekly_df.copy()
    tmp["week_no"] = tmp["week_start"].dt.isocalendar().week.astype(int)
    non_zero = tmp[tmp["sales"].fillna(0) > 0].copy()

    if non_zero.empty:
        return None, None

    return int(non_zero["week_no"].min()), int(non_zero["week_no"].max())


# =========================================================
# Supabase
# =========================================================
def get_supabase_client() -> Client:
    url = os.getenv("SUPABASE_URL") or get_secret(["SUPABASE_URL"])
    key = os.getenv("SUPABASE_KEY") or get_secret(["SUPABASE_KEY"])
    if not url or not key:
        raise ValueError("SUPABASE_URL / SUPABASE_KEY가 없습니다.")
    return create_client(url, key)


def insert_forecast_run(
    supabase: Client,
    sku: str,
    style_code: str,
    plant: str,
    shape_type: str,
    shape_reason: str,
    peak_week: Optional[int],
    peak_month: Optional[int],
    season_start_week: Optional[int],
    season_end_week: Optional[int],
):
    payload = {
        "SKU": sku,  # 실제 컬럼명이 대문자 SKU일 경우
        "style_code": style_code,
        "plant": plant,
        "shape_type": shape_type,
        "shape_reason": shape_reason,
        "peak_week": peak_week,
        "peak_month": peak_month,
        "season_start_week": season_start_week,
        "season_end_week": season_end_week,
    }

    result = supabase.table("sku_forecast_run").insert(payload).execute()
    inserted = result.data[0]
    return inserted


def insert_weekly_forecasts(supabase: Client, rows: List[dict]):
    if rows:
        supabase.table("sku_weekly_forecast").insert(rows).execute()


def insert_monthly_forecasts(supabase: Client, rows: List[dict]):
    if rows:
        supabase.table("sku_monthly_forecast").insert(rows).execute()


# =========================================================
# 데이터 생성
# =========================================================
def build_final_item_options(final_prepared: pd.DataFrame) -> pd.DataFrame:
    options = (
        final_prepared[["sku", "sku_name", "item_code", "style_code", "plant_name"]]
        .dropna(subset=["sku"])
        .drop_duplicates(subset=["sku"])
        .reset_index(drop=True)
    )
    return options


def build_weekly_insert_rows(
    forecast_run_id: int,
    sku: str,
    sty: str,
    weekly_forecast_df: pd.DataFrame,
    staged_weekly_df: pd.DataFrame
) -> List[dict]:
    if weekly_forecast_df.empty:
        return []

    stage_map = {}
    peak_week = None

    if not staged_weekly_df.empty:
        tmp = staged_weekly_df.copy()
        tmp["week_no"] = tmp["week_start"].dt.isocalendar().week.astype(int)
        stage_map = tmp.groupby("week_no")["stage"].last().to_dict()

        peak_row = tmp.sort_values(["sales", "week_no"], ascending=[False, True]).iloc[0]
        peak_week = int(peak_row["week_no"])

    rows = []
    for _, row in weekly_forecast_df.iterrows():
        week_no = int(pd.Timestamp(row["날짜"]).isocalendar().week)
        year = int(pd.Timestamp(row["날짜"]).isocalendar().year)
        year_week = f"{year}-{week_no:02d}"

        rows.append({
            "forecast_run_id": forecast_run_id,
            "year_week": year_week,
            "forecast_qty": int(row["forecast"]),
            "stage": stage_map.get(week_no, None),
            "sty": sty,
            "sku": sku,
            "is_peak_week": bool(peak_week == week_no),
        })
    return rows


def build_monthly_insert_rows(
    forecast_run_id: int,
    sku: str,
    sty: str,
    weekly_forecast_df: pd.DataFrame
) -> List[dict]:
    if weekly_forecast_df.empty:
        return []

    tmp = weekly_forecast_df.copy()
    tmp["month"] = pd.to_datetime(tmp["날짜"]).dt.to_period("M").dt.to_timestamp()
    tmp["year_month"] = tmp["month"].dt.strftime("%Y-%m")

    monthly = (
        tmp.groupby(["month", "year_month"], as_index=False)["forecast"]
        .sum()
        .sort_values("month")
        .reset_index(drop=True)
    )

    peak_month = None
    if not monthly.empty:
        peak_row = monthly.sort_values(["forecast", "year_month"], ascending=[False, True]).iloc[0]
        peak_month = str(peak_row["year_month"])

    rows = []
    for _, row in monthly.iterrows():
        rows.append({
            "forecast_run_id": forecast_run_id,
            "sku": sku,
            "sty": sty,
            "year_month": row["year_month"],
            "forecast_qty": int(row["forecast"]),
            "stage": None,
            "is_peak_month": bool(str(row["year_month"]) == peak_month),
        })
    return rows


# =========================================================
# 메인 ETL
# =========================================================
def run_etl(limit_sku: Optional[int] = None):
    supabase = get_supabase_client()

    plc_sheet = get_sheet_name("plc_db", "plc db")
    final_sheet = get_sheet_name("final", "final")

    print(f"[LOAD] plc sheet: {plc_sheet}")
    plc_df = load_sheet_as_df(plc_sheet)

    print(f"[LOAD] final sheet: {final_sheet}")
    final_df = load_sheet_as_df(final_sheet)

    if plc_df.empty:
        raise ValueError("plc db 시트가 비어 있습니다.")
    if final_df.empty:
        raise ValueError("final 시트가 비어 있습니다.")

    final_prepared = prepare_final_df(final_df)
    options_df = build_final_item_options(final_prepared)

    if limit_sku is not None:
        options_df = options_df.head(limit_sku).copy()

    success_count = 0
    fail_count = 0

    for idx, opt in options_df.iterrows():
        sku = str(opt["sku"]).strip()
        item_code = str(opt["item_code"]).strip()
        style_code = str(opt["style_code"]).strip()
        plant = str(opt.get("plant_name", "")).strip() or "전체"

        print(f"[{idx + 1}/{len(options_df)}] 처리중: sku={sku}, item_code={item_code}")

        try:
            item_name, weekly_df, monthly_df = prepare_plc_item_timeseries(plc_df, item_code)
            shape_type, shape_reason = classify_shape(monthly_df)
            staged_weekly_df = classify_weekly_stages_by_shape(weekly_df, shape_type)

            final_item_df = final_prepared[final_prepared["sku"].astype(str).str.strip() == sku].copy()
            weekly_forecast_df = forecast_with_ratio(staged_weekly_df, final_item_df)

            peak_week = get_peak_week(staged_weekly_df)
            peak_month = get_peak_month(monthly_df)
            season_start_week, season_end_week = get_season_start_end_week(staged_weekly_df)

            inserted_run = insert_forecast_run(
                supabase=supabase,
                sku=sku,
                style_code=style_code,
                plant=plant,
                shape_type=shape_type,
                shape_reason=shape_reason,
                peak_week=peak_week,
                peak_month=peak_month,
                season_start_week=season_start_week,
                season_end_week=season_end_week,
            )

            forecast_run_id = inserted_run["id"]

            weekly_rows = build_weekly_insert_rows(
                forecast_run_id=forecast_run_id,
                sku=sku,
                sty=style_code,
                weekly_forecast_df=weekly_forecast_df,
                staged_weekly_df=staged_weekly_df,
            )

            monthly_rows = build_monthly_insert_rows(
                forecast_run_id=forecast_run_id,
                sku=sku,
                sty=style_code,
                weekly_forecast_df=weekly_forecast_df,
            )

            insert_weekly_forecasts(supabase, weekly_rows)
            insert_monthly_forecasts(supabase, monthly_rows)

            success_count += 1
            print(f"  -> 완료: forecast_run_id={forecast_run_id}, weekly={len(weekly_rows)}, monthly={len(monthly_rows)}")

        except Exception as e:
            fail_count += 1
            print(f"  -> 실패: sku={sku}, error={e}")

    print("===================================")
    print(f"성공: {success_count}")
    print(f"실패: {fail_count}")
    print("완료")


if __name__ == "__main__":
    # 처음엔 3~5개만 넣어서 테스트 권장
    run_etl(limit_sku=5)
