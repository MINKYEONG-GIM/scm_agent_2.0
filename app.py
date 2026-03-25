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
# 공통 유틸
# =========================
def make_unique_headers(headers: List[str]) -> List[str]:
    """
    중복 컬럼명이 있을 때 고유한 이름으로 바꿉니다.
    예: ['A', 'A', 'B'] -> ['A', 'A_2', 'B']
    """
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
    """
    문자열 숫자를 안전하게 float로 변환합니다.
    예:
    '12,345' -> 12345.0
    '' -> np.nan
    """
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
    """
    '2025-01' 같은 값을 해당 ISO 주차의 월요일 날짜로 변환합니다.
    """
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

def is_yearweek_col(col: object) -> bool:
    return bool(re.match(r"^\d{4}-\d{1,2}$", str(col).strip()))

def preprocess_plc_db(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    PLC DB 시트 구조를 자동 감지해 long 포맷으로 표준화합니다.

    지원 구조
    - (A) 세로형: '연도/주' + 아이템 컬럼들
    - (B) 가로형: ['아이템명','아이템코드', '2025-01','2025-02', ...]

    반환 long 컬럼
    - year_week, item_name, item_code, sales, week_start, month, week
    """
    if df_raw.empty:
        return pd.DataFrame(columns=["year_week", "item_name", "item_code", "sales", "week_start", "month", "week"])

    df = df_raw.copy()
    df.columns = [str(c).strip() for c in df.columns]

    # (A) 세로형: '연도/주' 존재
    if "연도/주" in df.columns:
        year_week_col = "연도/주"
        item_cols = [c for c in df.columns if c not in {year_week_col, "", " "}]
        if not item_cols:
            raise ValueError("PLC DB: '연도/주' 외 아이템 컬럼이 없습니다.")

        long_df = df.melt(id_vars=[year_week_col], var_name="item_name", value_name="sales").rename(
            columns={year_week_col: "year_week"}
        )
        long_df["item_name"] = long_df["item_name"].astype(str).str.strip()
        long_df["item_code"] = long_df["item_name"].apply(normalize_item_code)

    else:
        # (B) 가로형: 주차 컬럼들이 존재하고 아이템코드/아이템명이 존재
        week_cols = [c for c in df.columns if is_yearweek_col(c)]
        if not week_cols:
            raise ValueError("PLC DB: '연도/주' 컬럼도 없고 주차(예: 2025-01) 컬럼도 찾지 못했습니다.")

        # 헤더 후보(일반적으로 '아이템명','아이템코드')
        name_col = "아이템명" if "아이템명" in df.columns else df.columns[0]
        code_col = "아이템코드" if "아이템코드" in df.columns else ("item_code" if "item_code" in df.columns else None)
        if code_col is None:
            raise ValueError("PLC DB: 아이템코드 컬럼을 찾지 못했습니다. (예: '아이템코드')")

        long_df = df.melt(id_vars=[name_col, code_col], value_vars=week_cols, var_name="year_week", value_name="sales")
        long_df = long_df.rename(columns={name_col: "item_name", code_col: "item_code"})
        long_df["item_name"] = long_df["item_name"].astype(str).str.strip()
        long_df["item_code"] = long_df["item_code"].apply(normalize_item_code)

    # 공통 정리
    long_df["year_week"] = long_df["year_week"].astype(str).str.strip()
    long_df = long_df[long_df["year_week"].str.match(r"^\d{4}-\d{1,2}$", na=False)].copy()

    long_df["sales"] = long_df["sales"].apply(clean_number).fillna(0)
    long_df["week_start"] = long_df["year_week"].apply(parse_yearweek_to_date)
    long_df = long_df.dropna(subset=["week_start"]).copy()

    extracted = long_df["year_week"].str.extract(r"(?P<year>\d{4})-(?P<week>\d{1,2})")
    long_df["week"] = pd.to_numeric(extracted["week"], errors="coerce").fillna(0).astype(int)
    long_df["month"] = long_df["week_start"].dt.to_period("M").dt.to_timestamp()

    # item_code 없으면 item_name에서라도 뽑기
    long_df["item_code"] = long_df["item_code"].fillna(long_df["item_name"].apply(normalize_item_code))

    return long_df.sort_values(["item_code", "week_start"]).reset_index(drop=True)

def build_timeseries_from_plc_long(
    plc_long: pd.DataFrame,
    item_code: Optional[str] = None,
    item_name: Optional[str] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    plc_long에서 특정 아이템(코드 우선, 없으면 이름)만 필터링하여
    주차/월 집계를 반환합니다.
    """
    if plc_long.empty:
        return pd.DataFrame(), pd.DataFrame()

    df = plc_long.copy()
    if item_code:
        code = normalize_item_code(item_code)
        df = df[df["item_code"] == code].copy()
    elif item_name:
        df = df[df["item_name"] == str(item_name).strip()].copy()

    if df.empty:
        return pd.DataFrame(), pd.DataFrame()

    weekly_df = (
        df.groupby(["year_week", "week_start", "month", "week", "item_code", "item_name"], as_index=False)["sales"]
        .sum()
        .sort_values("week_start")
        .reset_index(drop=True)
    )

    monthly_df = (
        weekly_df.groupby("month", as_index=False)["sales"]
        .sum()
        .sort_values("month")
        .reset_index(drop=True)
    )

    return weekly_df, monthly_df

def extract_item_code_from_code(code: object) -> Optional[str]:
    """
    스타일코드/SKU 코드에서 3~4번째 2자리(영문)를 아이템코드로 추출합니다.
    예: 'ABCD...' -> 'CD' (0-based index 2:4)
    """
    if code is None or (isinstance(code, float) and np.isnan(code)):
        return None

    s = str(code).strip().replace(" ", "")
    if len(s) < 4:
        return None

    two = s[2:4].upper()
    if re.fullmatch(r"[A-Z]{2}", two):
        return two
    return None

def normalize_item_code(value: object) -> Optional[str]:
    """
    PLC 컬럼명/아이템코드에서 영문 2자리 코드만 정규화해 반환합니다.
    - 'ab' -> 'AB'
    - 'AB_가디건' -> 'AB' (처음 등장하는 2자리 영문)
    """
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    s = str(value).strip().upper()
    m = re.search(r"[A-Z]{2}", s)
    return m.group(0) if m else None

def call_openai_chat_json(messages: List[dict]) -> dict:
    """
    Chat Completions API를 호출해서 JSON 응답을 받습니다.
    """
    api_key = get_gpt_gpi()
    if not api_key:
        raise ValueError("OpenAI API Key가 없습니다. st.secrets 또는 환경변수에 gpt_gpi / OPENAI_API_KEY를 설정하세요.")

    payload = {
        "model": "gpt-4.1-mini",
        "messages": messages,
        "temperature": 0,
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "shape_result",
                "schema": {
                    "type": "object",
                    "properties": {
                        "shape_label": {
                            "type": "string",
                            "enum": ["단봉형", "쌍봉형", "올시즌형"]
                        },
                        "reason": {
                            "type": "string"
                        }
                    },
                    "required": ["shape_label", "reason"],
                    "additionalProperties": False
                }
            }
        }
    }

    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")

    req = urllib.request.Request(
        OPENAI_CHAT_URL,
        data=data,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            result = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="ignore")
        raise ValueError(f"OpenAI API 호출 실패: {e.code} / {body}")
    except Exception as e:
        raise ValueError(f"OpenAI API 호출 중 오류: {str(e)}")

    content = result["choices"][0]["message"]["content"]
    return json.loads(content)

def classify_shape(item_name: str, monthly_df: pd.DataFrame) -> Tuple[str, str]:
    if monthly_df.empty:
        return "판단불가", "월별 데이터가 없습니다."

    y = monthly_df["sales"].values.astype(float)

    if len(y) < 3:
        return "판단불가", "월별 데이터가 3개 미만입니다."

    y_smooth = smooth_series(y, window=2)
    month_labels = monthly_df["month"].dt.strftime("%Y-%m").tolist()

    prompt = f"""

    
아이템의 월별 매출 형태를 아래 3개 중 하나로만 판단하라.

- 반드시 월별 매출 기준으로만 판단할 것
- 주차별 매출은 참고하지 말 것

분류 순서
1. 쌍봉형
2. 단봉형
3. 올시즌형

판단 기준
- 쌍봉형: 의미 있는 피크가 2개 이상이고, 두 피크 사이에 저점이 존재함
- 단봉형: 의미 있는 큰 피크가 1개임
- 올시즌형: 큰 중심 피크 없이 전체 기간에 비교적 고르게 분포함

주의
- 반드시 월별 매출 기준으로만 판단할 것
- 주차별 매출은 참고하지 말 것
- 작은 잡음은 피크로 보지 말 것
- 반드시 3개 중 하나만 선택할 것
- reason은 짧고 명확한 한글로 작성할 것

아이템명: {item_name}
월 라벨: {month_labels}
월별 매출: {[float(v) for v in y]}
스무딩 값: {[round(float(v), 2) for v in y_smooth]}
""".strip()

    messages = [
        {
            "role": "developer",
            "content": "너는 월별 매출 형태를 쌍봉형, 단봉형, 올시즌형 중 하나로만 분류하는 분석가다. 반드시 JSON만 반환한다. 반드시 월별 매출 기준으로만 판단한다."
        },
        {
            "role": "user",
            "content": prompt
        }
    ]

    # 1차 판단: GPT
    try:
        result = call_openai_chat_json(messages)
        return result["shape_label"], f"GPT 1차 판별: {result['reason']}"
    except Exception as e:
        pass

    # 2차 fallback: 로직
    is_double, double_peaks = is_double_peak(y_smooth)
    if is_double:
        return "쌍봉형", f"GPT 실패, 로직 fallback: 의미 있는 피크 2개 ({[month_labels[i] for i in double_peaks]})"

    is_single, single_peaks = is_single_peak(y_smooth)
    if is_single:
        return "단봉형", f"GPT 실패, 로직 fallback: 의미 있는 피크 1개 ({[month_labels[i] for i in single_peaks]})"

    if is_all_season(y_smooth):
        return "올시즌형", "GPT 실패, 로직 fallback: 큰 피크 없이 전체적으로 고르게 분포"

    return "단봉형", "GPT 실패 및 로직 미확정으로 단봉형 fallback"

# =========================
# 구글 시트 로딩 함수
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

    raise ValueError("구글 서비스 계정 정보가 없습니다.")


def get_sheets_config() -> dict:
    """
    secrets.toml의 [sheets] 섹션을 dict로 반환합니다.
    필수 키: sheet_id
    선택 키: worksheet
    """
    if "sheets" not in st.secrets:
        raise ValueError("st.secrets['sheets'] 설정이 없습니다. secrets.toml에 [sheets] 섹션을 추가하세요.")
    return dict(st.secrets["sheets"])

def resolve_worksheet_name(sheets_cfg: dict, keys: List[str], default: Optional[str] = None) -> Optional[str]:
    for k in keys:
        v = sheets_cfg.get(k)
        if v is not None and str(v).strip():
            return str(v).strip()
    return default


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

@st.cache_data(ttl=300)
def load_final_sheet_data() -> pd.DataFrame:
    sheets_cfg = get_sheets_config()
    ws = resolve_worksheet_name(
        sheets_cfg,
        keys=["final", "FINAL", "final_sheet", "final_worksheet", "FINAL_WORKSHEET_NAME"],
        default=None,
    )
    if not ws:
        return pd.DataFrame()
    return load_sheet_as_df(ws)


# =========================
# 데이터 전처리
# =========================
def get_item_columns(df: pd.DataFrame) -> List[str]:
    """
    아이템 선택용 컬럼 목록 반환
    '연도/주'를 제외한 컬럼 중 값이 있는 컬럼만 반환
    """
    exclude_cols = {"연도/주", "", " "}
    candidate_cols = [c for c in df.columns if str(c).strip() not in exclude_cols]

    item_cols = []
    for col in candidate_cols:
        series = df[col].astype(str).str.strip().replace("", np.nan)
        if series.notna().any():
            item_cols.append(col)

    return item_cols


def prepare_item_timeseries(df: pd.DataFrame, item_name: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    선택한 아이템의
    1) 주차별 매출 데이터
    2) 월별 매출 데이터
    를 만들어 반환합니다.
    """
    if df.empty:
        raise ValueError("시트 데이터가 비어 있습니다.")

    if "연도/주" not in df.columns:
        raise ValueError("필수 컬럼 '연도/주'가 없습니다.")

    if item_name not in df.columns:
        raise ValueError(f"선택한 아이템 컬럼 '{item_name}'이 없습니다.")

    temp = df[["연도/주", item_name]].copy()
    temp.columns = ["year_week", "sales"]

    # 연도/주 형식 데이터만 사용
    temp["year_week"] = temp["year_week"].astype(str).str.strip()
    temp = temp[temp["year_week"].str.match(r"^\d{4}-\d{1,2}$", na=False)].copy()

    if temp.empty:
        raise ValueError("연도/주 형식의 데이터가 없습니다. 예: 2025-01")

    # 숫자 변환
    temp["sales"] = temp["sales"].apply(clean_number)

    # 날짜 변환
    temp["week_start"] = temp["year_week"].apply(parse_yearweek_to_date)
    temp = temp.dropna(subset=["week_start"]).copy()

    # 숫자 없는 건 0 처리
    temp["sales"] = temp["sales"].fillna(0)

    # 주차 정렬
    weekly_df = temp.sort_values("week_start").reset_index(drop=True)

    # 월 컬럼 생성
    weekly_df["month"] = weekly_df["week_start"].dt.to_period("M").dt.to_timestamp()

    # 월별 매출 합산
    monthly_df = (
        weekly_df.groupby("month", as_index=False)["sales"]
        .sum()
        .sort_values("month")
        .reset_index(drop=True)
    )

    return weekly_df, monthly_df

def prepare_item_timeseries_by_code(df: pd.DataFrame, item_code: str) -> Tuple[str, pd.DataFrame, pd.DataFrame]:
    """
    PLC DB(가로형)에서 아이템코드(영문 2자리)로 매칭되는 컬럼을 찾아
    주차/월 집계를 반환합니다.
    반환: (matched_column_name, weekly_df, monthly_df)
    """
    if df.empty:
        raise ValueError("PLC DB 데이터가 비어 있습니다.")
    if "연도/주" not in df.columns:
        raise ValueError("PLC DB에 필수 컬럼 '연도/주'가 없습니다.")

    target = normalize_item_code(item_code)
    if not target:
        raise ValueError("매칭할 아이템코드(영문 2자리)가 올바르지 않습니다.")

    candidates: List[str] = []
    for c in df.columns:
        if str(c).strip() in {"연도/주", "", " "}:
            continue
        if normalize_item_code(c) == target:
            candidates.append(c)

    if not candidates:
        raise ValueError(f"PLC DB에서 아이템코드 '{target}'에 매칭되는 컬럼을 찾지 못했습니다.")

    matched_col = candidates[0]
    weekly_df, monthly_df = prepare_item_timeseries(df, matched_col)

    extracted = weekly_df["year_week"].astype(str).str.extract(r"(?P<year>\d{4})-(?P<week>\d{1,2})")
    weekly_df["week"] = pd.to_numeric(extracted["week"], errors="coerce").fillna(0).astype(int)

    return matched_col, weekly_df, monthly_df


# =========================
# 차트 생성
# =========================
def build_dual_line_chart(item_name: str, weekly_df: pd.DataFrame, monthly_df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()

    # 주차별 매출 선
    fig.add_trace(
        go.Scatter(
            x=weekly_df["week_start"],
            y=weekly_df["sales"],
            mode="lines+markers",
            name="주차별 매출",
            hovertemplate="주차 시작일: %{x|%Y-%m-%d}<br>매출: %{y:,.0f}<extra></extra>",
        )
    )

    # 월별 매출 선
    fig.add_trace(
        go.Scatter(
            x=monthly_df["month"],
            y=monthly_df["sales"],
            mode="lines+markers",
            name="월별 매출",
            hovertemplate="월: %{x|%Y-%m}<br>매출: %{y:,.0f}<extra></extra>",
        )
    )

    fig.update_layout(
        title=f"{item_name} 주차별/월별 매출 추이",
        xaxis_title="연도/주",
        yaxis_title="매출",
        height=620,
        hovermode="x unified",
        margin=dict(l=30, r=30, t=70, b=30),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
    )

    fig.update_yaxes(tickformat=",.0f")

    return fig

# =========================
# 월별 매출 형태 판별 (단봉 / 다봉)
# =========================
def smooth_series(values: np.ndarray, window: int = 2) -> np.ndarray:
    """
    월별 매출을 약하게 smoothing
    """
    if len(values) < window:
        return values.copy()

    return pd.Series(values).rolling(
        window=window,
        center=True,
        min_periods=1
    ).mean().values


def find_significant_peaks(
    values: np.ndarray,
    min_peak_ratio: float = 0.35,
    min_prominence_ratio: float = 0.10,
    min_distance: int = 1
) -> List[int]:
    """
    의미 있는 peak만 찾기
    """
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
    peaks = find_significant_peaks(
        values,
        min_peak_ratio=0.25,
        min_prominence_ratio=0.05,
        min_distance=2
    )

    if len(peaks) < 2:
        return False, peaks

    mx = np.max(values)
    if mx <= 0:
        return False, peaks

    strong = [p for p in peaks if values[p] >= mx * 0.6]

    if len(strong) < 2:
        return False, peaks

    strong = sorted(strong)

    for i in range(len(strong) - 1):
        p1 = strong[i]
        p2 = strong[i + 1]

        # 두 피크 간 거리 6 이상
        if p2 - p1 < 6:
            continue

        # 피크 사이 저점 존재 여부 확인
        valley = np.min(values[p1:p2 + 1])
        lower_peak = min(values[p1], values[p2])

        if lower_peak > 0 and valley / lower_peak <= 0.85:
            return True, [p1, p2]

    return False, peaks


def is_single_peak(values: np.ndarray) -> Tuple[bool, List[int]]:
    peaks = find_significant_peaks(
        values,
        min_peak_ratio=0.30,
        min_prominence_ratio=0.08,
        min_distance=2
    )

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

def get_season(week: int) -> str:
    if 9 <= week <= 18:
        return "SPRING"
    if 19 <= week <= 30:
        return "SUMMER"
    if 31 <= week <= 40:
        return "FALL"
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

def season_label_kr(code: str) -> str:
    mapping = {
        "SUMMER_PEAK": "여름형",
        "WINTER_PEAK": "겨울형",
        "SPRING_FALL_PEAK": "봄가을형",
        "SPRING_PEAK": "봄형",
        "FALL_PEAK": "가을형",
        "ALL_SEASON": "올시즌형",
    }
    return mapping.get(code, code)

def analyze_core_season(weekly_df: pd.DataFrame) -> Tuple[str, dict]:
    """
    PLC 주차별 데이터(week 포함)를 기준으로 시즌 비중을 계산하고 핵심시즌을 반환합니다.
    반환: (season_category_code, ratios_percent_dict)
    """
    if weekly_df.empty:
        return "ALL_SEASON", {"SPRING": 0.0, "SUMMER": 0.0, "FALL": 0.0, "WINTER": 0.0}

    if "week" not in weekly_df.columns:
        extracted = weekly_df["year_week"].astype(str).str.extract(r"(?P<year>\d{4})-(?P<week>\d{1,2})")
        weeks = pd.to_numeric(extracted["week"], errors="coerce").fillna(0).astype(int)
    else:
        weeks = weekly_df["week"].fillna(0).astype(int)

    df2 = weekly_df.copy()
    df2["week"] = weeks
    df2["season"] = df2["week"].apply(get_season)

    season_sum = df2.groupby("season", as_index=False)["sales"].sum()
    pivot = {s: 0.0 for s in ["SPRING", "SUMMER", "FALL", "WINTER"]}
    for _, r in season_sum.iterrows():
        pivot[str(r["season"])] = float(r["sales"])

    total = sum(pivot.values())
    if total <= 0:
        ratios = {k: 0.0 for k in pivot}
        return "ALL_SEASON", ratios

    ratios = {k: (v / total) for k, v in pivot.items()}
    row = pd.Series(
        {
            "SPRING_RATIO": ratios["SPRING"],
            "SUMMER_RATIO": ratios["SUMMER"],
            "FALL_RATIO": ratios["FALL"],
            "WINTER_RATIO": ratios["WINTER"],
        }
    )
    cat = classify_item(row)
    ratios_percent = {k: round(v * 100, 1) for k, v in ratios.items()}
    return cat, ratios_percent

def analyze_monthly_shape_and_peaks(monthly_df: pd.DataFrame) -> Tuple[str, List[str], str]:
    """
    PLC 월별 매출을 기준으로 단봉/쌍봉/올시즌형과 피크월(YYYY-MM)을 반환합니다.
    """
    if monthly_df.empty or "sales" not in monthly_df.columns:
        return "판단불가", [], "월별 데이터가 없습니다."

    y = monthly_df["sales"].values.astype(float)
    if len(y) < 3:
        return "판단불가", [], "월별 데이터가 3개 미만입니다."

    y_smooth = smooth_series(y, window=2)
    month_labels = monthly_df["month"].dt.strftime("%Y-%m").tolist()

    is_double, double_peaks = is_double_peak(y_smooth)
    if is_double:
        peaks = [month_labels[i] for i in double_peaks]
        return "쌍봉형", peaks, f"의미 있는 피크 2개: {', '.join(peaks)}"

    is_single, single_peaks = is_single_peak(y_smooth)
    if is_single:
        peaks = [month_labels[i] for i in single_peaks]
        return "단봉형", peaks, f"의미 있는 피크 1개: {', '.join(peaks)}"

    if is_all_season(y_smooth):
        return "올시즌형", [], "큰 피크 없이 전체적으로 고르게 분포"

    mx_idx = int(np.argmax(y)) if len(y) else 0
    peak = month_labels[mx_idx] if month_labels else ""
    return "단봉형", [peak] if peak else [], "명확한 다중 피크가 없어 최대 월을 피크로 간주"

# =========================
# 메인 화면
# =========================
def main():
    st.set_page_config(page_title="PLC 매출 분석", layout="wide")
    st.title("PLC 매출 분석")

    plc_raw = load_sheet_data()
    if plc_raw.empty:
        st.warning("PLC DB(시트) 데이터를 불러오지 못했습니다.")
        return

    try:
        plc_long = preprocess_plc_db(plc_raw)
    except Exception as e:
        st.error(f"PLC DB 전처리 실패: {e}")
        return

    if plc_long.empty:
        st.warning("PLC DB 전처리 결과가 비었습니다. 데이터 구조를 확인해주세요.")
        return

    tab1, tab2 = st.tabs(["아이템 직접 선택", "스타일/SKU → 아이템코드 매칭(PLC 분석)"])

    with tab1:
        items = (
            plc_long[["item_name", "item_code"]]
            .dropna()
            .drop_duplicates()
            .sort_values(["item_name", "item_code"])
            .reset_index(drop=True)
        )
        if items.empty:
            st.warning("PLC DB에서 아이템 목록을 만들지 못했습니다.")
            return

        items["_label"] = items.apply(lambda r: f"{r['item_name']} ({r['item_code']})", axis=1)
        labels = items["_label"].tolist()

        st.markdown("개별 차트 확인할 아이템")
        default_idx = labels.index("가디건 (CK)") if "가디건 (CK)" in labels else 0
        selected_label = st.selectbox("개별 차트 확인할 아이템", options=labels, index=default_idx, label_visibility="collapsed")
        selected_row = items[items["_label"] == selected_label].iloc[0]

        sel_code = str(selected_row["item_code"]).strip()
        weekly_df, monthly_df = build_timeseries_from_plc_long(plc_long, item_code=sel_code)
        if weekly_df.empty or monthly_df.empty:
            st.warning("선택 아이템의 시계열을 만들지 못했습니다.")
            return

        shape_label, shape_reason = classify_shape(selected_label, monthly_df)
        st.markdown(f"### 형태: {shape_label}")
        st.caption(shape_reason)

        fig = build_dual_line_chart(selected_label, weekly_df, monthly_df)
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        final_df = load_final_sheet_data()
        if final_df.empty:
            st.warning("FINAL 시트 설정(워크시트명)이 없거나 데이터를 불러오지 못했습니다. secrets.toml의 [sheets]에 final(또는 final_worksheet 등) 키로 워크시트명을 넣어주세요.")
            return

        st.markdown("FINAL 시트에서 스타일/SKU를 선택하면, 3~4번째 2자리(영문) 아이템코드로 PLC DB 컬럼을 매칭해 분석합니다.")

        cols = [c for c in final_df.columns if str(c).strip()]
        style_col = st.selectbox(
            "스타일코드 컬럼",
            options=cols,
            index=cols.index("스타일코드") if "스타일코드" in cols else (cols.index("style") if "style" in cols else 0),
        )
        sku_col = st.selectbox(
            "SKU코드 컬럼",
            options=cols,
            index=cols.index("sku") if "sku" in cols else (cols.index("SKU") if "SKU" in cols else (cols.index("SKU코드") if "SKU코드" in cols else 0)),
        )

        tmp = final_df.copy()
        tmp["style_item_code"] = tmp[style_col].apply(extract_item_code_from_code)
        tmp["sku_item_code"] = tmp[sku_col].apply(extract_item_code_from_code)
        tmp["match_item_code"] = tmp["style_item_code"].fillna(tmp["sku_item_code"])
        tmp = tmp[tmp["match_item_code"].notna()].copy()
        tmp["match_item_code"] = tmp["match_item_code"].astype(str).str.upper().str.strip()

        if tmp.empty:
            st.warning("FINAL 데이터에서 아이템코드(3~4번째 2자리 영문)를 추출하지 못했습니다. 스타일/SKU 값 형식을 확인해주세요.")
            return

        tmp["_label"] = tmp.apply(
            lambda r: f"{r.get(style_col, '')} / {r.get(sku_col, '')}  →  {r['match_item_code']}",
            axis=1,
        )

        selected_label = st.selectbox("스타일/SKU 선택", options=tmp["_label"].tolist())
        selected_row = tmp[tmp["_label"] == selected_label].iloc[0]
        item_code = str(selected_row["match_item_code"]).strip().upper()

        plc_weekly_df, plc_monthly_df = build_timeseries_from_plc_long(plc_long, item_code=item_code)
        if plc_weekly_df.empty or plc_monthly_df.empty:
            st.error(f"PLC DB에서 아이템코드 '{item_code}' 매칭 결과가 없습니다.")
            return

        # 매칭된 아이템명(가능하면 같이 표시)
        matched_names = plc_weekly_df["item_name"].dropna().unique().tolist() if "item_name" in plc_weekly_df.columns else []
        matched_name = matched_names[0] if matched_names else ""

        st.markdown("### 매칭 결과")
        st.write({"match_item_code": item_code, "matched_item_name": matched_name})

        core_season_code, ratios_percent = analyze_core_season(plc_weekly_df)
        shape2, peak_months, shape_reason2 = analyze_monthly_shape_and_peaks(plc_monthly_df)

        st.markdown("### PLC 분석 결과")
        st.write(
            {
                "핵심시즌": season_label_kr(core_season_code),
                "시즌 비중(%)": ratios_percent,
                "월별 형태": shape2,
                "피크월": peak_months,
            }
        )
        st.caption(shape_reason2)

        fig2 = build_dual_line_chart(f"PLC({item_code}) {matched_name}".strip(), plc_weekly_df, plc_monthly_df)
        st.plotly_chart(fig2, use_container_width=True)

    


if __name__ == "__main__":
    main()
