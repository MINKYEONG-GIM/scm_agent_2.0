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


@st.cache_data(ttl=300)
def load_plc_df() -> pd.DataFrame:
    sheets_cfg = get_sheets_config()
    plc_sheet = sheets_cfg.get("plc_db") or "plc db"
    return load_sheet_as_df(plc_sheet)


@st.cache_data(ttl=300)
def load_final_df() -> pd.DataFrame:
    sheets_cfg = get_sheets_config()
    final_sheet = sheets_cfg.get("final") or "final"
    return load_sheet_as_df(final_sheet)


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


def prepare_plc_item_timeseries(
    plc_df: pd.DataFrame,
    item_code: str
) -> Tuple[str, pd.DataFrame, pd.DataFrame]:
    """
    plc db에서 item_code에 해당하는 행을 찾아
    주차별/월별 시계열을 생성한다.
    반환:
    - item_name
    - weekly_df
    - monthly_df
    """
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
            "year_week": col,
            "week_start": week_start,
            "sales": 0 if pd.isna(sales) else float(sales)
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


def get_final_item_options(final_df: pd.DataFrame) -> pd.DataFrame:
    df = prepare_final_df(final_df).copy()

    # sku_name + item_code 기준으로 유니크
    options = (
        df[["sku_name", "item_code", "sku"]]
        .dropna(subset=["sku_name"])
        .drop_duplicates()
        .sort_values(["sku_name", "sku"])
        .reset_index(drop=True)
    )
    return options


# =========================
# 차트 생성
# =========================
def build_dual_line_chart(
    item_name: str,
    weekly_df: pd.DataFrame,
    monthly_df: pd.DataFrame
) -> go.Figure:
    fig = go.Figure()

    # 주차별 단계별 색상 선
    stage_df = weekly_df.copy().reset_index(drop=True)

    if "stage" in stage_df.columns:
        current_stage = None
        segment_x = []
        segment_y = []

        for i, row in stage_df.iterrows():
            stage = row["stage"]
            x = row["week_start"]
            y = row["sales"]

            if current_stage is None:
                current_stage = stage
                segment_x = [x]
                segment_y = [y]
            elif stage == current_stage:
                segment_x.append(x)
                segment_y.append(y)
            else:
                fig.add_trace(
                    go.Scatter(
                        x=segment_x,
                        y=segment_y,
                        mode="lines+markers",
                        name=current_stage,
                        line=dict(color=STAGE_COLORS.get(current_stage, "#333")),
                        marker=dict(size=7),
                        hovertemplate="주차 시작일: %{x|%Y-%m-%d}<br>판매량: %{y:,.0f}<br>단계: " + current_stage + "<extra></extra>",
                        showlegend=True
                    )
                )
                current_stage = stage
                segment_x = [x]
                segment_y = [y]

        # 마지막 구간
        if segment_x:
            fig.add_trace(
                go.Scatter(
                    x=segment_x,
                    y=segment_y,
                    mode="lines+markers",
                    name=current_stage,
                    line=dict(color=STAGE_COLORS.get(current_stage, "#333")),
                    marker=dict(size=7),
                    hovertemplate="주차 시작일: %{x|%Y-%m-%d}<br>판매량: %{y:,.0f}<br>단계: " + current_stage + "<extra></extra>",
                    showlegend=True
                )
            )
    else:
        # 혹시 stage 없을 때 fallback
        fig.add_trace(
            go.Scatter(
                x=weekly_df["week_start"],
                y=weekly_df["sales"],
                mode="lines+markers",
                name="주차별 판매량",
                hovertemplate="주차 시작일: %{x|%Y-%m-%d}<br>판매량: %{y:,.0f}<extra></extra>",
            )
        )

    # 월별 매출 선
    fig.add_trace(
        go.Scatter(
            x=monthly_df["month"],
            y=monthly_df["sales"],
            mode="lines+markers",
            name="월별 매출",
            line=dict(width=3, color="#cfcfcf"),
            marker=dict(size=7, color="#cfcfcf"),
            connectgaps=True,
            yaxis="y2",
            hovertemplate="월: %{x|%Y-%m}<br>매출: %{y:,.0f}<extra></extra>",
        )
    )

    fig.update_layout(
        title=f"{item_name} 주차별 단계 / 월별 형태 기준 매출 추이",
        xaxis_title="날짜",
        yaxis_title="주차별 판매량",
        yaxis2=dict(
            title="월별 매출",
            overlaying="y",
            side="right",
            showgrid=False
        ),
        height=650,
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


def classify_weekly_stages_by_shape(
    weekly_df: pd.DataFrame,
    shape_label: str
) -> pd.DataFrame:
    """
    월별 shape_label(단봉형/쌍봉형/올시즌형)에 따라
    주차별 판매량을 기반으로 단계 라벨을 강제 부여한다.
    """

    df = weekly_df.copy().reset_index(drop=True)
    y = df["sales"].astype(float).fillna(0).values
    n = len(df)

    if n == 0:
        df["stage"] = []
        return df

    # 기본값
    df["stage"] = "성숙"

    # 판매량 smoothing
    smooth = pd.Series(y).rolling(window=3, center=True, min_periods=1).mean().values

    # 전주 대비 증감
    diff = np.diff(smooth, prepend=smooth[0])

    # ----------------------------
    # 공통 유틸
    # ----------------------------
    def safe_argmax(arr):
        if len(arr) == 0:
            return 0
        return int(np.argmax(arr))

    def clip_idx(v):
        return max(0, min(n - 1, int(v)))

    # ============================
    # 1) 단봉형
    # 도입 > 성장 > 피크 > 성숙 > 쇠퇴
    # ============================
    if shape_label == "단봉형":
        peak_idx = int(np.argmax(y))

        # 도입: 앞쪽 최대 4주
        intro_end = min(3, max(1, peak_idx // 3))
        # 성장: 도입 다음부터 피크 직전
        growth_start = intro_end + 1
        growth_end = max(growth_start, peak_idx - 1)

        # 피크: 최고점 1주
        peak_start = peak_idx
        peak_end = peak_idx

        # 성숙: 피크 직후 2~4주 정도
        maturity_start = min(n - 1, peak_end + 1)
        maturity_end = min(n - 1, maturity_start + 2)

        # 쇠퇴: 이후 전체
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

    # ============================
    # 2) 쌍봉형
    # 도입 > 성장 > 피크 > 성숙 > 비시즌 > 성숙 > 피크2 > 성숙 > 쇠퇴
    # ============================
    if shape_label == "쌍봉형":
        peaks = find_significant_peaks(
            smooth,
            min_peak_ratio=0.25,
            min_prominence_ratio=0.05,
            min_distance=2
        )

        # 강한 피크만
        if len(peaks) >= 2:
            peaks = sorted(peaks, key=lambda i: smooth[i], reverse=True)[:2]
            peaks = sorted(peaks)
            peak1, peak2 = peaks[0], peaks[1]
        else:
            # 실패 시 fallback
            peak1 = safe_argmax(smooth[: max(1, n // 2)])
            peak2 = safe_argmax(smooth[max(peak1 + 1, 1):]) + max(peak1 + 1, 1)
            if peak2 >= n:
                peak2 = n - 1

        # valley = 두 피크 사이 최저점
        if peak2 > peak1 + 1:
            valley_rel = np.argmin(smooth[peak1:peak2 + 1])
            valley_idx = peak1 + valley_rel
        else:
            valley_idx = min(n - 1, peak1 + 1)

        intro_end = min(3, max(1, peak1 // 3))
        growth_start = intro_end + 1
        growth_end = max(growth_start, peak1 - 1)

        peak1_idx = peak1

        # 첫 성숙
        maturity1_start = min(n - 1, peak1_idx + 1)
        maturity1_end = min(n - 1, max(maturity1_start, valley_idx - 2))

        # 비시즌
        offseason_start = min(n - 1, max(maturity1_end + 1, valley_idx - 1))
        offseason_end = min(n - 1, valley_idx + 1)

        # 두 번째 성숙
        maturity2_start = min(n - 1, offseason_end + 1)
        maturity2_end = min(n - 1, max(maturity2_start, peak2 - 1))

        peak2_idx = peak2

        maturity3_start = min(n - 1, peak2_idx + 1)
        maturity3_end = min(n - 1, maturity3_start + 1)

        decline_start = min(n - 1, maturity3_end + 1)

        df.loc[:intro_end, "stage"] = "도입"
        if growth_start <= growth_end:
            df.loc[growth_start:growth_end, "stage"] = "성장"

        df.loc[peak1_idx:peak1_idx, "stage"] = "피크"

        if maturity1_start <= maturity1_end:
            df.loc[maturity1_start:maturity1_end, "stage"] = "성숙"

        if offseason_start <= offseason_end:
            df.loc[offseason_start:offseason_end, "stage"] = "비시즌"

        if maturity2_start <= maturity2_end:
            df.loc[maturity2_start:maturity2_end, "stage"] = "성숙"

        df.loc[peak2_idx:peak2_idx, "stage"] = "피크2"

        if maturity3_start <= maturity3_end:
            df.loc[maturity3_start:maturity3_end, "stage"] = "성숙"

        if decline_start < n:
            df.loc[decline_start:, "stage"] = "쇠퇴"

        return df

    # ============================
    # 3) 올시즌형
    # 요청에 명시된 강제 규칙이 없으므로
    # 도입 > 성장 > 성숙 > 쇠퇴 로 단순 처리
    # ============================
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

def extract_item_code_from_sku(sku: str) -> str:
    s = str(sku).strip()
    if len(s) >= 4:
        return s[2:4]
    return ""

def prepare_final_df(final_df: pd.DataFrame) -> pd.DataFrame:
    df = final_df.copy()

    required_cols = ["sku", "sku_name", "날짜", "판매량"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"final 시트 필수 컬럼이 없습니다: {missing}")

    df["item_code"] = df["sku"].apply(extract_item_code_from_sku)
    df["판매량"] = df["판매량"].apply(clean_number).fillna(0)
    df["날짜"] = pd.to_datetime(
        df["날짜"].astype(str)
            .str.replace(".", "-", regex=False)
            .str.replace(" ", "", regex=False),
        errors="coerce"
)
    return df



# =========================
# 메인 화면
# =========================

STAGE_COLORS = {
    "도입": "#1f77b4",   # 파랑
    "성장": "#2ca02c",   # 초록
    "피크": "#d62728",   # 빨강
    "피크2": "#d62728",  # 빨강
    "성숙": "#9467bd",   # 보라
    "비시즌": "#7f7f7f", # 회색
    "쇠퇴": "#8c564b",   # 갈색
}


def main():
    st.set_page_config(page_title="아이템 매출 추이", layout="wide")

    plc_df = load_plc_df()
    final_df = load_final_df()

    if plc_df.empty:
        st.warning("plc db 데이터가 없습니다.")
        return

    if final_df.empty:
        st.warning("final 데이터가 없습니다.")
        return

    final_prepared = prepare_final_df(final_df)
    options_df = get_final_item_options(final_prepared)

    if options_df.empty:
        st.warning("final에서 선택 가능한 SKU 데이터가 없습니다.")
        return

    options_df["label"] = options_df.apply(
        lambda r: f"{r['sku_name']} | 코드:{r['item_code']} | SKU:{r['sku']}",
        axis=1
    )

    selected_label = st.selectbox(
        "개별 차트 확인할 상품",
        options=options_df["label"].tolist()
    )

    selected_row = options_df[options_df["label"] == selected_label].iloc[0]
    selected_item_code = selected_row["item_code"]

    item_name, weekly_df, monthly_df = prepare_plc_item_timeseries(plc_df, selected_item_code)
    shape_label, shape_reason = classify_shape(item_name, monthly_df)
    weekly_df = classify_weekly_stages_by_shape(weekly_df, shape_label)

    st.markdown(f"### 아이템명: {item_name}")
    st.markdown(f"### 형태: {shape_label}")
    st.caption(shape_reason)

    if shape_label == "단봉형":
        st.markdown("**주차 단계 순서:** 도입 > 성장 > 피크 > 성숙 > 쇠퇴")
    elif shape_label == "쌍봉형":
        st.markdown("**주차 단계 순서:** 도입 > 성장 > 피크 > 성숙 > 비시즌 > 성숙 > 피크2 > 성숙 > 쇠퇴")
    else:
        st.markdown("**주차 단계 순서:** 도입 > 성장 > 성숙 > 쇠퇴")

    fig = build_dual_line_chart(item_name, weekly_df, monthly_df)
    st.plotly_chart(fig, use_container_width=True)

    st.dataframe(
        weekly_df[["year_week", "sales", "stage"]],
        use_container_width=True,
        hide_index=True
    )

    


if __name__ == "__main__":
    main()
