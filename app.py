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

def call_openai_chat_json(messages: List[dict], json_schema: Optional[dict] = None) -> dict:
    """
    Chat Completions API를 호출해서 JSON 응답을 받습니다.
    """
    api_key = get_gpt_gpi()
    if not api_key:
        raise ValueError("OpenAI API Key가 없습니다. st.secrets 또는 환경변수에 gpt_gpi / OPENAI_API_KEY를 설정하세요.")

    if json_schema is None:
        json_schema = {
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

    payload = {
        "model": "gpt-4.1-mini",
        "messages": messages,
        "temperature": 0,
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                **json_schema
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


def forecast_with_gpt(
    item_name: str,
    shape_label: str,
    weekly_df: pd.DataFrame,
    final_item_df: pd.DataFrame
) -> pd.DataFrame:

    # ------------------------------------------------------------
    # 비중 기반 예측(절대값 직접 생성 금지)
    # - 작년(weekly_df)의 주차별 비중 분포를 기반으로
    # - 올해는 "현재까지 누적 실적"이 작년 같은 주차까지 누적 대비 어느 정도인지로 스케일만 보정
    # - 남은 주차 판매량은 작년 남은 주차 비중대로 배분
    # ------------------------------------------------------------
    df_last = weekly_df.copy()
    df_last["week_no"] = df_last["week_start"].dt.isocalendar().week.astype(int)
    df_last["sales"] = pd.to_numeric(df_last["sales"], errors="coerce").fillna(0.0)

    last_total = float(df_last["sales"].sum())
    if last_total <= 0:
        return pd.DataFrame(columns=["날짜", "forecast"])

    df_last["ratio"] = df_last["sales"] / last_total
    ratio_by_week = df_last.groupby("week_no")["ratio"].sum().to_dict()
    last_sales_by_week = df_last.groupby("week_no")["sales"].sum().to_dict()

    # 올해 실측 주차별(ISO week) 판매량
    df_this = final_item_df.dropna(subset=["날짜"]).copy()
    if df_this.empty:
        this_sales_by_week = {}
    else:
        df_this["iso_year"] = df_this["날짜"].dt.isocalendar().year.astype(int)
        df_this["week_no"] = df_this["날짜"].dt.isocalendar().week.astype(int)
        df_this["판매량"] = pd.to_numeric(df_this["판매량"], errors="coerce").fillna(0.0)
        # "올해" 기준으로만 집계 (현재 연도)
        this_year = int(pd.Timestamp.today().year)
        df_this = df_this[df_this["iso_year"] == this_year].copy()
        this_sales_by_week = df_this.groupby("week_no")["판매량"].sum().to_dict()

    this_year = int(pd.Timestamp.today().year)
    current_week_no = int(pd.Timestamp.today().isocalendar().week)

    # 올해 현재까지 누적 / 작년 같은 주차까지 누적
    this_to_date = float(sum(v for w, v in this_sales_by_week.items() if int(w) <= current_week_no))
    last_to_date = float(sum(v for w, v in last_sales_by_week.items() if int(w) <= current_week_no))

    # 작년 같은 기간 누적이 0이면 스케일 추정이 불가하므로 보수적으로 0 예측
    if last_to_date <= 0:
        return pd.DataFrame(columns=["날짜", "forecast"])

    # ------------------------------------------------------------
    # 예외 규칙: 올해 판매가 계속 0인 제품
    # - 다다음주(현재+2)부터 판매 1장을 가정
    # - 그 이후는 작년 비중(ratio) 상대비로 판매량 산출
    #   예: forecast[w] = round(1 * ratio[w] / ratio[seed_week])
    # ------------------------------------------------------------
    this_has_any_sales = any(float(v) > 0 for v in this_sales_by_week.values())
    if (not this_has_any_sales) and this_to_date <= 0:
        remaining_weeks = sorted([int(w) for w in ratio_by_week.keys() if int(w) > current_week_no])
        if not remaining_weeks:
            return pd.DataFrame(columns=["날짜", "forecast"])

        seed_week = current_week_no + 2
        seed_value = 1

        # seed_week이 범위를 벗어나거나 비중이 0이면, 남은 주차 중 비중>0인 첫 주차로 대체
        if seed_week not in ratio_by_week or float(ratio_by_week.get(seed_week, 0.0)) <= 0:
            seed_week = None
            for w in remaining_weeks:
                if float(ratio_by_week.get(w, 0.0)) > 0:
                    seed_week = w
                    break
            if seed_week is None:
                # 남은 주차 비중이 모두 0이면 예측 불가
                return pd.DataFrame(columns=["날짜", "forecast"])

        seed_ratio = float(ratio_by_week.get(seed_week, 0.0))
        if seed_ratio <= 0:
            return pd.DataFrame(columns=["날짜", "forecast"])

        forecast_weeks = [w for w in remaining_weeks if w >= seed_week]
        forecast_values = []
        forecast_dates = []
        for w in forecast_weeks:
            r = float(ratio_by_week.get(w, 0.0))
            v = int(round(seed_value * (r / seed_ratio)))
            forecast_values.append(max(0, v))
            d = pd.to_datetime(f"{this_year}-W{w:02d}-1", format="%G-W%V-%u", errors="coerce")
            forecast_dates.append(d)

        forecast_df = pd.DataFrame({"날짜": forecast_dates, "forecast": forecast_values}).dropna(subset=["날짜"])
        return forecast_df

    scale = this_to_date / last_to_date
    expected_total = last_total * scale
    remaining_total = max(0.0, expected_total - this_to_date)

    # 남은 주차(현재 주차 이후) 비중만 추출 후 재정규화
    remaining_weeks = sorted([int(w) for w in ratio_by_week.keys() if int(w) > current_week_no])
    if not remaining_weeks:
        return pd.DataFrame(columns=["날짜", "forecast"])

    remaining_ratio_sum = float(sum(ratio_by_week.get(w, 0.0) for w in remaining_weeks))
    if remaining_ratio_sum <= 0:
        return pd.DataFrame(columns=["날짜", "forecast"])

    forecast_values = []
    forecast_dates = []
    for w in remaining_weeks:
        r = float(ratio_by_week.get(w, 0.0)) / remaining_ratio_sum
        v = int(round(remaining_total * r))
        forecast_values.append(v)
        # ISO week의 월요일 날짜
        d = pd.to_datetime(f"{this_year}-W{w:02d}-1", format="%G-W%V-%u", errors="coerce")
        forecast_dates.append(d)

    forecast_df = pd.DataFrame({"날짜": forecast_dates, "forecast": forecast_values}).dropna(subset=["날짜"])
    return forecast_df

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

    # plant_name + sku_name + item_code 기준으로 유니크
    options = (
        df[["plant_name", "sku_name", "item_code", "sku"]]
        .dropna(subset=["sku_name", "plant_name"])
        .drop_duplicates()
        .sort_values(["plant_name", "sku_name", "sku"])
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

    weekly_week_no = weekly_df["week_start"].dt.isocalendar().week.astype(int)

    # 주차별 판매량 연결선
    fig.add_trace(
        go.Scatter(
            x=weekly_df["week_start"],
            y=weekly_df["sales"],
            mode="lines",
            name="주차별 판매량(연결선)",
            line=dict(color="#b0b0b0", width=2),
            hoverinfo="skip",
            showlegend=False,
            connectgaps=True,
        )
    )

    # 주차별 단계별 색상 선
    stage_df = weekly_df.copy().reset_index(drop=True)
    stage_df["week_no"] = stage_df["week_start"].dt.isocalendar().week.astype(int)

    if "stage" in stage_df.columns:
        current_stage = None
        segment_x = []
        segment_y = []
        segment_week = []

        for i, row in stage_df.iterrows():
            stage = row["stage"]
            x = row["week_start"]
            y = row["sales"]
            w = int(row["week_no"])

            if current_stage is None:
                current_stage = stage
                segment_x = [x]
                segment_y = [y]
                segment_week = [w]
            elif stage == current_stage:
                segment_x.append(x)
                segment_y.append(y)
                segment_week.append(w)
            else:
                fig.add_trace(
                    go.Scatter(
                        x=segment_x,
                        y=segment_y,
                        customdata=segment_week,
                        mode="lines+markers",
                        name=current_stage,
                        line=dict(color=STAGE_COLORS.get(current_stage, "#333"), width=3),
                        marker=dict(size=7),
                        hovertemplate="주차: %{customdata}주차<br>주차 시작일: %{x|%Y-%m-%d}<br>판매량: %{y:,.0f}<br>단계: " + current_stage + "<extra></extra>",
                        showlegend=True
                    )
                )
                current_stage = stage
                segment_x = [x]
                segment_y = [y]
                segment_week = [w]

        # 마지막 구간
        if segment_x:
            fig.add_trace(
                go.Scatter(
                    x=segment_x,
                    y=segment_y,
                    customdata=segment_week,
                    mode="lines+markers",
                    name=current_stage,
                    line=dict(color=STAGE_COLORS.get(current_stage, "#333"), width=3),
                    marker=dict(size=7),
                    hovertemplate="주차: %{customdata}주차<br>주차 시작일: %{x|%Y-%m-%d}<br>판매량: %{y:,.0f}<br>단계: " + current_stage + "<extra></extra>",
                    showlegend=True
                )
            )

    # 월별 매출
    fig.add_trace(
        go.Scatter(
            x=monthly_df["month"],
            y=monthly_df["sales"],
            customdata=monthly_df["month"].dt.isocalendar().week.astype(int),
            mode="lines+markers",
            name="월별 매출",
            line=dict(width=3, color="#bfbfbf"),
            marker=dict(size=7, color="#bfbfbf"),
            fill="tozeroy",
            fillcolor="rgba(191, 191, 191, 0.25)",
            connectgaps=True,
            yaxis="y2",
            hovertemplate="월: %{x|%Y-%m}<br>(참고) %{customdata}주차<br>매출: %{y:,.0f}<extra></extra>",
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

    fig.update_yaxes(tickformat=",.0f", rangemode="tozero")
    fig.update_layout(
        yaxis=dict(rangemode="tozero"),
        yaxis2=dict(rangemode="tozero"),
    )
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

    # plant_name은 매장 필터용(없으면 '전체'로 처리)
    if "plant_name" not in df.columns:
        df["plant_name"] = "전체"

    # sku 문자열 정리
    df["sku"] = df["sku"].astype(str).str.strip()
    df["sku_name"] = df["sku_name"].astype(str).str.strip()
    df["plant_name"] = df["plant_name"].astype(str).str.strip().replace("", "전체")

    df["item_code"] = df["sku"].apply(extract_item_code_from_sku)
    df["판매량"] = df["판매량"].apply(clean_number).fillna(0)

    # 선택 컬럼(있으면 숫자 정리)
    optional_numeric_cols = ["기초재고", "분배량", "출고량(회전 등)", "로스"]
    for c in optional_numeric_cols:
        if c in df.columns:
            df[c] = df[c].apply(clean_number)

    # 날짜 문자열 정리
    raw_date = (
        df["날짜"]
        .astype(str)
        .str.strip()
        .str.replace(".", "-", regex=False)
        .str.replace("/", "-", regex=False)
        .str.replace(" ", "", regex=False)
    )

    # 예: 02월25일 -> 2026-02-25 로 변환
    current_year = pd.Timestamp.today().year
    raw_date = raw_date.str.replace(
        r"^(\d{1,2})월(\d{1,2})일$",
        rf"{current_year}-\1-\2",
        regex=True
    )

    df["날짜"] = pd.to_datetime(raw_date, errors="coerce")

    return df


def build_year_compare_table(
    weekly_df: pd.DataFrame,
    final_item_df: pd.DataFrame,
    selected_sku: str,
    selected_sku_name: str
) -> pd.DataFrame:
    """
    표 컬럼:
    SKU / SKU_NAME / 주차 / 작년의 해당 주차 판매비중(%) / 올해 해당 주차 판매량 (장)
    """

    # -----------------------------
    # 1) 작년 주차별 판매비중 계산
    # -----------------------------
    last_year_df = weekly_df.copy()

    last_year_df["week_no"] = last_year_df["week_start"].dt.isocalendar().week.astype(int)
    last_year_df["sales"] = pd.to_numeric(last_year_df["sales"], errors="coerce").fillna(0)

    total_last_year_sales = last_year_df["sales"].sum()

    if total_last_year_sales > 0:
        last_year_df["last_year_ratio_pct"] = (
            last_year_df["sales"] / total_last_year_sales * 100
        )
    else:
        last_year_df["last_year_ratio_pct"] = 0.0

    # 주차 라벨: 예) 8주차
    last_year_df["주차"] = last_year_df["week_no"].astype(str) + "주차"

    # -----------------------------
    # 2) 올해 주차별 지표 계산
    # -----------------------------
    this_year_df = final_item_df.copy()
    this_year_df = this_year_df.dropna(subset=["날짜"]).copy()

    if not this_year_df.empty:
        this_year_df["week_no"] = this_year_df["날짜"].dt.isocalendar().week.astype(int)
        this_year_df["판매량"] = pd.to_numeric(this_year_df["판매량"], errors="coerce").fillna(0)

        agg_map = {"판매량": "sum"}

        if "분배량" in this_year_df.columns:
            this_year_df["분배량"] = pd.to_numeric(this_year_df["분배량"], errors="coerce").fillna(0)
            agg_map["분배량"] = "sum"

        if "출고량(회전 등)" in this_year_df.columns:
            this_year_df["출고량(회전 등)"] = pd.to_numeric(this_year_df["출고량(회전 등)"], errors="coerce").fillna(0)
            agg_map["출고량(회전 등)"] = "sum"

        if "로스" in this_year_df.columns:
            this_year_df["로스"] = pd.to_numeric(this_year_df["로스"], errors="coerce").fillna(0)
            agg_map["로스"] = "sum"

        this_year_weekly = this_year_df.groupby("week_no", as_index=False).agg(agg_map)

        # 기초재고: 주차 내 가장 이른 날짜 행의 값(없으면 NaN)
        if "기초재고" in this_year_df.columns:
            tmp_base = this_year_df.dropna(subset=["기초재고"]).copy()
            if not tmp_base.empty:
                tmp_base = tmp_base.sort_values(["week_no", "날짜"])
                base_weekly = tmp_base.groupby("week_no", as_index=False).first()[["week_no", "기초재고"]]
            else:
                base_weekly = pd.DataFrame(columns=["week_no", "기초재고"])

            this_year_weekly = this_year_weekly.merge(base_weekly, on="week_no", how="left")

        this_year_weekly = this_year_weekly.rename(columns={"판매량": "올해 해당 주차 판매량 (장)"})
    else:
        this_year_weekly = pd.DataFrame(
            columns=["week_no", "올해 해당 주차 판매량 (장)", "기초재고", "분배량", "출고량(회전 등)", "로스"]
        )

    # -----------------------------
    # 3) 작년 주차 기준으로 merge
    # -----------------------------
    # 표에는 5단계 중심으로 보이게: 피크/피크2는 성숙으로 표기(그래프 단계와 동일 출처)
    last_year_df["stage_for_table"] = last_year_df["stage"].replace({
        "피크": "성숙",
        "피크2": "성숙",
    }).fillna("")

    # 주차는 '주차' 열에 있으므로 이 열에는 단계명만 표시
    last_year_df["작년 단계"] = last_year_df["stage_for_table"].astype(str)

    result = last_year_df[
        ["week_no", "주차", "last_year_ratio_pct", "작년 단계"]
    ].merge(
        this_year_weekly,
        on="week_no",
        how="left"
    )

    result = result.sort_values("week_no").reset_index(drop=True)

    for col in ["올해 해당 주차 판매량 (장)", "분배량", "출고량(회전 등)", "로스"]:
        if col in result.columns:
            result[col] = pd.to_numeric(result[col], errors="coerce").fillna(0).round().astype(int)

    if "기초재고" in result.columns:
        result["기초재고"] = pd.to_numeric(result["기초재고"], errors="coerce").fillna(0).round().astype(int)

    result["SKU"] = selected_sku
    result["SKU_NAME"] = selected_sku_name

    # last_year_ratio_pct 는 이미 0~100(%) 단위로 계산됨
    result["작년의 해당 주차 판매비중(%)"] = result["last_year_ratio_pct"].round(1)

    result = result[
        [
            "SKU",
            "SKU_NAME",
            "week_no",
            "주차",
            "작년의 해당 주차 판매비중(%)",
            "기초재고",
            "올해 해당 주차 판매량 (장)",
            "분배량",
            "출고량(회전 등)",
            "로스",
            "작년 단계",
        ]
    ].copy()

    return result


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

    # 표시용 라벨(상품 선택 드롭다운에는 매장명 미포함)
    options_df["display_label"] = options_df.apply(
        lambda r: f"{r['sku_name']} | 코드:{r['item_code']} | SKU:{r['sku']}",
        axis=1
    )
    # 내부 식별용 키(전체 매장일 때도 선택이 겹치지 않도록)
    options_df["option_id"] = options_df.apply(
        lambda r: f"{r['plant_name']}||{r['sku']}",
        axis=1
    )

    col_a, col_b = st.columns([1, 3])

    with col_a:
        plant_values = options_df["plant_name"].dropna().astype(str).str.strip()
        plant_values = plant_values[plant_values != ""]
        plant_options = ["전체"] + sorted([p for p in plant_values.unique().tolist() if p != "전체"])

        selected_plant = st.selectbox(
            "매장 선택",
            options=plant_options
        )

    with col_b:
        filtered_options_df = options_df.copy()
        if selected_plant != "전체":
            filtered_options_df = filtered_options_df[filtered_options_df["plant_name"] == selected_plant].copy()

        selected_option_id = st.selectbox(
            "개별 차트 확인할 상품",
            options=filtered_options_df["option_id"].tolist(),
            format_func=lambda oid: filtered_options_df.loc[
                filtered_options_df["option_id"] == oid, "display_label"
            ].iloc[0]
        )

    selected_row = filtered_options_df[filtered_options_df["option_id"] == selected_option_id].iloc[0]
    selected_item_code = selected_row["item_code"]

    selected_sku = str(selected_row["sku"]).strip()
    selected_sku_name = str(selected_row["sku_name"]).strip()

    final_item_df = final_prepared[
        final_prepared["sku"].astype(str).str.strip() == selected_sku
    ].copy()

    if selected_plant != "전체":
        final_item_df = final_item_df[
            final_item_df["plant_name"].astype(str).str.strip() == selected_plant
        ].copy()
    
    st.write("selected_sku:", selected_sku)
    st.write("final_item_df 건수:", len(final_item_df))
    st.write("날짜 null 개수:", final_item_df["날짜"].isna().sum())

    item_name, weekly_df, monthly_df = prepare_plc_item_timeseries(plc_df, selected_item_code)
    shape_label, shape_reason = classify_shape(item_name, monthly_df)
    weekly_df = classify_weekly_stages_by_shape(weekly_df, shape_label)

    compare_table_df = build_year_compare_table(
        weekly_df=weekly_df,
        final_item_df=final_item_df,
        selected_sku=selected_sku,
        selected_sku_name=selected_sku_name
    )

    try:
        forecast_df = forecast_with_gpt(
            item_name,
            shape_label,
            weekly_df,
            final_item_df
        )
    except Exception as e:
        forecast_df = pd.DataFrame(columns=["날짜", "forecast"])
        st.error(f"GPT 예측 호출 실패: {e}")

    # -----------------------------
    # 주차별 작년 비중 / 올해 판매량 비교표
    # - 미래 주차(현재 주차 이후)는 GPT 예측값으로 채우고 빨간색 표시
    # -----------------------------
    st.markdown("### 주차별 작년 비중 / 올해 판매량 비교표")

    this_year = int(pd.Timestamp.today().year)
    current_week_no = int(pd.Timestamp.today().isocalendar().week)

    forecast_week_map = {}
    if not forecast_df.empty and "날짜" in forecast_df.columns and "forecast" in forecast_df.columns:
        tmp_fc = forecast_df.dropna(subset=["날짜"]).copy()
        if not tmp_fc.empty:
            tmp_fc["year"] = tmp_fc["날짜"].dt.isocalendar().year.astype(int)
            tmp_fc = tmp_fc[tmp_fc["year"] == this_year].copy()
            if not tmp_fc.empty:
                tmp_fc["week_no"] = tmp_fc["날짜"].dt.isocalendar().week.astype(int)
                tmp_fc["forecast"] = pd.to_numeric(tmp_fc["forecast"], errors="coerce").fillna(0)
                forecast_week_map = (
                    tmp_fc.groupby("week_no")["forecast"].sum().round().astype(int).to_dict()
                )

    compare_table_df = compare_table_df.copy()
    is_future_week = compare_table_df["week_no"].astype(int) > current_week_no
    has_forecast = compare_table_df["week_no"].astype(int).map(lambda w: w in forecast_week_map)
    predict_mask = is_future_week & has_forecast

    if predict_mask.any():
        compare_table_df.loc[predict_mask, "올해 해당 주차 판매량 (장)"] = (
            compare_table_df.loc[predict_mask, "week_no"].astype(int).map(forecast_week_map).fillna(0).astype(int)
        )

    # -----------------------------
    # 미래 주차 기초재고 예측
    # 기초재고(t) = 기초재고(t-1) - 판매량(t-1) + 분배량(t-1) - 출고량(t-1)
    # -----------------------------
    compare_table_df = compare_table_df.sort_values("week_no").reset_index(drop=True)

    for col in ["기초재고", "올해 해당 주차 판매량 (장)", "분배량", "출고량(회전 등)"]:
        if col not in compare_table_df.columns:
            compare_table_df[col] = 0
        compare_table_df[col] = pd.to_numeric(compare_table_df[col], errors="coerce").fillna(0).astype(int)

    # 기준이 되는 "마지막 실측 기초재고" 주차를 잡는다(현재 주차까지 중, 기초재고>0인 마지막 주차)
    anchor_candidates = compare_table_df[
        (compare_table_df["week_no"].astype(int) <= current_week_no)
        & (compare_table_df["기초재고"].astype(int) > 0)
    ]
    if not anchor_candidates.empty:
        anchor_week = int(anchor_candidates["week_no"].max())
    else:
        anchor_week = None

    base_pred_mask = pd.Series(False, index=compare_table_df.index)
    if anchor_week is not None:
        week_list = compare_table_df["week_no"].astype(int).tolist()
        # anchor_week 이후부터 순차 계산(현재 주차 이후는 예측으로 간주)
        for i in range(1, len(week_list)):
            w_prev = int(week_list[i - 1])
            w_cur = int(week_list[i])
            if w_cur <= anchor_week:
                continue

            prev_base = int(compare_table_df.loc[i - 1, "기초재고"])
            prev_sales = int(compare_table_df.loc[i - 1, "올해 해당 주차 판매량 (장)"])
            prev_dist = int(compare_table_df.loc[i - 1, "분배량"])
            prev_ship = int(compare_table_df.loc[i - 1, "출고량(회전 등)"])

            predicted_base = prev_base - prev_sales + prev_dist - prev_ship
            compare_table_df.loc[i, "기초재고"] = int(predicted_base)

            if w_cur > current_week_no:
                base_pred_mask.iloc[i] = True

    # -----------------------------
    # 로스 계산(조건부 표시)
    # - 기초재고 < 판매량일 때만 표시
    # - 로스 = 기초재고 - 판매량
    # -----------------------------
    sales_col = "올해 해당 주차 판매량 (장)"
    loss = compare_table_df["기초재고"].astype(int) - compare_table_df[sales_col].astype(int)
    compare_table_df["로스"] = np.where(compare_table_df["기초재고"].astype(int) < compare_table_df[sales_col].astype(int), loss, 0).astype(int)

    display_df = compare_table_df[
        [
            "주차",
            "작년의 해당 주차 판매비중(%)",
            "기초재고",
            "올해 해당 주차 판매량 (장)",
            "분배량",
            "출고량(회전 등)",
            "로스",
            "작년 단계",
        ]
    ].copy()

    def _style_compare_table(_):
        styles = pd.DataFrame("", index=display_df.index, columns=display_df.columns)

        # 현재 주차: 강조(노랑)
        mask_current = compare_table_df["week_no"].astype(int) == current_week_no
        styles.loc[mask_current, "올해 해당 주차 판매량 (장)"] = "background-color: #FFF3BF; font-weight: 700;"

        # 미래 주차 예측값: 빨강
        styles.loc[predict_mask, "올해 해당 주차 판매량 (장)"] = "color: #C92A2A; font-weight: 800;"
        styles.loc[base_pred_mask.values, "기초재고"] = "color: #C92A2A; font-weight: 800;"
        styles.loc[is_future_week, "로스"] = "color: #C92A2A; font-weight: 800;"
        return styles

    st.dataframe(
        display_df.style.apply(_style_compare_table, axis=None),
        use_container_width=True,
        hide_index=True,
        column_config={
            "작년의 해당 주차 판매비중(%)": st.column_config.NumberColumn(
                "작년의 해당 주차 판매비중(%)",
                format="%.2f%%",
            ),
            "기초재고": st.column_config.NumberColumn(
                "기초재고",
                format="%d",
            ),
            "올해 해당 주차 판매량 (장)": st.column_config.NumberColumn(
                "올해 해당 주차 판매량 (장)",
                format="%d",
            ),
            "분배량": st.column_config.NumberColumn(
                "분배량",
                format="%d",
            ),
            "출고량(회전 등)": st.column_config.NumberColumn(
                "출고량(회전 등)",
                format="%d",
            ),
            "로스": st.column_config.NumberColumn(
                "로스",
                format="%d",
            ),
        }
    )
    st.markdown(
        "<span style='color:#C92A2A; font-weight:800;'>빨간색 수치는 AI 예측값입니다.</span>",
        unsafe_allow_html=True
    )

    st.markdown(f"### 아이템명: {item_name}")
    st.markdown(f"### 형태: {shape_label}")
    st.caption(shape_reason)

    if shape_label == "단봉형":
        st.markdown("**주차 단계 순서:** 도입 > 성장 > 피크 > 성숙 > 쇠퇴")
    elif shape_label == "쌍봉형":
        st.markdown("**주차 단계 순서:** 도입 > 성장 > 피크 > 성숙 > 비시즌 > 성숙 > 피크2 > 성숙 > 쇠퇴")
    else:
        st.markdown("**주차 단계 순서:** 도입 > 성장 > 성숙 > 쇠퇴")

    col1, col2 = st.columns([1, 1])

    with col1:

        st.markdown("### 작년 매출")

        fig1 = build_dual_line_chart(
            item_name,
            weekly_df,
            monthly_df
        )

        st.plotly_chart(fig1, use_container_width=True)


    with col2:
        st.markdown("### 올해 매출 + AI 예측")
    
        fig2 = go.Figure()

        this_year = int(pd.Timestamp.today().year)
        year_start = pd.Timestamp(this_year, 1, 1)
        year_end = pd.Timestamp(this_year, 12, 31)
    
        plot_final_df = final_item_df.dropna(subset=["날짜"]).copy()
    
        real_week = (
            plot_final_df
            .sort_values("날짜")
            .groupby(pd.Grouper(key="날짜", freq="W"))["판매량"]
            .sum()
            .reset_index()
        )
    
        real_week = real_week[real_week["판매량"] > 0].copy()
        real_week = real_week[
            (real_week["날짜"] >= year_start) & (real_week["날짜"] <= year_end)
        ].copy()

        forecast_plot_df = forecast_df.copy()
        if not forecast_plot_df.empty:
            forecast_plot_df = forecast_plot_df[
                (forecast_plot_df["날짜"] >= year_start) & (forecast_plot_df["날짜"] <= year_end)
            ].copy()
    
        if real_week.empty:
            st.warning("올해 매출 데이터가 없습니다. final 시트의 날짜 형식 또는 sku 매칭을 확인하세요.")
        else:
            real_week = real_week.copy()
            real_week["week_no"] = real_week["날짜"].dt.isocalendar().week.astype(int)
            fig2.add_trace(
                go.Scatter(
                    x=real_week["날짜"],
                    y=real_week["판매량"],
                    customdata=real_week["week_no"],
                    name="올해 매출",
                    mode="lines+markers",
                    hovertemplate=" %{customdata}주차<br>날짜: %{x|%Y-%m-%d}<br>판매량: %{y:,.0f}<extra></extra>",
                )
            )
    
        if not forecast_plot_df.empty:
            forecast_plot_df = forecast_plot_df.copy()
            forecast_plot_df["week_no"] = forecast_plot_df["날짜"].dt.isocalendar().week.astype(int)
            fig2.add_trace(
                go.Scatter(
                    x=forecast_plot_df["날짜"],
                    y=forecast_plot_df["forecast"],
                    customdata=forecast_plot_df["week_no"],
                    name="GPT 예측",
                    mode="lines+markers",
                    line=dict(dash="dash"),
                    hovertemplate="주차: %{customdata}주차<br>날짜: %{x|%Y-%m-%d}<br>예측 판매량: %{y:,.0f}<extra></extra>",
                )
            )
    
        fig2.update_layout(
            title=f"{item_name} 올해 매출 및 연말 예측",
            xaxis_title="날짜",
            yaxis_title="판매량",
            height=650,
            hovermode="x unified",
            xaxis=dict(range=[year_start, year_end]),
            yaxis=dict(rangemode="tozero")
        )
    
        st.plotly_chart(fig2, use_container_width=True)

if __name__ == "__main__":
    main()
