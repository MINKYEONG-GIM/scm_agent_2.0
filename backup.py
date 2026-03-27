import os
import json
import math
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

def classify_shape(item_name: str, monthly_df: pd.DataFrame, use_gpt: bool = True) -> Tuple[str, str]:
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

    # 1차 판단: GPT(옵션)
    if use_gpt:
        try:
            result = call_openai_chat_json(messages)
            return result["shape_label"], f"GPT 1차 판별: {result['reason']}"
        except Exception:
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


def get_current_stage_label(weekly_df_with_stage: pd.DataFrame, week_no: int) -> str:
    """
    주차별 stage가 있는 weekly_df에서 특정 ISO week_no의 단계명을 반환합니다.
    같은 week_no가 여러 행이면 마지막 행을 사용합니다.
    """
    if weekly_df_with_stage is None or weekly_df_with_stage.empty:
        return ""
    if "week_start" not in weekly_df_with_stage.columns or "stage" not in weekly_df_with_stage.columns:
        return ""

    tmp = weekly_df_with_stage.copy()
    tmp["week_no"] = tmp["week_start"].dt.isocalendar().week.astype(int)
    sub = tmp[tmp["week_no"].astype(int) == int(week_no)]
    if sub.empty:
        return ""
    return str(sub.iloc[-1]["stage"]).strip()


def compute_reorder_summary(
    *,
    weekly_df: pd.DataFrame,
    final_df_scope: pd.DataFrame,
    selected_sku: str,
    selected_sku_name: str,
    item_name: str,
    shape_label: str,
    lead_days: Optional[int],
    this_year: int,
) -> dict:
    """
    화면 표시에 필요한 값만 계산:
    - deadline_date: 로스 시작(해당 주 월요일) - lead_time(일)
    - qty: 리오더 권장 수량(장)
    - loss_start_week: 로스 시작 주차(ISO week)

    lead_days가 없거나(리드타임 미기입), 예측기간 내 로스가 없으면 deadline_date=None, qty=0.
    """
    if lead_days is None:
        return {"deadline_date": None, "qty": 0, "loss_start_week": None}

    compare_table_df_local = build_year_compare_table(
        weekly_df=weekly_df,
        final_item_df=final_df_scope,
        selected_sku=str(selected_sku).strip(),
        selected_sku_name=str(selected_sku_name).strip(),
        week_label_year=int(this_year),
    )

    try:
        forecast_df_local = forecast_with_gpt(
            item_name,
            shape_label,
            weekly_df,
            final_df_scope,
        )
    except Exception:
        forecast_df_local = pd.DataFrame(columns=["날짜", "forecast"])

    current_week_no_local = int(pd.Timestamp.today().isocalendar().week)
    sales_col = "올해 해당 주차 판매량 (장)"

    # 미래 주차 예측 판매량 반영(기존 로직 유지)
    forecast_week_map = {}
    if (
        not forecast_df_local.empty
        and "날짜" in forecast_df_local.columns
        and "forecast" in forecast_df_local.columns
    ):
        tmp_fc = forecast_df_local.dropna(subset=["날짜"]).copy()
        if not tmp_fc.empty:
            tmp_fc["year"] = tmp_fc["날짜"].dt.isocalendar().year.astype(int)
            tmp_fc = tmp_fc[tmp_fc["year"] == int(this_year)].copy()
            if not tmp_fc.empty:
                tmp_fc["week_no"] = tmp_fc["날짜"].dt.isocalendar().week.astype(int)
                tmp_fc["forecast"] = pd.to_numeric(tmp_fc["forecast"], errors="coerce").fillna(0)
                forecast_week_map = (
                    tmp_fc.groupby("week_no")["forecast"].sum().round().astype(int).to_dict()
                )

    compare_table_df_local = compare_table_df_local.copy()
    compare_table_df_local = (
        compare_table_df_local.sort_values("week_no", ascending=True, kind="mergesort")
        .reset_index(drop=True)
    )
    is_future_week = compare_table_df_local["week_no"].astype(int) > current_week_no_local
    has_forecast = compare_table_df_local["week_no"].astype(int).map(lambda w: w in forecast_week_map)
    predict_mask = is_future_week & has_forecast

    if predict_mask.any():
        compare_table_df_local.loc[predict_mask, sales_col] = (
            compare_table_df_local.loc[predict_mask, "week_no"]
            .astype(int)
            .map(forecast_week_map)
            .fillna(0)
            .astype(int)
        )

    # 기초재고 롤링 계산(기존 로직 유지)
    for col in ["기초재고", sales_col, "분배량", "출고량(회전 등)"]:
        if col not in compare_table_df_local.columns:
            compare_table_df_local[col] = 0
        compare_table_df_local[col] = pd.to_numeric(compare_table_df_local[col], errors="coerce").fillna(0).astype(int)

    week_list = compare_table_df_local["week_no"].astype(int).tolist()
    for i in range(1, len(week_list)):
        w_cur = int(week_list[i])

        observed_base = int(compare_table_df_local.loc[i, "기초재고"])
        if (w_cur <= current_week_no_local) and (observed_base != 0):
            continue

        prev_base = int(compare_table_df_local.loc[i - 1, "기초재고"])
        prev_sales = int(compare_table_df_local.loc[i - 1, sales_col])
        prev_dist = int(compare_table_df_local.loc[i - 1, "분배량"])
        prev_ship = int(compare_table_df_local.loc[i - 1, "출고량(회전 등)"])

        predicted_base = prev_base - prev_sales + prev_dist - prev_ship
        compare_table_df_local.loc[i, "기초재고"] = int(predicted_base)

    base_raw = compare_table_df_local["기초재고"].astype(int).copy()
    compare_table_df_local["기초재고"] = np.maximum(base_raw, 0).astype(int)

    # 로스 계산(기존 로직 유지)
    n_rows = len(compare_table_df_local)
    loss_vals = []
    prev_loss = 0
    for i in range(n_rows):
        w = int(compare_table_df_local.loc[i, "week_no"])
        if w <= current_week_no_local:
            loss_vals.append(0)
            continue

        raw_b = int(base_raw.iloc[i])
        sales = int(compare_table_df_local.loc[i, sales_col])
        if raw_b <= 0:
            cur_loss = prev_loss - sales
        elif raw_b < sales:
            cur_loss = raw_b - sales
        else:
            cur_loss = 0
        prev_loss = cur_loss
        loss_vals.append(cur_loss)
    compare_table_df_local["로스"] = loss_vals

    neg_loss = compare_table_df_local[
        (compare_table_df_local["week_no"].astype(int) > current_week_no_local)
        & (compare_table_df_local["로스"].astype(float) < 0)
    ]
    if neg_loss.empty:
        return {"deadline_date": None, "qty": 0, "loss_start_week": None}

    loss_start_week = int(neg_loss.iloc[0]["week_no"])
    loss_start_monday = pd.to_datetime(
        f"{this_year}-W{loss_start_week:02d}-1",
        format="%G-W%V-%u",
        errors="coerce",
    )
    if pd.isna(loss_start_monday):
        deadline_date = None
    else:
        deadline_date = (loss_start_monday - pd.Timedelta(days=int(lead_days))).normalize()

    weeks_lead = max(1, math.ceil(float(lead_days) / 7.0))
    rec_week = loss_start_week - weeks_lead

    wm = compare_table_df_local["week_no"].astype(int)
    qty = int(
        compare_table_df_local.loc[
            (wm >= rec_week) & (wm < rec_week + weeks_lead),
            sales_col,
        ].sum()
    )
    if qty < 1:
        qty = max(1, abs(int(float(neg_loss.iloc[0]["로스"]))))

    return {
        "deadline_date": deadline_date,
        "qty": int(qty),
        "loss_start_week": int(loss_start_week),
    }

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


@st.cache_data(ttl=300)
def load_reorder_df() -> pd.DataFrame:
    sheets_cfg = get_sheets_config()
    reorder_sheet = sheets_cfg.get("reorder") or "reorder"
    return load_sheet_as_df(reorder_sheet)


def get_reorder_lead_time_days(reorder_df: pd.DataFrame, sku: str) -> Optional[int]:
    """
    reorder 시트에서 선택 SKU에 해당하는 lead_time(일)을 반환합니다.
    헤더가 sku가 중복이면 make_unique_headers로 sku, sku_2 등이 됩니다.
    """
    if reorder_df is None or reorder_df.empty:
        return None

    sku_key = str(sku).strip()
    if not sku_key:
        return None

    lt_col = None
    for c in reorder_df.columns:
        if str(c).strip().lower() == "lead_time":
            lt_col = c
            break
    if lt_col is None:
        return None

    sku_cols = [c for c in reorder_df.columns if str(c).strip().lower().startswith("sku")]
    if not sku_cols:
        return None

    for col in sku_cols:
        mask = reorder_df[col].astype(str).str.strip() == sku_key
        sub = reorder_df.loc[mask]
        if sub.empty:
            continue
        for _, row in sub.iterrows():
            v = clean_number(row[lt_col])
            if pd.notna(v):
                return int(round(float(v)))
    return None


def iso_week_monday_month_day(year: int, week_no: int) -> Optional[Tuple[int, int]]:
    """해당 연도 ISO 주차의 월요일 날짜를 (월, 일)로 반환합니다."""
    ts = pd.to_datetime(f"{year}-W{int(week_no):02d}-1", format="%G-W%V-%u", errors="coerce")
    if pd.isna(ts):
        return None
    return int(ts.month), int(ts.day)


def format_calendar_week_label(calendar_year: int, iso_week_no: int) -> str:
    """
    기준 연도(예: 2026)의 ISO 주차를 '26년 M월 W주차'로 표시합니다.
    W는 해당 달에서 월요일이 속한 '몇 번째 주'(1~5)입니다.
    """
    ts = pd.to_datetime(f"{calendar_year}-W{int(iso_week_no):02d}-1", format="%G-W%V-%u", errors="coerce")
    if pd.isna(ts):
        return f"{iso_week_no}주차"
    yy = calendar_year % 100
    m = int(ts.month)
    week_in_month = (int(ts.day) - 1) // 7 + 1
    return f"{yy:02d}년 {m}월 {week_in_month}주차"


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
        df[["plant_name", "sku_name", "item_code", "sku", "style_code"]]
        .dropna(subset=["sku_name", "plant_name"])
        .drop_duplicates()
        .sort_values(["plant_name", "style_code", "sku_name", "sku"])
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


def style_code_from_material(material: str) -> str:
    """final의 MATERIAL(또는 sku) 앞 10자리를 스타일코드로 사용합니다."""
    s = str(material).strip()
    return s[:10] if s else ""


def prepare_final_df(final_df: pd.DataFrame) -> pd.DataFrame:
    """
    final 데이터는 운영 중 컬럼 구조가 바뀔 수 있어
    - 구버전(final 시트): sku / sku_name / 날짜 / 판매량 (+ plant_name 선택)
    - 신버전(final DB): CALMONTH ... SSTOC_TMP_AMT (18컬럼)

    이 함수는 어떤 구조가 들어와도 아래 "표준 컬럼"으로 정규화해서 반환합니다.
    표준 컬럼: sku, sku_name, style_code, 날짜, 판매량, plant_name, item_code (+ 선택: 기초재고, 분배량, 출고량(회전 등), 로스)
    """
    df = final_df.copy()

    # --------------------------
    # 1) 신버전(final DB) 감지
    # --------------------------
    new_cols = {"CALDAY", "PLANT", "MATERIAL", "SALE"}
    is_new_schema = all(c in df.columns for c in new_cols)

    if is_new_schema:
        # 신 스키마 -> 표준 스키마로 매핑
        df = df.copy()

        df["sku"] = df["MATERIAL"].astype(str).str.strip()
        df["sku_name"] = df.get("MATERIAL", "").astype(str).str.strip()
        df["plant_name"] = df.get("PLANT", "전체").astype(str).str.strip().replace("", "전체")

        sale_raw = df["SALE"].apply(clean_number).fillna(0)
        if "SSTOC_TMP_QTY" in df.columns:
            sstoc = df["SSTOC_TMP_QTY"].apply(clean_number)
        else:
            sstoc = pd.Series(np.nan, index=df.index, dtype=float)

        # SSTOC_TMP_QTY 음수 행: 판매량=|SALE|, 출고량(회전 등)=|SSTOC|-|SALE|
        # (양수 SSTOC는 분배량으로만 반영, 음수 행의 분배량 증분은 없음)
        mask_sstoc_neg = sstoc.notna() & (sstoc < 0)

        df["판매량"] = sale_raw.astype(float)
        df.loc[mask_sstoc_neg, "판매량"] = sale_raw.loc[mask_sstoc_neg].abs()

        # 날짜는 CALDAY(YYYYMMDD) 기반
        calday = df["CALDAY"].astype(str).str.strip()
        # 혹시 float로 들어온 20260301.0 같은 값 방지
        calday = calday.str.replace(r"\.0$", "", regex=True)
        df["날짜"] = pd.to_datetime(calday, format="%Y%m%d", errors="coerce")

        # 재고/입고/주문을 기존 화면의 보조 지표로 연결(있으면)
        # - 기초재고: HSTOC_QTY
        # - 분배량: IPGO + SSTOC_TMP_QTY(양수만)
        # - 출고량(회전 등): 기본 ORDQTY, SSTOC 음수 행은 |SSTOC|-|SALE|
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

        # item_code는 기존 plc db(아이템코드) 매칭용인데,
        # 신 sku(MATERIAL)가 영문+숫자 조합일 수 있어 기본은 기존 규칙을 유지하되,
        # 실패 가능성을 낮추기 위해 비어 있으면 sku로 대체한다.
        df["item_code"] = df["sku"].apply(extract_item_code_from_sku)
        df.loc[df["item_code"].astype(str).str.strip() == "", "item_code"] = df["sku"]
        df["style_code"] = df["sku"].map(style_code_from_material)

        # 기존 코드가 기대하는 컬럼만 남기지는 않고, 원본 컬럼은 그대로 둔다(추후 확장 대비)
        return df

    # --------------------------
    # 2) 구버전(final 시트) 처리
    # --------------------------
    required_cols = ["sku", "sku_name", "날짜", "판매량"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(
            f"final 시트 필수 컬럼이 없습니다: {missing}. "
            f"현재 컬럼: {list(df.columns)}"
        )

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

    df["style_code"] = df["sku"].map(style_code_from_material)

    return df


def build_year_compare_table(
    weekly_df: pd.DataFrame,
    final_item_df: pd.DataFrame,
    selected_sku: str,
    selected_sku_name: str,
    week_label_year: int,
) -> pd.DataFrame:
    """
    표 컬럼:
    SKU / SKU_NAME / 주차 / 작년의 해당 주차 판매비중(%) / 올해 해당 주차 판매량 (장)
    week_label_year: 주차 열을 'YY년 M월 W주차'로 만들 때 사용할 기준 연도(보통 올해).
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

    last_year_df["주차"] = last_year_df["week_no"].astype(int).map(
        lambda w: format_calendar_week_label(week_label_year, int(w))
    )

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
    last_year_df["예측 단계"] = last_year_df["stage_for_table"].astype(str)

    result = last_year_df[
        ["week_no", "주차", "last_year_ratio_pct", "예측 단계"]
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

    # 일부 데이터 소스에서는 아래 컬럼이 없을 수 있어(스키마 변경/부분 적재),
    # 표 생성 단계에서 항상 존재하도록 0으로 보정한다.
    ensure_cols_defaults = {
        "기초재고": 0,
        "올해 해당 주차 판매량 (장)": 0,
        "분배량": 0,
        "출고량(회전 등)": 0,
        "로스": 0,
    }
    for c, default_v in ensure_cols_defaults.items():
        if c not in result.columns:
            result[c] = default_v

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
            "예측 단계",
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

    try:
        reorder_df = load_reorder_df()
    except Exception as e:
        reorder_df = pd.DataFrame()
        st.warning(f"reorder 시트를 불러오지 못했습니다: {e}")

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

    st.markdown("## 리오더 확인 대시보드 ")

    # SKU 단위로 유니크(매장 필터 제거)
    sku_option_df = (
        options_df[["sku", "sku_name", "item_code", "style_code"]]
        .dropna(subset=["sku"])
        .drop_duplicates(subset=["sku"])
        .copy()
    )
    sku_option_df["style_code"] = sku_option_df["style_code"].astype(str).str.strip().fillna("")
    sku_option_df["sku_name"] = sku_option_df["sku_name"].astype(str).str.strip().fillna("")
    sku_option_df["sku"] = sku_option_df["sku"].astype(str).str.strip()
    sku_option_df["item_code"] = sku_option_df["item_code"].astype(str).str.strip().fillna("")

    # 스타일 필터(선택)
    style_vals = sku_option_df["style_code"].dropna().astype(str).str.strip()
    style_vals = style_vals[style_vals != ""]
    style_options = ["전체"] + sorted(style_vals.unique().tolist())

    col_a, _ = st.columns([1, 2])
    with col_a:
        selected_style = st.selectbox("스타일코드 필터", options=style_options)

    if selected_style != "전체":
        sku_option_df = sku_option_df[sku_option_df["style_code"].astype(str).str.strip() == selected_style].copy()

    if sku_option_df.empty:
        st.warning("선택 조건에 해당하는 SKU가 없습니다.")
        return

    this_year = int(pd.Timestamp.today().year)
    today = pd.Timestamp.today().normalize()
    current_week_no = int(pd.Timestamp.today().isocalendar().week)

    rows = []
    prog = st.progress(0)
    total = len(sku_option_df)

    for i, r in sku_option_df.reset_index(drop=True).iterrows():
        sku = str(r["sku"]).strip()
        sku_name = str(r["sku_name"]).strip()
        item_code = str(r["item_code"]).strip()
        style_code = str(r["style_code"]).strip()

        final_item_all_plants_df = final_prepared[final_prepared["sku"].astype(str).str.strip() == sku].copy()
        lead_days = get_reorder_lead_time_days(reorder_df, sku)

        # plc db 기반 단계(쇠퇴 여부) 확인
        item_name = ""
        shape_label = "판단불가"
        shape_reason = ""
        current_stage = ""

        try:
            item_name, weekly_df, monthly_df = prepare_plc_item_timeseries(plc_df, item_code)
            shape_label, shape_reason = classify_shape(item_name, monthly_df, use_gpt=True)
            weekly_df_staged = classify_weekly_stages_by_shape(weekly_df, shape_label)
            current_stage = get_current_stage_label(weekly_df_staged, current_week_no)
        except Exception as e:
            weekly_df_staged = None
            current_stage = ""
            shape_reason = f"plc db 매칭 실패: {str(e)}"

        # 4) 확인 불필요 스타일(이미 쇠퇴기)
        if str(current_stage).strip() == "쇠퇴":
            sector = "4. 확인 불필요 스타일(쇠퇴기)"
            summary = {"deadline_date": None, "qty": 0, "loss_start_week": None}
            weeks_left = None
        else:
            # 리오더 요약 계산(로스 기반)
            if weekly_df_staged is None or final_item_all_plants_df.empty:
                summary = {"deadline_date": None, "qty": 0, "loss_start_week": None}
            else:
                summary = compute_reorder_summary(
                    weekly_df=weekly_df_staged,
                    final_df_scope=final_item_all_plants_df,
                    selected_sku=sku,
                    selected_sku_name=sku_name,
                    item_name=item_name,
                    shape_label=shape_label,
                    lead_days=lead_days,
                    this_year=this_year,
                )

            deadline = summary.get("deadline_date")
            qty = int(summary.get("qty") or 0)

            if lead_days is None:
                # 리드타임이 없으면 우선 확인 시급으로 올림(데이터 보완 필요)
                sector = "1. 확인 시급 스타일(리드타임 확인 필요)"
                weeks_left = None
            elif deadline is None or pd.isna(deadline) or qty <= 0:
                # 로스가 예측기간에 안 잡히면, '지금 당장'은 아니므로 6주+로 분류
                sector = "3. 6주 이상 후 리오더 스타일"
                weeks_left = None
            else:
                days_left = int((pd.Timestamp(deadline).normalize() - today).days)
                weeks_left = int(math.floor(days_left / 7.0))

                # 1) 확인 시급: 이번주~2주 내(<=2주)
                if weeks_left <= 2:
                    sector = "1. 확인 시급 스타일(이번주~2주 내)"
                # 2) 후순위 확인: 3~5주
                elif 3 <= weeks_left <= 5:
                    sector = "2. 후순위 확인 스타일(3~5주 내)"
                # 3) 6주 이상
                else:
                    sector = "3. 6주 이상 후 리오더 스타일"

        rows.append(
            {
                "섹터": sector,
                "style_code": style_code,
                "sku": sku,
                "sku_name": sku_name,
                "lead_time_days": lead_days if lead_days is not None else "",
                "reorder_deadline": summary.get("deadline_date"),
                "weeks_left": "" if weeks_left is None else int(weeks_left),
                "reorder_qty": int(summary.get("qty") or 0),
                "loss_start_week": summary.get("loss_start_week") if summary.get("loss_start_week") is not None else "",
                "current_stage": current_stage,
                "shape_label": shape_label,
                "shape_reason": shape_reason,
            }
        )

        prog.progress(min(1.0, (i + 1) / max(1, total)))

    result_df = pd.DataFrame(rows)
    if result_df.empty:
        st.warning("표시할 결과가 없습니다.")
        return

    # 표 표시용 정렬
    def _deadline_sort_key(v):
        try:
            if v is None or pd.isna(v):
                return pd.Timestamp.max
            return pd.Timestamp(v)
        except Exception:
            return pd.Timestamp.max

    result_df["_deadline_sort"] = result_df["reorder_deadline"].map(_deadline_sort_key)
    result_df = result_df.sort_values(["섹터", "_deadline_sort", "style_code", "sku_name", "sku"]).reset_index(drop=True)

    # 섹터별 출력(모든 SKU가 4개 중 하나에 반드시 포함)
    st.markdown("### 1) 확인 시급 스타일")
    st.caption("마감까지 2주 이하(또는 lead_time 누락으로 즉시 확인 필요)인 스타일/SKU와 권장 수량(장)")
    df1 = result_df[result_df["섹터"].astype(str).str.startswith("1.")].copy()
    st.dataframe(
        df1[["style_code", "sku", "sku_name", "lead_time_days", "reorder_deadline", "weeks_left", "reorder_qty", "loss_start_week"]],
        use_container_width=True,
        hide_index=True,
    )

    st.markdown("### 2) 후순위 확인 스타일")
    st.caption("마감까지 3~5주 남아 있어, 선행 섹터 처리 후 확인하면 되는 스타일/SKU")
    df2 = result_df[result_df["섹터"].astype(str).str.startswith("2.")].copy()
    st.dataframe(
        df2[["style_code", "sku", "sku_name", "lead_time_days", "reorder_deadline", "weeks_left", "reorder_qty", "loss_start_week"]],
        use_container_width=True,
        hide_index=True,
    )

    st.markdown("### 3) 6주 이상 후 리오더 스타일")
    st.caption("마감까지 6주 이상 남았거나, 예측기간 내 로스가 없어 당장 리오더 우선순위가 낮은 스타일/SKU")
    df3 = result_df[result_df["섹터"].astype(str).str.startswith("3.")].copy()
    st.dataframe(
        df3[["style_code", "sku", "sku_name", "lead_time_days", "reorder_deadline", "weeks_left", "reorder_qty", "loss_start_week"]],
        use_container_width=True,
        hide_index=True,
    )

    st.markdown("### 4) 확인 불필요 스타일(쇠퇴기)")
    st.caption("현재 주차 기준 단계가 '쇠퇴'로 분류되어, 추가 발주를 보지 않는 스타일/SKU")
    df4 = result_df[result_df["섹터"].astype(str).str.startswith("4.")].copy()
    st.dataframe(
        df4[["style_code", "sku", "sku_name", "current_stage", "shape_label"]],
        use_container_width=True,
        hide_index=True,
    )

    return

if __name__ == "__main__":
    main()
