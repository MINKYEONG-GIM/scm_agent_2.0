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
# 7. 공통 유틸
# =========================
def make_unique_headers(headers: List[str]) -> List[str]:
    """
    중복 컬럼명이 있을 때 자동으로 고유 이름으로 변경
    예: ['A', 'A', 'B'] -> ['A', 'A_2', 'B']
    """
    seen = {}
    result = []

    for h in headers:
        key = str(h).strip() if h is not None else ""
        if key == "":
            key = "unnamed"

        if key not in seen:
            seen[key] = 1
            result.append(key)
        else:
            seen[key] += 1
            result.append(f"{key}_{seen[key]}")

    return result


def clean_number(x) -> Optional[float]:
    """
    '1,234', '', None 같은 값을 float로 변환
    """
    if x is None:
        return None

    s = str(x).strip()
    if s == "":
        return None

    s = s.replace(",", "")
    try:
        return float(s)
    except ValueError:
        return None


def parse_year_week(week_str: str) -> Optional[Tuple[int, int]]:
    """
    '2025-01' -> (2025, 1)
    """
    if week_str is None:
        return None

    s = str(week_str).strip()
    m = re.match(r"^(\d{4})-(\d{1,2})$", s)
    if not m:
        return None

    year = int(m.group(1))
    week = int(m.group(2))
    return (year, week)


def sort_weeks(df: pd.DataFrame, week_col: str = "연도/주") -> pd.DataFrame:
    """
    연도/주 기준 정렬
    """
    temp = df.copy()
    parsed = temp[week_col].apply(parse_year_week)

    temp["_year"] = parsed.apply(lambda x: x[0] if x else 9999)
    temp["_week"] = parsed.apply(lambda x: x[1] if x else 9999)

    temp = temp.sort_values(["_year", "_week"]).reset_index(drop=True)
    temp = temp.drop(columns=["_year", "_week"])
    return temp


def prepare_item_timeseries(df: pd.DataFrame, week_col: str = "연도/주") -> pd.DataFrame:
    """
    원본 시트 데이터를 분석 가능한 형태로 변환
    - 빈 행 제거
    - 주차 정렬
    - 아이템 컬럼 숫자화
    """
    if week_col not in df.columns:
        raise ValueError(f"'{week_col}' 컬럼이 없습니다.")

    temp = df.copy()

    # 주차 없는 행 제거
    temp[week_col] = temp[week_col].astype(str).str.strip()
    temp = temp[temp[week_col] != ""].copy()

    # 주차 형식이 맞는 행만 유지
    temp = temp[temp[week_col].apply(lambda x: parse_year_week(x) is not None)].copy()

    # 정렬
    temp = sort_weeks(temp, week_col=week_col)

    # 숫자 컬럼 정리
    item_cols = [c for c in temp.columns if c != week_col]

    for col in item_cols:
        temp[col] = temp[col].apply(clean_number)

    return temp


def get_item_columns(df: pd.DataFrame, week_col: str = "연도/주") -> List[str]:
    """
    분석 대상 아이템 컬럼만 반환
    시즌분류, 메모성 컬럼은 제외
    """
    excluded_keywords = ["시즌분류"]
    cols = []

    for c in df.columns:
        if c == week_col:
            continue
        if any(k in c for k in excluded_keywords):
            continue
        cols.append(c)

    return cols


# =========================
# 8. 형태 분석용 특징 추출
# =========================
def count_local_peaks(y: np.ndarray) -> int:
    """
    단순 로컬 피크 개수
    y[i-1] < y[i] >= y[i+1]
    """
    if len(y) < 3:
        return 0

    peaks = 0
    for i in range(1, len(y) - 1):
        if y[i] > y[i - 1] and y[i] >= y[i + 1]:
            peaks += 1
    return peaks


def safe_skew(y: np.ndarray) -> float:
    """
    왜도 계산
    """
    if len(y) < 3:
        return 0.0

    mean = np.mean(y)
    std = np.std(y)

    if std == 0:
        return 0.0

    z = (y - mean) / std
    return float(np.mean(z ** 3))


def linear_slope(y: np.ndarray) -> float:
    """
    전체 구간 단순 선형 추세 기울기
    """
    if len(y) < 2:
        return 0.0

    x = np.arange(len(y))
    slope, _ = np.polyfit(x, y, 1)
    return float(slope)


def segment_mean_ratio(y: np.ndarray, start: int, end: int, total_mean: float) -> float:
    """
    특정 구간 평균 / 전체 평균
    """
    if total_mean == 0 or start >= end:
        return 0.0
    return float(np.mean(y[start:end]) / total_mean)


def extract_shape_features(weeks: List[str], values: List[float]) -> dict:
    """
    아이템 시계열에서 형태 분류용 특징 추출
    """
    y = np.array(values, dtype=float)

    # 결측 제거
    mask = ~np.isnan(y)
    y = y[mask]
    cleaned_weeks = [w for i, w in enumerate(weeks) if mask[i]]

    if len(y) < 8:
        return {
            "valid": False,
            "reason": "데이터 포인트가 8개 미만입니다."
        }

    # 음수는 0으로 보정
    y = np.where(y < 0, 0, y)

    total_mean = float(np.mean(y))
    total_std = float(np.std(y))
    peak_idx = int(np.argmax(y))
    peak_value = float(np.max(y))
    peak_week = cleaned_weeks[peak_idx]

    left = y[:peak_idx]
    right = y[peak_idx + 1:]

    left_width = peak_idx
    right_width = len(y) - peak_idx - 1

    left_mean_ratio = segment_mean_ratio(y, 0, len(y) // 3, total_mean)
    mid_mean_ratio = segment_mean_ratio(y, len(y) // 3, 2 * len(y) // 3, total_mean)
    right_mean_ratio = segment_mean_ratio(y, 2 * len(y) // 3, len(y), total_mean)

    overall_slope = linear_slope(y)
    first_half_slope = linear_slope(y[: max(2, len(y)//2)])
    second_half_slope = linear_slope(y[len(y)//2 :])

    peaks = count_local_peaks(y)
    skewness = safe_skew(y)

    # 피크 중심 좌우 면적
    left_area = float(np.sum(left)) if len(left) > 0 else 0.0
    right_area = float(np.sum(right)) if len(right) > 0 else 0.0

    # 정규분포 비슷한지 보는 보조값
    # peak가 너무 한쪽 끝에 있으면 정규형 가능성 낮음
    peak_position_ratio = peak_idx / max(1, len(y) - 1)

    # 피크 양옆이 점차 감소하는 정도
    monotonic_drop_left = 0
    if len(left) >= 2:
        inc_count = 0
        for i in range(len(left) - 1):
            if left[i] <= left[i + 1]:
                inc_count += 1
        monotonic_drop_left = inc_count / (len(left) - 1)

    monotonic_drop_right = 0
    if len(right) >= 2:
        dec_count = 0
        for i in range(len(right) - 1):
            if right[i] >= right[i + 1]:
                dec_count += 1
        monotonic_drop_right = dec_count / (len(right) - 1)

    cv = (total_std / total_mean) if total_mean != 0 else 0.0

    return {
        "valid": True,
        "n_points": len(y),
        "weeks": cleaned_weeks,
        "values": [float(v) for v in y],
        "mean": total_mean,
        "std": total_std,
        "cv": float(cv),
        "peak_idx": peak_idx,
        "peak_week": peak_week,
        "peak_value": peak_value,
        "peak_position_ratio": float(peak_position_ratio),
        "left_width": int(left_width),
        "right_width": int(right_width),
        "left_area": left_area,
        "right_area": right_area,
        "left_mean_ratio": float(left_mean_ratio),
        "mid_mean_ratio": float(mid_mean_ratio),
        "right_mean_ratio": float(right_mean_ratio),
        "overall_slope": float(overall_slope),
        "first_half_slope": float(first_half_slope),
        "second_half_slope": float(second_half_slope),
        "num_local_peaks": int(peaks),
        "skewness": float(skewness),
        "left_rise_consistency": float(monotonic_drop_left),
        "right_fall_consistency": float(monotonic_drop_right),
    }


def rule_based_shape_hint(features: dict) -> str:
    """
    GPT에 보내기 전 기본 힌트 생성
    """
    if not features.get("valid"):
        return "판단불가"

    peak_pos = features["peak_position_ratio"]
    peaks = features["num_local_peaks"]
    skewness = features["skewness"]
    first_slope = features["first_half_slope"]
    second_slope = features["second_half_slope"]
    left_ratio = features["left_mean_ratio"]
    mid_ratio = features["mid_mean_ratio"]
    right_ratio = features["right_mean_ratio"]

    # 다봉형
    if peaks >= 3:
        return "다봉형"

    # 이봉형
    if peaks == 2:
        return "이봉형"

    # 지속상승
    if first_slope > 0 and second_slope > 0 and right_ratio > left_ratio:
        return "지속상승형"

    # 지속하락
    if first_slope < 0 and second_slope < 0 and left_ratio > right_ratio:
        return "지속하락형"

    # 정규형 비슷
    if (
        peaks == 1
        and 0.35 <= peak_pos <= 0.65
        and abs(skewness) < 0.6
        and mid_ratio > left_ratio
        and mid_ratio > right_ratio
    ):
        return "정규분포형"

    # 우측꼬리
    if peaks == 1 and peak_pos < 0.45 and skewness > 0.5:
        return "우측꼬리형"

    # 좌측꼬리
    if peaks == 1 and peak_pos > 0.55 and skewness < -0.5:
        return "좌측꼬리형"

    return "불규칙형"


# =========================
# 9. GPT API 호출
# =========================
def call_openai_chat(messages: List[dict], model: str = "gpt-4.1-mini", temperature: float = 0.1) -> str:
    """
    Chat Completions API 호출
    """
    api_key = get_gpt_gpi()
    if not api_key:
        raise ValueError("OpenAI API Key가 없습니다. st.secrets 또는 환경변수에 gpt_gpi / OPENAI_API_KEY 를 설정하세요.")

    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
    }

    data = json.dumps(payload).encode("utf-8")
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

    try:
        return result["choices"][0]["message"]["content"].strip()
    except Exception:
        raise ValueError(f"응답 파싱 실패: {result}")


def classify_shape_with_gpt(item_name: str, features: dict) -> dict:
    """
    GPT에게 형태 분류 요청
    """
    if not features.get("valid"):
        return {
            "item": item_name,
            "shape_label": "판단불가",
            "reason": features.get("reason", "유효한 데이터 부족"),
            "confidence": "low"
        }

    hint = rule_based_shape_hint(features)

    prompt = f"""
너는 주차별 매출 추이를 형태 기준으로 분류하는 분석가다.

반드시 아래 카테고리 중 하나만 선택해라.
- 정규분포형
- 우측꼬리형
- 좌측꼬리형
- 이봉형
- 다봉형
- 지속상승형
- 지속하락형
- 불규칙형

판단 기준:
- 정규분포형: 가운데 근처에서 1개의 뚜렷한 peak가 있고, 좌우가 비교적 대칭
- 우측꼬리형: peak 이후 오른쪽 꼬리가 길다
- 좌측꼬리형: peak 이전 왼쪽 꼬리가 길다
- 이봉형: 큰 peak가 2개
- 다봉형: peak가 3개 이상
- 지속상승형: 전체적으로 계속 상승
- 지속하락형: 전체적으로 계속 하락
- 불규칙형: 위 어디에도 명확히 해당하지 않음

아이템명: {item_name}

기초 힌트:
- 규칙 기반 예비 판단: {hint}

특징값:
{json.dumps(features, ensure_ascii=False, indent=2)}

응답 형식은 반드시 JSON 하나만 반환:
{{
  "shape_label": "카테고리명",
  "confidence": "high|medium|low",
  "reason": "한글로 2~4문장 설명"
}}
""".strip()

    messages = [
        {
            "role": "system",
            "content": "너는 시계열 형태 분류를 정확하게 수행하는 데이터 분석가다. 반드시 JSON만 반환한다."
        },
        {
            "role": "user",
            "content": prompt
        }
    ]

    raw = call_openai_chat(messages=messages, model="gpt-4.1-mini", temperature=0.0)

    try:
        parsed = json.loads(raw)
        return {
            "item": item_name,
            "shape_label": parsed.get("shape_label", "판단불가"),
            "confidence": parsed.get("confidence", "low"),
            "reason": parsed.get("reason", ""),
            "rule_hint": hint,
        }
    except Exception:
        return {
            "item": item_name,
            "shape_label": hint,
            "confidence": "low",
            "reason": f"GPT 응답 JSON 파싱 실패. 원문: {raw}",
            "rule_hint": hint,
        }


# =========================
# 10. 전체 아이템 분류
# =========================
def run_shape_classification(df: pd.DataFrame, week_col: str = "연도/주") -> pd.DataFrame:
    """
    모든 아이템에 대해 형태 분류 실행
    """
    prepared = prepare_item_timeseries(df, week_col=week_col)
    item_cols = get_item_columns(prepared, week_col=week_col)

    results = []

    weeks = prepared[week_col].tolist()

    progress = st.progress(0)
    total = len(item_cols)

    for idx, item in enumerate(item_cols, start=1):
        values = prepared[item].tolist()
        features = extract_shape_features(weeks, values)
        result = classify_shape_with_gpt(item, features)

        # 보조 통계도 같이 저장
        result["peak_week"] = features.get("peak_week")
        result["peak_value"] = features.get("peak_value")
        result["num_local_peaks"] = features.get("num_local_peaks")
        result["skewness"] = features.get("skewness")
        result["n_points"] = features.get("n_points")

        results.append(result)
        progress.progress(idx / total)

    progress.empty()

    result_df = pd.DataFrame(results)
    return result_df


# =========================
# 11. 개별 아이템 차트
# =========================
def draw_item_chart(df: pd.DataFrame, item_name: str, week_col: str = "연도/주"):
    """
    선택 아이템의 주차별 매출 추이 시각화
    """
    prepared = prepare_item_timeseries(df, week_col=week_col)

    if item_name not in prepared.columns:
        st.error(f"'{item_name}' 컬럼이 없습니다.")
        return

    chart_df = prepared[[week_col, item_name]].copy()
    chart_df = chart_df.dropna(subset=[item_name])

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=chart_df[week_col],
            y=chart_df[item_name],
            mode="lines+markers",
            name=item_name
        )
    )

    fig.update_layout(
        title=f"{item_name} 주차별 매출 추이",
        xaxis_title="연도/주",
        yaxis_title="매출",
        height=500,
    )

    st.plotly_chart(fig, use_container_width=True)


# =========================
# 12. Streamlit UI
# =========================
st.title("아이템 매출 추이 형태 분류")

st.caption("각 아이템의 주차별 매출 추이를 읽어서 GPT API로 형태를 분류합니다.")

try:
    raw_df = load_sheet_data()
    st.success("시트 데이터 로딩 완료")

    with st.expander("원본 데이터 미리보기"):
        st.dataframe(raw_df.head(20), use_container_width=True)

    prepared_df = prepare_item_timeseries(raw_df)
    item_list = get_item_columns(prepared_df)

    selected_item = st.selectbox("개별 차트 확인할 아이템", item_list)
    draw_item_chart(raw_df, selected_item)

    if st.button("전체 아이템 형태 분류 실행"):
        result_df = run_shape_classification(raw_df)

        st.subheader("분류 결과")
        st.dataframe(result_df, use_container_width=True)

        csv = result_df.to_csv(index=False).encode("utf-8-sig")
        st.download_button(
            label="결과 CSV 다운로드",
            data=csv,
            file_name="item_shape_classification.csv",
            mime="text/csv"
        )

except Exception as e:
    st.error(f"오류 발생: {str(e)}")
