
import re
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st
from supabase import create_client


SUPABASE_URL = st.secrets["SUPABASE_URL"]
SUPABASE_KEY = st.secrets["SUPABASE_KEY"]
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

# 회전 출고(잉여 재고) 판단: PLC(주) = 기초재고÷주간예측이 이 값을 넘는 매장에서 잉여 계산. 화면에서 변경하지 않음.
DEFAULT_PLC_WEEKS: float = 4.0

# True면 is_forecast=true 행만 사용. 화면에서 변경하지 않음.
FORECAST_ROWS_ONLY: bool = True

# 리오더 제안 수량: (해당 SKU 주간 예측 합) × 이 주수. 웹에서 선택하지 않으며 코드만 수정.
DEFAULT_REORDER_SALES_WEEKS: float = 4.0


# =========================
# 공통 유틸
# =========================
def first_existing_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    if df is None or df.empty:
        return None
    for c in candidates:
        if c in df.columns:
            return c
    lower_map = {str(x).lower(): x for x in df.columns}
    for c in candidates:
        if c.lower() in lower_map:
            return lower_map[c.lower()]
    return None


def clean_number(value) -> float:
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


def to_int_safe(x, default: int = 0) -> int:
    v = clean_number(x)
    if pd.isna(v):
        return default
    return int(round(v))


# 예: "26년 12월 5주차", "2026-W03"
_YW_KO_RE = re.compile(r"^\s*(\d+)\s*년\s*(\d+)\s*월\s*(\d+)\s*주차\s*$", re.UNICODE)
_YW_ISO_RE = re.compile(r"^\s*(\d{4})\s*[-]?\s*[Ww]?\s*(\d{1,2})\s*$")


def year_week_sort_key(label: str) -> Tuple:
    """주차 라벨 정렬용 키 (최신 우선 = reverse=True)."""
    s = str(label).strip()
    m = _YW_KO_RE.match(s)
    if m:
        yy, mo, wk = int(m.group(1)), int(m.group(2)), int(m.group(3))
        cal_y = 2000 + yy if yy < 100 else yy
        return (0, cal_y, mo, wk)
    m2 = _YW_ISO_RE.match(s)
    if m2:
        return (1, int(m2.group(1)), int(m2.group(2)), 0)
    return (2, s)


def sort_year_week_labels(labels: List[str]) -> List[str]:
    seen = set()
    uniq: List[str] = []
    for x in labels:
        t = str(x).strip()
        if not t or t in seen:
            continue
        seen.add(t)
        uniq.append(t)
    return sorted(uniq, key=year_week_sort_key, reverse=True)


def diagnose_sku_weekly_forecast_unusable(weekly_df: pd.DataFrame) -> str:
    """
    normalize_weekly_slice 결과가 비었을 때, 테이블·원인을 문장으로 설명.
    """
    t = "`public.sku_weekly_forecast`"
    if weekly_df.empty:
        return (
            f"{t}에서 **조회된 행이 없습니다**. "
            "테이블에 데이터가 있는지 확인하세요."
        )
    sku_c = first_existing_col(weekly_df, ["sku", "SKU"])
    yw_c = first_existing_col(weekly_df, ["year_week", "yearweek"])
    missing_cols: List[str] = []
    if not sku_c:
        missing_cols.append("`sku`")
    if not yw_c:
        missing_cols.append("`year_week`")
    if missing_cols:
        cols = ", ".join(weekly_df.columns.astype(str).tolist())
        return (
            f"{t}에 필요한 컬럼이 없습니다: {', '.join(missing_cols)}. "
            f"(현재 컬럼: {cols})"
        )
    sk = weekly_df[sku_c].astype(str).str.strip()
    yw = weekly_df[yw_c].astype(str).str.strip()
    if (sk == "").all() and len(weekly_df) > 0:
        return f"{t}에 **`sku` 값이 비어 있는 행만** 있어 계산할 수 없습니다."
    if (yw == "").all() and len(weekly_df) > 0:
        return f"{t}에 **`year_week` 값이 비어 있는 행만** 있어 계산할 수 없습니다."
    return (
        f"{t} 데이터를 읽었으나 **`sku`·`year_week`가 모두 유효한 행이 없습니다**. "
        "값에 공백만 있는지 확인하세요."
    )


def diagnose_center_stock_columns(center_df: pd.DataFrame) -> Optional[str]:
    if center_df.empty:
        return "`public.center_stock`에 **행이 없습니다** — 물류센터 재고는 **0**으로 계산됩니다."
    sku_c = first_existing_col(center_df, ["sku", "SKU"])
    qty_c = first_existing_col(center_df, ["stock_qty", "stockqty", "qty"])
    miss: List[str] = []
    if not sku_c:
        miss.append("`sku`")
    if not qty_c:
        miss.append("`stock_qty`")
    if not miss:
        return None
    return f"`public.center_stock`에 없는 컬럼: {', '.join(miss)} → 물류센터 재고는 **0**으로 계산됩니다."


def diagnose_reorder_columns(reorder_df: pd.DataFrame) -> Optional[str]:
    if reorder_df.empty:
        return "`public.reorder`에 **행이 없습니다** — MOQ·리드타임은 반영하지 않습니다."
    sku_c = first_existing_col(reorder_df, ["sku", "SKU"])
    if sku_c:
        return None
    return "`public.reorder`에 `sku` 컬럼이 없어 MOQ·리드타임을 쓸 수 없습니다."


def load_supabase_table(table_name: str, page_size: int = 1000) -> pd.DataFrame:
    all_rows = []
    start = 0
    while True:
        end = start + page_size - 1
        res = supabase.table(table_name).select("*").range(start, end).execute()
        rows = res.data or []
        if not rows:
            break
        all_rows.extend(rows)
        if len(rows) < page_size:
            break
        start += page_size
    return pd.DataFrame(all_rows)


@st.cache_data(ttl=300)
def load_sku_weekly_forecast_df() -> pd.DataFrame:
    return load_supabase_table("sku_weekly_forecast")


@st.cache_data(ttl=300)
def load_center_stock_df() -> pd.DataFrame:
    return load_supabase_table("center_stock")


@st.cache_data(ttl=300)
def load_reorder_df() -> pd.DataFrame:
    return load_supabase_table("reorder")


def normalize_weekly_slice(weekly_df: pd.DataFrame) -> pd.DataFrame:
    """
    sku_weekly_forecast → 표준 컬럼: sku, store, year_week, demand_w, begin_stock,
    is_forecast, sku_name, created_at (테이블의 모든 행 사용, 배치별 필터 없음)
    """
    if weekly_df.empty:
        return pd.DataFrame()

    sku_c = first_existing_col(weekly_df, ["sku", "SKU"])
    store_c = first_existing_col(
        weekly_df, ["store_name", "storename", "매장", "plant", "PLANT"]
    )
    yw_c = first_existing_col(weekly_df, ["year_week", "yearweek"])
    dem_c = first_existing_col(
        weekly_df, ["sale_qty", "forecast_qty", "forecastqty", "saleqty"]
    )
    st_c = first_existing_col(weekly_df, ["begin_stock", "beginstock", "stock_qty"])
    fc_c = first_existing_col(weekly_df, ["is_forecast", "isforecast"])
    name_c = first_existing_col(weekly_df, ["sku_name", "skuname", "SKU_NAME"])
    style_c = first_existing_col(
        weekly_df, ["style_code", "stylecode", "style", "STYLE_CODE", "style_cd"]
    )
    ca_c = first_existing_col(weekly_df, ["created_at", "createdat"])
    fr_c = first_existing_col(weekly_df, ["forecast_run_id", "forecast_runid", "id"])

    # 수요 컬럼: 없으면 0으로 두고 진행 (year_week·sku는 있으나 sale_qty 누락인 경우)
    if not sku_c or not yw_c:
        return pd.DataFrame()

    out = weekly_df.copy()
    out["_sku"] = out[sku_c].astype(str).str.strip()
    out["_store"] = (
        out[store_c].astype(str).str.strip()
        if store_c
        else pd.Series("_unknown", index=out.index)
    )
    out["_year_week"] = out[yw_c].astype(str).str.strip()
    if dem_c:
        out["_demand"] = out[dem_c].apply(clean_number)
    else:
        out["_demand"] = 0.0
    out["_stock"] = out[st_c].apply(to_int_safe) if st_c else 0
    if fc_c:
        out["_is_fc"] = out[fc_c].apply(
            lambda x: bool(x) if pd.notna(x) else True
        )
    else:
        out["_is_fc"] = True
    out["_sku_name"] = (
        out[name_c].astype(str).str.strip()
        if name_c
        else out["_sku"]
    )
    out["_style_code"] = (
        out[style_c].astype(str).str.strip()
        if style_c
        else pd.Series("", index=out.index)
    )
    if ca_c:
        out["_created"] = pd.to_datetime(out[ca_c], errors="coerce")
    else:
        out["_created"] = pd.NaT
    if fr_c:
        out["_frid"] = out[fr_c]
    else:
        out["_frid"] = np.nan

    out = out[out["_sku"] != ""].copy()
    out = out[out["_year_week"] != ""].copy()
    return out


def dedupe_weekly_latest(df: pd.DataFrame) -> pd.DataFrame:
    """동일 (sku, store, year_week) 최신 행만."""
    if df.empty:
        return df
    df = df.sort_values(["_sku", "_store", "_year_week", "_created"], na_position="last")
    return df.drop_duplicates(subset=["_sku", "_store", "_year_week"], keep="last")


def center_stock_by_sku(center_df: pd.DataFrame) -> pd.Series:
    if center_df.empty:
        return pd.Series(dtype=float)
    sku_c = first_existing_col(center_df, ["sku", "SKU"])
    qty_c = first_existing_col(center_df, ["stock_qty", "stockqty", "qty"])
    if not sku_c or not qty_c:
        return pd.Series(dtype=float)
    sku_key = center_df[sku_c].astype(str).str.strip()
    g = center_df.groupby(sku_key, dropna=False)[qty_c].apply(
        lambda s: sum(to_int_safe(x) for x in s)
    )
    return g


def reorder_lt_moq_for_skus(
    skus: List[str],
    moq_df: pd.DataFrame,
    summary: pd.DataFrame,
) -> Tuple[int, int]:
    """스타일 등 여러 SKU에 대해 리드타임·MOQ는 각 SKU별 값 중 최댓값(보수적)."""
    lts: List[int] = []
    moqs: List[int] = []
    for s in skus:
        s = str(s).strip()
        if not s:
            continue
        got = False
        if not moq_df.empty and s in moq_df["sku"].astype(str).values:
            m = moq_df[moq_df["sku"].astype(str) == s].iloc[-1]
            lts.append(int(m["lead_time_days"]))
            moqs.append(int(m["minimum_capacity"]))
            got = True
        if not got and not summary.empty:
            gr = summary[summary["sku"].astype(str) == s]
            if not gr.empty:
                lts.append(int(gr["리드타임_일"].iloc[0]))
                moqs.append(int(gr["MOQ(참고)"].iloc[0]))
    return (max(lts) if lts else 0, max(moqs) if moqs else 0)


def reorder_guidance_for_store_slice(
    sl: pd.DataFrame,
    center_by_sku: pd.Series,
    lead_time_days: int,
    minimum_capacity: int,
    sales_weeks: float,
    ref_date: Optional[pd.Timestamp] = None,
    scope_prefix: str = "",
) -> str:
    """
    store_level에서 이미 필터된 행(sl)만으로 리오더 문구 생성.
    (단일 SKU 또는 동일 style_code에 속한 여러 SKU 행 합산.)
    scope_prefix: 문장 앞에 붙는 HTML 조각(예: 스타일 표시).
    """
    if sl.empty:
        return (
            f"{scope_prefix}해당 조건의 매장 행이 없어 리오더 안내를 표시할 수 없습니다."
        )

    weekly_total = float(sl["주간예측수요"].sum())
    store_stock_sum = int(sl["기초재고"].sum())
    center_qty = 0
    if isinstance(center_by_sku, pd.Series):
        for sku_key in sl["sku"].astype(str).unique():
            sku_key = str(sku_key).strip()
            if sku_key and sku_key in center_by_sku.index:
                center_qty += int(to_int_safe(center_by_sku.loc[sku_key]))
    total_inv = store_stock_sum + center_qty

    lt = max(0, int(lead_time_days))
    moq = max(0, int(minimum_capacity))
    sw = float(sales_weeks)
    if sw <= 0:
        sw = float(DEFAULT_REORDER_SALES_WEEKS)

    order_qty_raw = int(round(weekly_total * sw)) if weekly_total > 0 else 0
    order_qty = max(order_qty_raw, moq) if moq > 0 else order_qty_raw

    if ref_date is None:
        ref_date = pd.Timestamp.now()
    try:
        base = pd.Timestamp(ref_date).normalize()
    except Exception:
        base = pd.Timestamp.now().normalize()

    eps = 1e-6
    if weekly_total <= eps:
        sw_txt = str(int(sw)) if abs(sw - round(sw)) < 1e-9 else str(sw)
        return (
            f"{scope_prefix}리오더 리드타임 <strong>{lt}일</strong> 기준, 주간 수요 합이 <strong>0</strong>이라 "
            f"결품 방지 발주 기한을 산출할 수 없습니다. "
            f"(최소발주수량: <strong>{moq}장</strong>, <strong>{sw_txt}주</strong> 판매량 기준 발주 시 제안 수량: "
            f"<strong>{order_qty}장</strong>)."
        )

    cover_days = 7.0 * float(total_inv) / weekly_total
    days_until_must_order = cover_days - float(lt)
    deadline = base + pd.Timedelta(days=int(np.floor(max(0.0, days_until_must_order))))

    mo = int(deadline.month)
    dd = int(deadline.day)
    sw_disp = int(sw) if abs(sw - round(sw)) < 1e-9 else sw
    sw_txt = str(sw_disp) if isinstance(sw_disp, int) else str(sw_disp)

    return (
        f"{scope_prefix}리오더 리드타임 <strong>{lt}일</strong> 기준, <strong>{mo}월 {dd}일</strong> 이내 "
        f"<strong>{order_qty}장</strong> 리오더 발주 필요합니다 "
        f"(최소발주수량: <strong>{moq}장</strong>, <strong>{sw_txt}주</strong> 판매량 기준 발주)."
    )


def reorder_params_by_sku(reorder_df: pd.DataFrame) -> pd.DataFrame:
    if reorder_df.empty:
        return pd.DataFrame(columns=["sku", "lead_time_days", "minimum_capacity"])
    sku_c = first_existing_col(reorder_df, ["sku", "SKU"])
    lt_c = first_existing_col(reorder_df, ["lead_time", "leadtime"])
    moq_c = first_existing_col(reorder_df, ["minimum_capacity", "minimumcapacity", "moq"])
    if not sku_c:
        return pd.DataFrame(columns=["sku", "lead_time_days", "minimum_capacity"])
    r = reorder_df.copy()
    r["_sku"] = r[sku_c].astype(str).str.strip()
    r["_lt"] = r[lt_c].apply(to_int_safe) if lt_c else 0
    r["_moq"] = r[moq_c].apply(to_int_safe) if moq_c else 0
    r = r.sort_values("_sku").drop_duplicates(subset=["_sku"], keep="last")
    return r[["_sku", "_lt", "_moq"]].rename(
        columns={"_sku": "sku", "_lt": "lead_time_days", "_moq": "minimum_capacity"}
    )


def compute_store_rows_for_week(
    slice_norm: pd.DataFrame,
    year_week: str,
    plc_weeks_threshold: float,
) -> pd.DataFrame:
    """
    한 주차 기준 매장별: 수요, 재고, PLC(주), 역할, 결품부족분, 회전가능 잉여.
    """
    w = slice_norm[slice_norm["_year_week"] == str(year_week).strip()].copy()
    if w.empty:
        return pd.DataFrame()

    rows = []
    for _, r in w.iterrows():
        sku = r["_sku"]
        store = r["_store"]
        d = float(r["_demand"]) if pd.notna(r["_demand"]) else 0.0
        stock = int(r["_stock"])
        sku_name = r["_sku_name"]

        eps = 1e-6
        weekly_sales = max(d, 0.0)
        plc = (stock / weekly_sales) if weekly_sales > eps else (np.inf if stock > 0 else 0.0)

        shortage = weekly_sales > stock + eps
        deficit = max(0.0, weekly_sales - stock) if shortage else 0.0

        is_source = weekly_sales > eps and plc > plc_weeks_threshold
        excess_transfer = max(0.0, stock - plc_weeks_threshold * weekly_sales) if is_source else 0.0

        if shortage:
            role = "결품위험"
        elif is_source:
            role = "회전출고(판매부진)"
        else:
            role = "정상"

        style_val = str(r["_style_code"]).strip() if "_style_code" in r.index else ""
        rows.append(
            {
                "sku": sku,
                "sku_name": sku_name,
                "style_code": style_val,
                "매장": store,
                "주차": year_week,
                "주간예측수요": round(weekly_sales, 2),
                "기초재고": stock,
                "PLC_주": round(float(plc), 2) if np.isfinite(plc) else None,
                "역할": role,
                "결품부족분": int(round(deficit)),
                "회전가능잉여": int(round(excess_transfer)),
            }
        )
    return pd.DataFrame(rows)


def aggregate_sku_summary(store_df: pd.DataFrame, center_by_sku: pd.Series, moq_df: pd.DataFrame) -> pd.DataFrame:
    if store_df.empty:
        return pd.DataFrame()

    moq_map = (
        moq_df.set_index("sku")["minimum_capacity"].to_dict()
        if not moq_df.empty and "sku" in moq_df.columns
        else {}
    )
    lt_map = (
        moq_df.set_index("sku")["lead_time_days"].to_dict()
        if not moq_df.empty and "sku" in moq_df.columns
        else {}
    )

    parts = []
    for sku, g in store_df.groupby("sku"):
        total_deficit = int(g["결품부족분"].sum())
        total_rotation = int(g["회전가능잉여"].sum())
        n_short = int((g["역할"] == "결품위험").sum())
        n_source = int((g["역할"] == "회전출고(판매부진)").sum())
        if isinstance(center_by_sku, pd.Series) and sku in center_by_sku.index:
            center_qty = int(to_int_safe(center_by_sku.loc[sku]))
        else:
            center_qty = 0

        after_center = max(0, total_deficit - center_qty)
        pool_use = min(total_rotation, after_center)
        after_all = max(0, after_center - pool_use)

        moq = int(moq_map.get(sku, 0))
        suggested_order = max(after_all, moq) if after_all > 0 and moq > 0 else after_all

        parts.append(
            {
                "sku": sku,
                "sku_name": g["sku_name"].iloc[0],
                "결품위험_매장수": n_short,
                "회전출고_매장수": n_source,
                "매장결품부족_합": total_deficit,
                "물류센터_재고": center_qty,
                "회전가능잉여_합": total_rotation,
                "회전으로_충당(상한)": int(pool_use),
                "물류+회전_반영_추가발주": int(after_all),
                "MOQ(참고)": moq,
                "MOQ반영_제안발주": int(suggested_order),
                "리드타임_일": int(lt_map.get(sku, 0)),
            }
        )
    out = pd.DataFrame(parts).sort_values("물류+회전_반영_추가발주", ascending=False)
    return out.reset_index(drop=True)


def inject_theme_css():
    st.markdown(
        """
        <style>
        .block-container { padding-top: 1.2rem; padding-bottom: 2rem; max-width: 1400px; }
        div[data-testid="stMetric"] {
            background: linear-gradient(145deg, #0f172a 0%, #1e293b 100%);
            border: 1px solid #334155;
            border-radius: 12px;
            padding: 14px 16px;
        }
        div[data-testid="stMetric"] label { color: #94a3b8 !important; }
        div[data-testid="stMetric"] [data-testid="stMetricValue"] { color: #f8fafc !important; }
        h1 { font-weight: 700; letter-spacing: -0.02em; }
        .hl-short { color: #f87171; font-weight: 600; }
        .hl-ok { color: #34d399; }
        .reorder-headline {
            font-size: 1.2rem;
            font-weight: 600;
            line-height: 1.45;
            margin: 0.35rem 0 0.5rem 0;
            padding: 12px 14px;
            background: linear-gradient(145deg, #1e293b 0%, #0f172a 100%);
            border: 1px solid #334155;
            border-radius: 10px;
            color: #e2e8f0;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def main():
    st.set_page_config(
        page_title="결품·발주 취합",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    inject_theme_css()

    st.title("결품 예측 · 매장 회전 · 물류센터 반영 발주")
    reorder_headline_ph = st.empty()

    weekly_raw = pd.DataFrame()
    center_df = pd.DataFrame()
    reorder_df = pd.DataFrame()
    for label, loader, tname in [
        ("sku_weekly_forecast", load_sku_weekly_forecast_df, "public.sku_weekly_forecast"),
        ("center_stock", load_center_stock_df, "public.center_stock"),
        ("reorder", load_reorder_df, "public.reorder"),
    ]:
        try:
            df = loader()
        except Exception as e:
            st.error(f"`{tname}` 테이블을 불러오지 못했습니다: {e}")
            return
        if label == "sku_weekly_forecast":
            weekly_raw = df
        elif label == "center_stock":
            center_df = df
        else:
            reorder_df = df

    if weekly_raw.empty:
        st.warning("`public.sku_weekly_forecast` 테이블에 **데이터 행이 없습니다**.")
        return

    norm = normalize_weekly_slice(weekly_raw.copy())
    if norm.empty:
        st.warning(diagnose_sku_weekly_forecast_unusable(weekly_raw))
        return

    norm_before_fc = norm.copy()
    if FORECAST_ROWS_ONLY:
        norm = norm_before_fc[norm_before_fc["_is_fc"] == True].copy()  # noqa: E712
        if norm.empty and not norm_before_fc.empty:
            st.warning(
                "`public.sku_weekly_forecast`에 **`is_forecast` = true** 인 행이 없어 필터 결과가 비었습니다. "
                "코드 상수 `FORECAST_ROWS_ONLY`를 `False`로 두거나 DB의 `is_forecast` 값을 확인하세요."
            )
            norm = norm_before_fc.copy()

    norm = dedupe_weekly_latest(norm)

    yw_list = sort_year_week_labels(norm["_year_week"].tolist())
    if not yw_list:
        st.warning(
            "`public.sku_weekly_forecast`에서 유효한 **`year_week`** 값을 찾지 못했습니다. "
            "(`year_week` 컬럼이 모두 비어 있을 수 있습니다.)"
        )
        return

    # 주차 선택 없음: 데이터에 있는 year_week 중 정렬상 가장 최신 1개만 사용 (발주 시점 산출용 단일 스냅샷)
    year_week = yw_list[0]
    plc_thr = float(DEFAULT_PLC_WEEKS)

    center_by_sku = center_stock_by_sku(center_df)
    moq_df = reorder_params_by_sku(reorder_df)

    c_warn = diagnose_center_stock_columns(center_df)
    if c_warn:
        st.info(c_warn)
    r_warn = diagnose_reorder_columns(reorder_df)
    if r_warn:
        st.info(r_warn)

    # 취합·회전·물류 반영·리오더 문구: 항상 전체 매장 데이터로 계산 (매장 조회 필터는 아래 상세 표에만 적용)
    store_level = compute_store_rows_for_week(norm, year_week, plc_thr)
    if store_level.empty:
        st.warning(
            f"`public.sku_weekly_forecast`에 자동 선택한 주차 **`{year_week}`** 와 일치하는 **`year_week`** 행이 없습니다. "
            "데이터의 `year_week`·필터 조건을 확인하세요."
        )
        return

    summary = aggregate_sku_summary(store_level, center_by_sku, moq_df)

    st.sidebar.markdown("### 리오더 (style_code)")
    sku_opts_headline = summary["sku"].tolist()
    if not sku_opts_headline:
        sku_opts_headline = sorted(store_level["sku"].unique().tolist())

    has_style = (
        "style_code" in store_level.columns
        and (store_level["style_code"].astype(str).str.strip() != "").any()
    )

    scope_prefix = ""
    sl_headline = pd.DataFrame()
    _lt, _moq = 0, 0
    pick = str(sku_opts_headline[0]) if sku_opts_headline else ""

    if has_style:
        style_opts = sorted(
            {str(x).strip() for x in store_level["style_code"].tolist() if str(x).strip() != ""}
        )
        guidance_style = st.sidebar.selectbox(
            "style_code",
            options=style_opts,
            index=0,
        )
        sl_headline = store_level[
            store_level["style_code"].astype(str) == str(guidance_style)
        ].copy()
        sk_list = sorted(sl_headline["sku"].astype(str).unique().tolist())
        _lt, _moq = reorder_lt_moq_for_skus(sk_list, moq_df, summary)
        scope_prefix = f"[style_code <strong>{guidance_style}</strong>] "
        pick = str(sk_list[0]) if sk_list else (str(sku_opts_headline[0]) if sku_opts_headline else "")
    else:
        st.sidebar.caption(
            "`style_code`가 없으면 이 항목을 고를 수 없습니다. 리오더·하단 상세는 첫 SKU로 자동 표시합니다."
        )
        if pick:
            sl_headline = store_level[store_level["sku"].astype(str) == pick].copy()
            g_row = summary[summary["sku"] == pick]
            if not g_row.empty:
                _lt = int(g_row["리드타임_일"].iloc[0])
                _moq = int(g_row["MOQ(참고)"].iloc[0])
            elif not moq_df.empty and pick in moq_df["sku"].astype(str).values:
                m = moq_df[moq_df["sku"].astype(str) == str(pick)].iloc[-1]
                _lt = int(m["lead_time_days"])
                _moq = int(m["minimum_capacity"])

    if st.sidebar.button("데이터 새로고침"):
        st.cache_data.clear()
        st.rerun()

    headline_txt = reorder_guidance_for_store_slice(
        sl_headline,
        center_by_sku,
        _lt,
        _moq,
        float(DEFAULT_REORDER_SALES_WEEKS),
        scope_prefix=scope_prefix,
    )
    reorder_headline_ph.markdown(
        f'<div class="reorder-headline">{headline_txt}</div>',
        unsafe_allow_html=True,
    )

    total_extra = int(summary["물류+회전_반영_추가발주"].sum())
    total_short_stores = int(store_level[store_level["역할"] == "결품위험"]["매장"].nunique())
    skus_at_risk = int((summary["결품위험_매장수"] > 0).sum())

    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric("추가 발주 필요 SKU 수", f"{skus_at_risk}")
    with m2:
        st.metric("결품 위험 매장·SKU 건수", f"{len(store_level[store_level['역할'] == '결품위험'])}")
    with m3:
        st.metric("고유 결품위험 매장 수", f"{total_short_stores}")
    with m4:
        st.metric("전체 SKU 추가발주(추정)", f"{total_extra:,}")

    st.markdown("---")
    st.subheader("SKU별 취합")
    st.caption(
        "매장결품부족_합: 결품 위험 매장의 max(예측−기초재고) 합. "
        "회전가능잉여_합: PLC 기준 초과 재고. "
        "물류+회전_반영_추가발주 = max(0, 부족합−물류센터) − min(회전잉여, 그 잔여)."
    )

    display_cols = [
        "sku",
        "sku_name",
        "결품위험_매장수",
        "회전출고_매장수",
        "매장결품부족_합",
        "물류센터_재고",
        "회전가능잉여_합",
        "회전으로_충당(상한)",
        "물류+회전_반영_추가발주",
        "MOQ(참고)",
        "MOQ반영_제안발주",
        "리드타임_일",
    ]
    st.dataframe(
        summary[display_cols],
        use_container_width=True,
        hide_index=True,
        column_config={
            "sku": st.column_config.TextColumn("SKU"),
            "sku_name": st.column_config.TextColumn("상품명"),
            "매장결품부족_합": st.column_config.NumberColumn("매장 부족 합", format="%d"),
            "물류센터_재고": st.column_config.NumberColumn("물류센터", format="%d"),
            "회전가능잉여_합": st.column_config.NumberColumn("회전 잉여 합", format="%d"),
            "물류+회전_반영_추가발주": st.column_config.NumberColumn("추가 발주", format="%d"),
            "MOQ반영_제안발주": st.column_config.NumberColumn("MOQ 반영 제안", format="%d"),
        },
    )

    st.markdown("---")
    st.subheader("매장별 상세 (드릴다운)")
    st.caption(
        "**매장** 선택은 이 표만 좁혀 보는 용도입니다. "
        "SKU 취합·메트릭·리오더 안내·회전/물류 로직은 **항상 전체 매장** 기준입니다."
    )

    _store_names = sorted(store_level["매장"].astype(str).unique().tolist())
    view_store = st.selectbox(
        "매장 (상세 표 조회만)",
        options=["전체"] + _store_names,
        index=0,
    )

    role_filter = st.multiselect(
        "역할 필터",
        options=["결품위험", "회전출고(판매부진)", "정상"],
        default=["결품위험", "회전출고(판매부진)"],
    )

    det = store_level[store_level["sku"] == pick].copy()
    if role_filter:
        det = det[det["역할"].isin(role_filter)].copy()

    view_det = det.copy()
    if view_store != "전체":
        view_det = view_det[view_det["매장"].astype(str) == str(view_store)].copy()

    det_display = view_det[
        [
            "매장",
            "주간예측수요",
            "기초재고",
            "PLC_주",
            "역할",
            "결품부족분",
            "회전가능잉여",
        ]
    ].sort_values(["역할", "결품부족분"], ascending=[True, False])

    st.dataframe(
        det_display,
        use_container_width=True,
        hide_index=True,
        column_config={
            "주간예측수요": st.column_config.NumberColumn("주간 예측", format="%.2f"),
            "PLC_주": st.column_config.NumberColumn("PLC(주)", format="%.2f"),
            "결품부족분": st.column_config.NumberColumn("결품 부족", format="%d"),
            "회전가능잉여": st.column_config.NumberColumn("회전 잉여", format="%d"),
        },
    )

    sub_sum = summary[summary["sku"] == pick]
    if not sub_sum.empty:
        row = sub_sum.iloc[0]
        st.markdown(
            f"**선택 SKU 요약** — 부족 합: **{int(row['매장결품부족_합']):,}**, "
            f"물류센터: **{int(row['물류센터_재고']):,}**, "
            f"회전 잉여 합: **{int(row['회전가능잉여_합']):,}**, "
            f"<span class='hl-short'>추가 발주(추정): **{int(row['물류+회전_반영_추가발주']):,}**</span>",
            unsafe_allow_html=True,
        )
    elif pick:
        st.info(
            "선택한 SKU가 `public.sku_weekly_forecast` 기준 상단 취합에 없을 수 있습니다. "
            "아래는 동일 테이블의 매장별 행입니다."
        )


if __name__ == "__main__":
    main()
