
import re
from typing import Any, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st
from supabase import create_client


SUPABASE_URL = st.secrets["SUPABASE_URL"]
SUPABASE_KEY = st.secrets["SUPABASE_KEY"]
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)


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
            "사이드바의 예측 배치 필터를 끄거나, 해당 배치에 맞는 데이터가 있는지 확인하세요."
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


def infer_run_batch_key(runs_df: pd.DataFrame) -> str:
    fr = first_existing_col(runs_df, ["forecast_run_id", "forecast_runid"])
    if fr and runs_df[fr].notna().any():
        return fr
    return "id"


def weekly_batch_key_col(weekly_df: pd.DataFrame) -> Optional[str]:
    """sku_weekly_forecast ↔ sku_forecast_run 연결 컬럼 (forecast_run_id 없으면 id)."""
    return first_existing_col(weekly_df, ["forecast_run_id", "forecast_runid", "id"])


def sortable_run_time(ts) -> float:
    """
    배치 정렬용 시각 → 비교 가능한 float (ns).
    tz-aware / naive 혼합·datetime 혼합 시 Timestamp 직접 비교 TypeError 방지.
    """
    if ts is None:
        return 0.0
    try:
        if pd.isna(ts):
            return 0.0
    except (TypeError, ValueError):
        pass
    try:
        t = pd.Timestamp(ts)
    except Exception:
        return 0.0
    if pd.isna(t):
        return 0.0
    try:
        if t.tzinfo is not None:
            t = t.tz_convert("UTC").tz_localize(None)
    except Exception:
        pass
    return float(t.value)


def list_run_batches(runs_df: pd.DataFrame, weekly_df: pd.DataFrame) -> List[Tuple[Any, pd.Timestamp, int]]:
    wk_fr = weekly_batch_key_col(weekly_df)
    if weekly_df.empty or wk_fr is None:
        return []

    keys_in_weekly = weekly_df[wk_fr].dropna().astype(object).unique().tolist()
    parts: List[Tuple[Any, pd.Timestamp, int]] = []

    if runs_df.empty:
        for k in keys_in_weekly:
            n_w = int((weekly_df[wk_fr] == k).sum())
            parts.append((k, pd.Timestamp.now(), n_w))
        parts.sort(key=lambda x: x[2], reverse=True)
        return parts

    # 주간 테이블이 id로 묶이면 부모도 sku_forecast_run.id 로 매칭
    if wk_fr == "id" and "id" in runs_df.columns:
        parent_key = "id"
    else:
        parent_key = infer_run_batch_key(runs_df)
    rd_col = first_existing_col(runs_df, ["run_date", "rundate", "created_at"])
    if not rd_col:
        rd_col = runs_df.columns[0]

    for k in keys_in_weekly:
        sub = runs_df[runs_df[parent_key] == k]
        if sub.empty and parent_key != "id" and "id" in runs_df.columns:
            sub = runs_df[runs_df["id"] == k]
        if sub.empty:
            n_w = int((weekly_df[wk_fr] == k).sum())
            parts.append((k, pd.Timestamp(1970, 1, 1), n_w))
            continue
        try:
            rd = pd.to_datetime(sub[rd_col], errors="coerce").max()
        except Exception:
            rd = pd.NaT
        if pd.isna(rd):
            rd = pd.Timestamp(1970, 1, 1)
        n_w = int((weekly_df[wk_fr] == k).sum())
        parts.append((k, rd, n_w))

    parts.sort(key=lambda x: (sortable_run_time(x[1]), x[2]), reverse=True)
    return parts


def filter_by_run_key(df: pd.DataFrame, run_key_col: str, batch_key: object) -> pd.DataFrame:
    if df.empty or run_key_col not in df.columns:
        return pd.DataFrame()
    return df[df[run_key_col] == batch_key].copy()


def normalize_weekly_slice(weekly_df: pd.DataFrame) -> pd.DataFrame:
    """
    sku_weekly_forecast → 표준 컬럼: sku, store, year_week, demand_w, begin_stock,
    is_forecast, sku_name, created_at, 배치 키(id 또는 forecast_run_id, 선택)
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

        rows.append(
            {
                "sku": sku,
                "sku_name": sku_name,
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
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_data(ttl=300)
def load_sku_forecast_run_df() -> pd.DataFrame:
    return load_supabase_table("sku_forecast_run")


def main():
    st.set_page_config(
        page_title="결품·발주 취합",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    inject_theme_css()

    st.title("결품 예측 · 매장 회전 · 물류센터 반영 발주")
    st.caption(
        "선택 주차 기준으로 매장별 주간 예측 대비 기초재고를 비교합니다. "
        "PLC(재고÷주간예측)가 긴 매장은 회전 출고 가능 잉여로 보고, "
        "물류센터 재고와 합산해 **추가 발주** 추정치를 냅니다."
    )

    weekly_raw = pd.DataFrame()
    center_df = pd.DataFrame()
    reorder_df = pd.DataFrame()
    runs_df = pd.DataFrame()
    for label, loader, tname in [
        ("sku_weekly_forecast", load_sku_weekly_forecast_df, "public.sku_weekly_forecast"),
        ("center_stock", load_center_stock_df, "public.center_stock"),
        ("reorder", load_reorder_df, "public.reorder"),
        ("sku_forecast_run", load_sku_forecast_run_df, "public.sku_forecast_run"),
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
        elif label == "reorder":
            reorder_df = df
        else:
            runs_df = df

    if weekly_raw.empty:
        st.warning("`public.sku_weekly_forecast` 테이블에 **데이터 행이 없습니다**.")
        return

    wk_fr = weekly_batch_key_col(weekly_raw)
    weekly_filtered = weekly_raw.copy()

    st.sidebar.markdown("### 데이터 범위")
    use_batch = False
    selected_batch_key: Optional[Any] = None
    if wk_fr and weekly_raw[wk_fr].notna().any():
        batches = list_run_batches(runs_df, weekly_raw)
        if batches:
            batch_filter_label = (
                "예측 배치(id ↔ sku_forecast_run.id)로 필터"
                if wk_fr == "id"
                else "예측 배치(forecast_run_id)로 필터"
            )
            use_batch = st.sidebar.checkbox(batch_filter_label, value=True)
            if use_batch:
                batch_labels = {
                    str(k): f"{pd.Timestamp(rd).strftime('%Y-%m-%d %H:%M')} · batch={k} · {n}행"
                    for k, rd, n in batches
                }
                batch_keys_ordered = [b[0] for b in batches]
                selected_batch_str = st.sidebar.selectbox(
                    "실행 배치",
                    options=[batch_labels[str(k)] for k in batch_keys_ordered],
                    index=0,
                )
                inv_lbl = {v: k for k, v in batch_labels.items()}
                selected_batch_key = inv_lbl[selected_batch_str]
                try:
                    selected_batch_key = type(batch_keys_ordered[0])(selected_batch_key)
                except (ValueError, TypeError, IndexError):
                    pass
                weekly_filtered = filter_by_run_key(weekly_raw, wk_fr, selected_batch_key)

    norm = normalize_weekly_slice(weekly_filtered)
    if norm.empty:
        st.warning(diagnose_sku_weekly_forecast_unusable(weekly_filtered))
        return

    norm_before_fc = norm.copy()
    fc_only = st.sidebar.checkbox(
        "미래 예측 행만(is_forecast=true)",
        value=False,
        help="DB에 is_forecast=true 행이 없으면 켜 두면 주차 목록이 비어 보일 수 있습니다.",
    )
    if fc_only:
        norm = norm_before_fc[norm_before_fc["_is_fc"] == True].copy()  # noqa: E712
        if norm.empty and not norm_before_fc.empty:
            st.warning(
                "`public.sku_weekly_forecast`에 **`is_forecast` = true** 인 행이 없어 필터 결과가 비었습니다. "
                "위 옵션을 끄거나 DB의 `is_forecast` 값을 확인하세요."
            )
            norm = norm_before_fc.copy()

    norm = dedupe_weekly_latest(norm)

    yw_list = sort_year_week_labels(norm["_year_week"].tolist())
    if not yw_list:
        st.warning(
            "`public.sku_weekly_forecast`에서 유효한 **`year_week`** 값을 찾지 못했습니다. "
            "(배치 필터로 행이 0건이 되었거나, `year_week` 컬럼이 모두 비어 있을 수 있습니다.)"
        )
        return

    year_week = st.sidebar.selectbox("기준 주차 (year_week)", options=yw_list, index=0)
    plc_thr = st.sidebar.number_input(
        "회전 출고 판단 PLC(주) 기준",
        min_value=1.0,
        max_value=52.0,
        value=4.0,
        step=0.5,
        help="재고÷주간예측이 이 값보다 크면 판매 부진·회전 출고 후보로 잉여 수량을 계산합니다.",
    )

    if st.sidebar.button("데이터 새로고침"):
        st.cache_data.clear()
        st.rerun()

    center_by_sku = center_stock_by_sku(center_df)
    moq_df = reorder_params_by_sku(reorder_df)

    c_warn = diagnose_center_stock_columns(center_df)
    if c_warn:
        st.info(c_warn)
    r_warn = diagnose_reorder_columns(reorder_df)
    if r_warn:
        st.info(r_warn)

    store_level = compute_store_rows_for_week(norm, year_week, plc_thr)
    if store_level.empty:
        st.warning(
            f"`public.sku_weekly_forecast`에 선택한 주차 **`{year_week}`** 와 일치하는 **`year_week`** 행이 없습니다. "
            "사이드바에서 다른 주차를 선택하세요."
        )
        return

    summary = aggregate_sku_summary(store_level, center_by_sku, moq_df)

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

    sku_options = summary["sku"].tolist()
    if not sku_options:
        sku_options = sorted(store_level["sku"].unique().tolist())

    c1, c2 = st.columns([1, 2])
    with c1:
        pick = st.selectbox("SKU 선택", options=sku_options, index=0)
    with c2:
        role_filter = st.multiselect(
            "역할 필터",
            options=["결품위험", "회전출고(판매부진)", "정상"],
            default=["결품위험", "회전출고(판매부진)"],
        )

    det = store_level[store_level["sku"] == pick].copy()
    if role_filter:
        det = det[det["역할"].isin(role_filter)].copy()

    det_display = det[
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

    with st.expander("계산 로직 요약"):
        st.markdown(
            """
            1. **기준 주차** `year_week`의 `sale_qty`(또는 `forecast_qty`)를 주간 예측 수요로 사용합니다.
            2. **기초재고**는 `begin_stock`을 사용합니다.
            3. **결품 위험**: 예측 수요 > 기초재고 인 매장. 부족분 = 예측 − 재고.
            4. **PLC(주)** = 기초재고 ÷ 주간예측 (예측이 0이면 해석 제한).
            5. **회전 출고**: PLC > 기준주(기본 4주)인 매장에서, 잉여 = 재고 − 기준주×예측 (0 이상).
            6. **SKU 추가 발주** = 먼저 `center_stock`으로 부족을 충당한 뒤, 남은 부족에 대해 회전 잉여로 `min(회전 잉여, 남은 부족)` 만큼 차감합니다.
            7. `reorder.minimum_capacity`가 있으면 발주가 필요한 경우에만 max(추정, MOQ)를 **제안**으로 표시합니다.
            """
        )


if __name__ == "__main__":
    main()
