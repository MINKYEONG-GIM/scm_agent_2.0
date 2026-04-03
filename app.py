
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

# 리오더 제안 수량: (해당 SKU 주간 예측 합) × 이 주수. 웹에서 선택하지 않으며 코드만 수정.
DEFAULT_REORDER_SALES_WEEKS: float = 4.0

# 회전 출고 시 매장당 반드시 남겨 둘 최소 기초재고(장). 이보다 적게는 넘기지 않음.
MIN_STORE_RETAIN_QTY: int = 3


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


def sort_year_week_labels_asc(labels: List[str]) -> List[str]:
    """주차 라벨 정렬: 과거 → 최신 (스프레드시트 타임라인 행 순서)."""
    seen = set()
    uniq: List[str] = []
    for x in labels:
        t = str(x).strip()
        if not t or t in seen:
            continue
        seen.add(t)
        uniq.append(t)
    return sorted(uniq, key=year_week_sort_key, reverse=False)


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


def _is_forecast_to_bool(x) -> bool:
    """DB `is_forecast`: true=미래 예측 판매, false=실제 판매. 결측은 예측(true)로 두어 단일 행 스키마와 호환."""
    if pd.isna(x):
        return True
    if isinstance(x, (bool, np.bool_)):
        return bool(x)
    s = str(x).strip().lower()
    if s in ("false", "0", "no", "f", "n"):
        return False
    return True


def normalize_weekly_slice(weekly_df: pd.DataFrame) -> pd.DataFrame:
    """
    sku_weekly_forecast → 표준 컬럼. `sale_qty` + `is_forecast` 규약:
    false → 실제 판매(기판매), true → 미래 판매 예측. 동일 (sku, 매장, 주차)에 행이 둘 있을 수 있음.
    """
    if weekly_df.empty:
        return pd.DataFrame()

    sku_c = first_existing_col(weekly_df, ["sku", "SKU"])
    store_c = first_existing_col(
        weekly_df, ["store_name", "storename", "매장", "plant", "PLANT"]
    )
    yw_c = first_existing_col(weekly_df, ["year_week", "yearweek"])
    # 주간 수요(PLC·결품 등 계산): 기존과 동일한 후보 우선순위
    dem_c = first_existing_col(
        weekly_df, ["sale_qty", "forecast_qty", "forecastqty", "saleqty"]
    )
    # 화면용: 기판매(실적/기반) vs 예측 판매 — 컬럼이 있으면 각각 매핑
    fc_qty_c = first_existing_col(
        weekly_df, ["forecast_qty", "forecastqty", "week_forecast_qty", "pred_sale_qty"]
    )
    hist_c = first_existing_col(
        weekly_df,
        [
            "sold_qty",
            "actual_sale_qty",
            "hist_sale_qty",
            "base_sale_qty",
            "sale_qty",
            "saleqty",
        ],
    )
    if fc_qty_c and hist_c and fc_qty_c == hist_c:
        hist_c = first_existing_col(
            weekly_df,
            ["sold_qty", "actual_sale_qty", "hist_sale_qty", "base_sale_qty", "saleqty"],
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
    if fc_c:
        out["_is_fc"] = out[fc_c].apply(_is_forecast_to_bool)
    else:
        out["_is_fc"] = True

    dem_name = (str(dem_c).strip().lower() if dem_c else "") or ""
    use_sale_qty_by_flag = bool(fc_c) and dem_name in ("sale_qty", "saleqty")

    if dem_c:
        _raw_sale = out[dem_c].apply(clean_number)
    else:
        _raw_sale = pd.Series(0.0, index=out.index)

    if use_sale_qty_by_flag:
        fc_mask = out["_is_fc"].fillna(True).astype(bool)
        out["_hist_sale"] = _raw_sale.mask(fc_mask, np.nan)
        out["_fc_sale"] = _raw_sale.mask(~fc_mask, np.nan)
        out["_demand"] = _raw_sale
    else:
        if dem_c:
            out["_demand"] = _raw_sale
        else:
            out["_demand"] = 0.0
        if fc_qty_c:
            out["_fc_sale"] = out[fc_qty_c].apply(clean_number)
        else:
            out["_fc_sale"] = np.nan
        if hist_c:
            out["_hist_sale"] = out[hist_c].apply(clean_number)
        else:
            out["_hist_sale"] = np.nan
    out["_stock"] = out[st_c].apply(to_int_safe) if st_c else 0
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


def merge_weekly_actual_forecast(df: pd.DataFrame) -> pd.DataFrame:
    """
    (sku, 매장, 주차)당 1행으로 합침: 실적 행의 sale_qty→_hist_sale, 예측 행→_fc_sale.
    주간 수요(_demand)는 예측 값이 있으면 예측, 없으면 실적, 둘 다 없으면 0.
    기초재고는 예측 행이 있으면 그쪽 최신, 없으면 전체 최신.
    """
    if df.empty:
        return df
    rows: List[dict] = []
    for (_sku, _store, _yw), g in df.groupby(["_sku", "_store", "_year_week"], dropna=False):
        g = g.sort_values("_created", na_position="last")
        hv = pd.to_numeric(g["_hist_sale"], errors="coerce").dropna()
        fv = pd.to_numeric(g["_fc_sale"], errors="coerce").dropna()
        hist = float(hv.iloc[-1]) if len(hv) else np.nan
        fc = float(fv.iloc[-1]) if len(fv) else np.nan
        if pd.notna(fc):
            demand = float(fc)
        elif pd.notna(hist):
            demand = float(hist)
        else:
            dems = pd.to_numeric(g["_demand"], errors="coerce").dropna()
            demand = float(dems.iloc[-1]) if len(dems) else 0.0

        mask_fc = g["_is_fc"].fillna(True).astype(bool)
        gfc = g[mask_fc]
        if not gfc.empty:
            stock = int(to_int_safe(gfc.iloc[-1].get("_stock", 0)))
        else:
            stock = int(to_int_safe(g.iloc[-1].get("_stock", 0)))

        last = g.iloc[-1]
        style_val = ""
        if "_style_code" in last.index:
            style_val = str(last["_style_code"]).strip()
        rows.append(
            {
                "_sku": _sku,
                "_store": _store,
                "_year_week": _yw,
                "_hist_sale": hist,
                "_fc_sale": fc,
                "_demand": demand,
                "_stock": stock,
                "_is_fc": True,
                "_sku_name": last.get("_sku_name", _sku),
                "_style_code": style_val,
                "_created": last.get("_created", pd.NaT),
                "_frid": last.get("_frid", np.nan),
            }
        )
    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(["_sku", "_store", "_year_week"], na_position="last")
    return out.reset_index(drop=True)


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


def center_two_bucket_qty(center_df: pd.DataFrame, sku: str) -> Tuple[Tuple[str, int], Tuple[str, int]]:
    """
    와이드 표용 물류 재고 2열: (라벨1, 수량1), (라벨2, 수량2).
    창고/센터 컬럼이 있으면 재고 큰 순 1위 + 나머지 합(또는 2위), 없으면 전체→첫 열, 둘째는 0.
    """
    sku_key = str(sku).strip()
    sku_c = first_existing_col(center_df, ["sku", "SKU"])
    qty_c = first_existing_col(center_df, ["stock_qty", "stockqty", "qty"])
    if center_df.empty or not sku_key or not sku_c or not qty_c:
        return (("center 1", 0), ("center 2", 0))
    loc_c = first_existing_col(
        center_df,
        [
            "warehouse",
            "center_name",
            "dc_code",
            "dc",
            "center",
            "plant",
            "location",
            "물류센터",
            "창고",
        ],
    )
    csub = center_df[center_df[sku_c].astype(str).str.strip() == sku_key]
    if csub.empty:
        return (("center 1", 0), ("center 2", 0))

    def sum_qty(frame: pd.DataFrame) -> int:
        return int(sum(to_int_safe(x) for x in frame[qty_c]))

    if loc_c:
        g = csub.groupby(csub[loc_c].astype(str).str.strip(), dropna=False)[qty_c].apply(
            lambda s: sum(to_int_safe(x) for x in s)
        )
        items = sorted(
            ((str(name), int(to_int_safe(q))) for name, q in g.items()),
            key=lambda x: -x[1],
        )
        if len(items) >= 2:
            a, qa = items[0]
            rest = sum(q for _, q in items[1:])
            if len(items) == 2:
                b, qb = items[1]
                return ((a, qa), (b, qb))
            return ((a, qa), ("기타센터 합", rest))
        if len(items) == 1:
            return (items[0], ("center 2", 0))
    total = sum_qty(csub)
    return (("물류(전체)", total), ("—", 0))


def _sale_qty_from_norm_row(r: pd.Series) -> float:
    """표시용 판매량: 예측 전용(_fc_sale) 우선, 없으면 주간 수요(_demand)."""
    fc_raw = r["_fc_sale"] if "_fc_sale" in r.index else np.nan
    d = r["_demand"] if "_demand" in r.index else 0.0
    if pd.notna(fc_raw):
        return round(max(float(fc_raw), 0.0), 2)
    dd = float(d) if pd.notna(d) else 0.0
    return round(max(dd, 0.0), 2)


def _display_sale_one_value(r0: Optional[pd.Series]) -> Tuple[float, bool]:
    """
    매장 칸에 넣을 단일 판매 값과, 예측(빨간색) 여부.
    실적(_hist_sale)이 있으면 실적만 표시, 없으면 예측(_fc_sale) 표시=예측 스타일.
    """
    if r0 is None or (isinstance(r0, pd.Series) and r0.empty):
        return (0.0, False)
    hist_raw = r0["_hist_sale"] if "_hist_sale" in r0.index else np.nan
    fc_raw = r0["_fc_sale"] if "_fc_sale" in r0.index else np.nan
    if pd.notna(hist_raw):
        return (round(max(float(hist_raw), 0.0), 2), False)
    if pd.notna(fc_raw):
        return (round(max(float(fc_raw), 0.0), 2), True)
    return (0.0, False)


def style_week_store_wide(
    df: pd.DataFrame,
    forecast_mask: pd.DataFrame,
    sale_substr: str = " · 판매",
):
    """예측 판매 열만 빨간색(rose) 강조."""
    sale_cols = [c for c in df.columns if sale_substr in str(c)]
    inv_cols = [c for c in df.columns if "기초재고량" in str(c)]
    logistics_col = "물류재고(합)" if "물류재고(합)" in df.columns else None

    def _apply_row(row: pd.Series) -> List[str]:
        ri = int(row.name)
        styles: List[str] = []
        for c in row.index:
            if (
                c in sale_cols
                and not forecast_mask.empty
                and ri < len(forecast_mask)
                and c in forecast_mask.columns
            ):
                if bool(forecast_mask.iloc[ri][c]):
                    styles.append("color: #e11d48; font-weight: 600")
                    continue
            styles.append("")
        return styles

    styler = df.style.apply(_apply_row, axis=1)
    if sale_cols:
        styler = styler.format("{:.2f}", subset=sale_cols, na_rep="0.00")
    if inv_cols:
        styler = styler.format("{:.0f}", subset=inv_cols, na_rep="0")
    if logistics_col:
        styler = styler.format("{:.0f}", subset=[logistics_col], na_rep="0")
    return styler


def build_sku_week_wide_table(
    norm: pd.DataFrame,
    sku: str,
    center_df: pd.DataFrame,
    center_by_sku: pd.Series,
) -> Tuple[pd.DataFrame, pd.DataFrame, str]:
    """
    스프레드시트형: 행=주차(과거→최신), 열=물류재고(합) + 매장별 판매(단일 값) + 기초재고량.
    판매가 예쪽만 있을 때 forecast_mask 해당 열 True → 표에서 빨간색.
    """
    sku_key = str(sku).strip()
    empty_mask = pd.DataFrame()
    if norm.empty or not sku_key:
        return pd.DataFrame(), empty_mask, ""
    sub = norm[norm["_sku"].astype(str).str.strip() == sku_key].copy()
    if sub.empty:
        return pd.DataFrame(), empty_mask, ""

    if isinstance(center_by_sku, pd.Series) and sku_key in center_by_sku.index:
        center_qty_total = int(to_int_safe(center_by_sku.loc[sku_key]))
    else:
        center_qty_total = int(
            to_int_safe(center_stock_by_sku(center_df).get(sku_key, 0))
        )

    weeks = sort_year_week_labels_asc(sub["_year_week"].tolist())
    stores = sorted(
        {str(s).strip() for s in sub["_store"].tolist() if str(s).strip() and str(s).strip() != "_unknown"}
    )
    if not stores:
        stores = sorted({str(s).strip() for s in sub["_store"].tolist() if str(s).strip()})

    key_df = sub.sort_values("_created", na_position="last").drop_duplicates(
        subset=["_year_week", "_store"], keep="last"
    )

    rows_out: List[dict] = []
    mask_rows: List[dict] = []
    for yw in weeks:
        row: dict = {"주차": yw, "물류재고(합)": center_qty_total}
        mk: dict = {"주차": False, "물류재고(합)": False}
        sl = key_df[key_df["_year_week"].astype(str).str.strip() == str(yw).strip()]
        for st in stores:
            sale_col = f"{st} · 판매"
            inv_col = f"{st} · 기초재고량"
            m = sl[sl["_store"].astype(str).str.strip() == st]
            if m.empty:
                row[sale_col] = 0.0
                row[inv_col] = 0
                mk[sale_col] = False
                mk[inv_col] = False
            else:
                r0 = m.iloc[0]
                val, is_fc = _display_sale_one_value(r0)
                row[sale_col] = val
                row[inv_col] = int(to_int_safe(r0.get("_stock", 0)))
                mk[sale_col] = is_fc
                mk[inv_col] = False
        rows_out.append(row)
        mask_rows.append(mk)

    wide_df = pd.DataFrame(rows_out)
    fc_mask_df = pd.DataFrame(mask_rows)
    note = (
        "**물류재고(합)** 은 `center_stock`에서 해당 SKU **전 센터 재고 합**입니다(주차 무관·스냅샷). "
        "**판매** 는 실적이 있으면 실적만, 없으면 예측만 표시하며 **예측만 쓸 때 빨간색**입니다."
    )
    return wide_df, fc_mask_df, note


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
    회전 잉여는 PLC 초과분이어도 매장당 MIN_STORE_RETAIN_QTY 장은 남긴 뒤 넘길 수 있는 수량만 집계.
    """
    w = slice_norm[slice_norm["_year_week"] == str(year_week).strip()].copy()
    if w.empty:
        return pd.DataFrame()

    retain_floor = max(0, int(MIN_STORE_RETAIN_QTY))
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

        hist_raw = r["_hist_sale"] if "_hist_sale" in r.index else np.nan
        fc_raw = r["_fc_sale"] if "_fc_sale" in r.index else np.nan
        기판매량 = np.nan
        if pd.notna(hist_raw):
            기판매량 = round(max(float(hist_raw), 0.0), 2)
        예측판매량 = round(max(float(fc_raw), 0.0), 2) if pd.notna(fc_raw) else round(weekly_sales, 2)

        shortage = weekly_sales > stock + eps
        deficit = max(0.0, weekly_sales - stock) if shortage else 0.0

        # PLC 높음 = 판매 대비 재고 과다 → 잉여 출고 후보. 실제 넘길 수량은 바닥 재고(retain)를 남긴 만큼만.
        is_plc_source = weekly_sales > eps and plc > plc_weeks_threshold
        retain_cap = max(0, stock - retain_floor)
        plc_excess = (
            max(0.0, float(stock) - plc_weeks_threshold * weekly_sales) if is_plc_source else 0.0
        )
        excess_transfer_raw = min(plc_excess, float(retain_cap)) if is_plc_source else 0.0
        excess_transfer = int(round(excess_transfer_raw))
        # 재고가 retain 이하여서 넘길 수 없으면 출고 후보 아님 → 역할 정상 유지
        is_source = is_plc_source and retain_cap > 0 and excess_transfer > 0

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
                "기판매량": 기판매량,
                "예측판매량": 예측판매량,
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


def overview_kpis(
    store_level: pd.DataFrame,
    summary: pd.DataFrame,
    stockout_horizon_weeks: float = 2.0,
) -> dict:
    """
    상단 4메트릭.
    - 분배된 매장수: 선택 주차에서 기초재고 > 0 인 매장·SKU 중 고유 매장 수.
    - 누판율: 실적 기판매 합이 있으면 100×(실적합/총기초재고), 없으면 주간수요 합으로 동일 식 근사(%).
    - 2주 내 결품 위험 매장 수: 주간수요>0 이고 기초재고/주간수요 < 2 인 행이 하나라도 있는 고유 매장 수.
    - 추가 발주량: SKU 취합 표의 물류+회전_반영_추가발주 합.
    """
    out = {
        "분배된_매장수": 0,
        "누판율_퍼센트": 0.0,
        "주차내_결품위험_매장수": 0,
        "추가_발주량": 0,
    }
    if store_level.empty:
        return out

    inv = pd.to_numeric(store_level["기초재고"], errors="coerce").fillna(0)
    distributed_mask = inv > 0
    out["분배된_매장수"] = int(store_level.loc[distributed_mask, "매장"].nunique())

    hist = pd.to_numeric(store_level["기판매량"], errors="coerce")
    hist_sum = float(hist.sum()) if hist.notna().any() else 0.0
    dem = pd.to_numeric(store_level["주간예측수요"], errors="coerce").fillna(0.0)
    dem_sum = float(dem.sum())
    inv_sum = float(inv.sum())
    eps = 1e-6
    num = hist_sum if hist_sum > eps else dem_sum
    out["누판율_퍼센트"] = round(100.0 * num / max(inv_sum, 1.0), 2)

    d = pd.to_numeric(store_level["주간예측수요"], errors="coerce").fillna(0.0)
    s = pd.to_numeric(store_level["기초재고"], errors="coerce").fillna(0.0)
    risk = (d > eps) & (s / d < float(stockout_horizon_weeks))
    out["주차내_결품위험_매장수"] = int(store_level.loc[risk, "매장"].nunique())

    if summary is not None and not summary.empty and "물류+회전_반영_추가발주" in summary.columns:
        out["추가_발주량"] = int(pd.to_numeric(summary["물류+회전_반영_추가발주"], errors="coerce").fillna(0).sum())

    return out


def logistics_stock_two_rows(center_qty: int, store_deficit_total: int) -> pd.DataFrame:
    """
    물류 재고를 2행으로 표시: ① 센터 보유 합 ② 매장 결품 부족 충당 가정 후 센터 잔여.
    (센터에서 매장 부족분을 우선 충당한다고 가정한 수치입니다.)
    """
    c = max(0, int(center_qty))
    d = max(0, int(store_deficit_total))
    ship_to_stores = min(c, d)
    remaining = c - ship_to_stores
    return pd.DataFrame(
        [
            {"구분": "보유 재고 (물류센터)", "수량": c},
            {"구분": "매장 부족 충당 후 잔여", "수량": remaining},
        ]
    )


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

    st.title("스파오 리오더 의사결정 Agent")
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

    # 동일 (sku, 매장, 주차)에 실적·예측 행이 나뉘어 있으면 한 행으로 합침 (sale_qty 의미는 is_forecast로 구분)
    norm = merge_weekly_actual_forecast(norm)

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

    sku_choices = sorted(store_level["sku"].astype(str).unique().tolist())
    _detail_default = pick if pick in sku_choices else (sku_choices[0] if sku_choices else "")
    _detail_ix = sku_choices.index(_detail_default) if _detail_default in sku_choices else 0
    detail_sku = st.sidebar.selectbox(
        "매장 전개 · 물류 · 최종 발주 SKU",
        options=sku_choices or [""],
        index=min(_detail_ix, max(0, len(sku_choices) - 1)) if sku_choices else 0,
        help="주차×매장 표(물류재고 열·판매·기초재고), 최종 추가 발주 요약에 쓰는 SKU입니다.",
    )

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

    ov = overview_kpis(store_level, summary, stockout_horizon_weeks=2.0)

    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric("분배된 매장수", f"{ov['분배된_매장수']:,}")
    with m2:
        st.metric("누판율", f"{ov['누판율_퍼센트']:.1f}%")
    with m3:
        st.metric("2주 내 결품 위험 매장 수", f"{ov['주차내_결품위험_매장수']:,}")
    with m4:
        st.metric("추가 발주량", f"{ov['추가_발주량']:,}")

    st.markdown("---")
    st.subheader("SKU별 취합")
    st.caption(
        "매장결품부족_합: 결품 위험 매장의 max(예측−기초재고) 합. "
        "회전가능잉여_합: PLC(임계 초과)·판매 부진 매장만 대상으로, 매장마다 최소 기초재고 "
        f"{MIN_STORE_RETAIN_QTY}장을 남긴 뒤 넘길 수 있는 잉여만 합산(상한 캡). "
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
    st.subheader("선택 SKU · 주차 × 매장 표")
    wide_df, wide_fc_mask, wide_note = build_sku_week_wide_table(
        norm, str(detail_sku), center_df, center_by_sku
    )
    st.caption(
        "행=주차, 열=**물류재고(합)** + 매장별 **판매**(실적만 표시·예측만일 때 **빨간색**) + **기초재고량**. "
        + wide_note
    )
    if wide_df.empty:
        st.warning(
            f"SKU **`{detail_sku}`** 에 해당하는 `sku_weekly_forecast` 행이 없어 주차·매장 표를 만들 수 없습니다."
        )
    else:
        st.markdown("**주차 × 매장**")
        styled = style_week_store_wide(wide_df.reset_index(drop=True), wide_fc_mask.reset_index(drop=True))
        st.dataframe(styled, use_container_width=True, hide_index=True)

    sum_detail = summary[summary["sku"].astype(str) == str(detail_sku)]

    st.markdown("---")
    st.subheader("최종: 물류·회전 반영 후 추가 발주")
    st.caption(
        "**매장 부족 합**에서 물류 재고를 빼고, 남은 부족을 **회전 잉여**로 메운 뒤의 **추가 발주**와 MOQ 반영 제안입니다. "
        "(SKU 취합 표의 `물류+회전_반영_추가발주`와 동일 로직.)"
    )
    if sum_detail.empty:
        st.info("선택 SKU에 대한 취합 요약이 없습니다.")
    else:
        r0 = sum_detail.iloc[0]
        deficit_all = int(r0["매장결품부족_합"])
        center_q = int(r0["물류센터_재고"])
        rot_pool = int(r0["회전가능잉여_합"])
        rot_use = int(r0["회전으로_충당(상한)"])
        after_center = max(0, deficit_all - center_q)
        need_extra = int(r0["물류+회전_반영_추가발주"])
        moq_sug = int(r0["MOQ반영_제안발주"])
        use_center = min(center_q, deficit_all)
        st.markdown(
            f"**SKU:** `{detail_sku}` · {r0.get('sku_name', '')}  \n"
            f"- 매장 결품 **부족 합:** **{deficit_all:,}**  \n"
            f"- 물류센터 **보유:** **{center_q:,}** → 부족 충당에 **최대** **{use_center:,}** 사용 가정, "
            f"부족 **잔여:** **{after_center:,}**  \n"
            f"- 회전 가능 잉여 **합:** **{rot_pool:,}** → 위 잔여에 **{rot_use:,}** 까지 충당  \n"
            f"- <span class='hl-short'>** 추가 발주 필요(추정):** **{need_extra:,}** </span>  \n"
            f"- MOQ 반영 **제안 발주:** **{moq_sug:,}**",
            unsafe_allow_html=True,
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

    det = store_level[store_level["sku"].astype(str) == str(detail_sku)].copy()
    if role_filter:
        det = det[det["역할"].isin(role_filter)].copy()

    view_det = det.copy()
    if view_store != "전체":
        view_det = view_det[view_det["매장"].astype(str) == str(view_store)].copy()

    det_display = view_det[
        [
            "매장",
            "기판매량",
            "예측판매량",
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
            "기판매량": st.column_config.NumberColumn("기판매량", format="%.2f"),
            "예측판매량": st.column_config.NumberColumn("예측판매량", format="%.2f"),
            "주간예측수요": st.column_config.NumberColumn("주간수요(계산)", format="%.2f"),
            "PLC_주": st.column_config.NumberColumn("PLC(주)", format="%.2f"),
            "결품부족분": st.column_config.NumberColumn("결품 부족", format="%d"),
            "회전가능잉여": st.column_config.NumberColumn("회전 잉여", format="%d"),
        },
    )

    sub_sum = summary[summary["sku"].astype(str) == str(detail_sku)]
    if not sub_sum.empty:
        row = sub_sum.iloc[0]
        st.markdown(
            f"**선택 SKU 요약** — 부족 합: **{int(row['매장결품부족_합']):,}**, "
            f"물류센터: **{int(row['물류센터_재고']):,}**, "
            f"회전 잉여 합: **{int(row['회전가능잉여_합']):,}**, "
            f"<span class='hl-short'>추가 발주(추정): **{int(row['물류+회전_반영_추가발주']):,}**</span>",
            unsafe_allow_html=True,
        )
    elif detail_sku:
        st.info(
            "선택한 SKU가 `public.sku_weekly_forecast` 기준 상단 취합에 없을 수 있습니다. "
            "아래는 동일 테이블의 매장별 행입니다."
        )


if __name__ == "__main__":
    main()
