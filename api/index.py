import os
import re
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from fastapi import FastAPI, Query
from fastapi.responses import HTMLResponse, JSONResponse
from supabase import create_client

# 로컬 개발에서는 .env를 자동 로드(배포 환경에는 영향 없음)
try:
    from dotenv import load_dotenv  # type: ignore

    load_dotenv()
except Exception:
    pass

app = FastAPI()

# =========================
# 환경변수 (Vercel)
# =========================
SUPABASE_URL = os.getenv("SUPABASE_URL", "").strip()
SUPABASE_KEY = os.getenv("SUPABASE_KEY", "").strip()

if SUPABASE_URL and SUPABASE_KEY:
    supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
else:
    supabase = None

# 회전 출고(잉여 재고) 판단: PLC(주) = 기초재고÷주간예측이 이 값을 넘는 매장에서 잉여 계산.
DEFAULT_PLC_WEEKS: float = 4.0

# 리오더 제안 수량: (해당 SKU 주간 예측 합) × 이 주수.
DEFAULT_REORDER_SALES_WEEKS: float = 4.0

# 회전 출고 시 매장당 반드시 남겨 둘 최소 기초재고(장).
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


_YW_KO_RE = re.compile(r"^\s*(\d+)\s*년\s*(\d+)\s*월\s*(\d+)\s*주차\s*$", re.UNICODE)
_YW_ISO_RE = re.compile(r"^\s*(\d{4})\s*[-]?\s*[Ww]?\s*(\d{1,2})\s*$")


def year_week_sort_key(label: str) -> Tuple:
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
    seen = set()
    uniq: List[str] = []
    for x in labels:
        t = str(x).strip()
        if not t or t in seen:
            continue
        seen.add(t)
        uniq.append(t)
    return sorted(uniq, key=year_week_sort_key, reverse=False)


def diagnose_env() -> Optional[str]:
    if not SUPABASE_URL or not SUPABASE_KEY:
        return (
            "환경변수 `SUPABASE_URL`, `SUPABASE_KEY`가 설정되어 있지 않습니다. "
            "Vercel 프로젝트 Settings → Environment Variables에 추가하세요."
        )
    if supabase is None:
        return "Supabase 클라이언트를 만들지 못했습니다. 환경변수를 확인하세요."
    return None


def load_supabase_table(table_name: str, page_size: int = 1000) -> pd.DataFrame:
    if supabase is None:
        return pd.DataFrame()
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


def load_supabase_filtered(
    table_name: str,
    *,
    eq_filters: Optional[dict] = None,
    in_filters: Optional[dict] = None,
    order_by: Optional[str] = None,
    ascending: bool = True,
    range_from: Optional[int] = None,
    range_to: Optional[int] = None,
    select_cols: str = "*",
) -> pd.DataFrame:
    """
    서버리스 타임아웃 회피용: 필요한 행만 Supabase에서 필터링해 로드.
    - eq_filters: {"year_week": "2026-W03"}
    - in_filters: {"sku": ["A", "B"]}
    - range_from/to: pagination (inclusive)
    """
    if supabase is None:
        return pd.DataFrame()

    q = supabase.table(table_name).select(select_cols)
    if eq_filters:
        for k, v in eq_filters.items():
            q = q.eq(k, v)
    if in_filters:
        for k, vs in in_filters.items():
            vs2 = [x for x in (vs or []) if str(x).strip() != ""]
            if vs2:
                q = q.in_(k, vs2)
    if order_by:
        q = q.order(order_by, desc=not ascending)
    if range_from is not None and range_to is not None:
        q = q.range(int(range_from), int(range_to))
    res = q.execute()
    return pd.DataFrame(res.data or [])


def _is_forecast_to_bool(x) -> bool:
    if pd.isna(x):
        return True
    if isinstance(x, (bool, np.bool_)):
        return bool(x)
    s = str(x).strip().lower()
    if s in ("false", "0", "no", "f", "n"):
        return False
    return True


def normalize_weekly_slice(weekly_df: pd.DataFrame) -> pd.DataFrame:
    if weekly_df.empty:
        return pd.DataFrame()

    sku_c = first_existing_col(weekly_df, ["sku", "SKU"])
    store_c = first_existing_col(weekly_df, ["store_name", "storename", "매장", "plant", "PLANT"])
    yw_c = first_existing_col(weekly_df, ["year_week", "yearweek"])
    dem_c = first_existing_col(weekly_df, ["sale_qty", "forecast_qty", "forecastqty", "saleqty"])
    fc_qty_c = first_existing_col(
        weekly_df, ["forecast_qty", "forecastqty", "week_forecast_qty", "pred_sale_qty"]
    )
    hist_c = first_existing_col(
        weekly_df,
        ["sold_qty", "actual_sale_qty", "hist_sale_qty", "base_sale_qty", "sale_qty", "saleqty"],
    )
    if fc_qty_c and hist_c and fc_qty_c == hist_c:
        hist_c = first_existing_col(
            weekly_df, ["sold_qty", "actual_sale_qty", "hist_sale_qty", "base_sale_qty", "saleqty"]
        )
    st_c = first_existing_col(weekly_df, ["begin_stock", "beginstock", "stock_qty"])
    fc_c = first_existing_col(weekly_df, ["is_forecast", "isforecast"])
    name_c = first_existing_col(weekly_df, ["sku_name", "skuname", "SKU_NAME"])
    style_c = first_existing_col(weekly_df, ["style_code", "stylecode", "style", "STYLE_CODE", "style_cd"])
    ca_c = first_existing_col(weekly_df, ["created_at", "createdat"])
    fr_c = first_existing_col(weekly_df, ["forecast_run_id", "forecast_runid", "id"])

    if not sku_c or not yw_c:
        return pd.DataFrame()

    out = weekly_df.copy()
    out["_sku"] = out[sku_c].astype(str).str.strip()
    out["_store"] = (
        out[store_c].astype(str).str.strip() if store_c else pd.Series("_unknown", index=out.index)
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
        out["_demand"] = _raw_sale if dem_c else 0.0
        out["_fc_sale"] = out[fc_qty_c].apply(clean_number) if fc_qty_c else np.nan
        out["_hist_sale"] = out[hist_c].apply(clean_number) if hist_c else np.nan

    out["_stock"] = out[st_c].apply(to_int_safe) if st_c else 0
    out["_sku_name"] = out[name_c].astype(str).str.strip() if name_c else out["_sku"]
    out["_style_code"] = out[style_c].astype(str).str.strip() if style_c else pd.Series("", index=out.index)
    out["_created"] = pd.to_datetime(out[ca_c], errors="coerce") if ca_c else pd.NaT
    out["_frid"] = out[fr_c] if fr_c else np.nan

    out = out[out["_sku"] != ""].copy()
    out = out[out["_year_week"] != ""].copy()
    return out


def merge_weekly_actual_forecast(df: pd.DataFrame) -> pd.DataFrame:
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
                "_style_code": str(last.get("_style_code", "")).strip(),
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
    g = center_df.groupby(sku_key, dropna=False)[qty_c].apply(lambda s: sum(to_int_safe(x) for x in s))
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


def compute_store_rows_for_week(slice_norm: pd.DataFrame, year_week: str, plc_weeks_threshold: float) -> pd.DataFrame:
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
        sold = np.nan
        if pd.notna(hist_raw):
            sold = round(max(float(hist_raw), 0.0), 2)
        fc = round(max(float(fc_raw), 0.0), 2) if pd.notna(fc_raw) else round(weekly_sales, 2)

        shortage = weekly_sales > stock + eps
        deficit = max(0.0, weekly_sales - stock) if shortage else 0.0

        is_plc_source = weekly_sales > eps and plc >= plc_weeks_threshold and stock > retain_floor
        retain_cap = max(0, stock - retain_floor)
        plc_excess = max(0.0, float(stock) - plc_weeks_threshold * weekly_sales) if is_plc_source else 0.0
        excess_transfer_raw = min(plc_excess, float(retain_cap)) if is_plc_source else 0.0
        excess_transfer = int(round(excess_transfer_raw))
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
                "기판매량": sold,
                "예측판매량": fc,
                "주간예측수요": round(weekly_sales, 2),
                "기초재고": stock,
                "PLC_주": round(float(plc), 2) if np.isfinite(plc) else None,
                "역할": role,
                "결품부족분": int(round(deficit)),
                "회전가능잉여": int(round(excess_transfer)),
                "회전출고가능": bool(is_plc_source),
            }
        )
    return pd.DataFrame(rows)


def aggregate_sku_summary(store_df: pd.DataFrame, center_by_sku: pd.Series, moq_df: pd.DataFrame) -> pd.DataFrame:
    if store_df.empty:
        return pd.DataFrame()

    moq_map = moq_df.set_index("sku")["minimum_capacity"].to_dict() if not moq_df.empty else {}
    lt_map = moq_df.set_index("sku")["lead_time_days"].to_dict() if not moq_df.empty else {}

    parts = []
    for sku, g in store_df.groupby("sku"):
        total_deficit = int(g["결품부족분"].sum())
        total_rotation = int(g["회전가능잉여"].sum())
        n_short = int((g["역할"] == "결품위험").sum())
        n_source = int((g["역할"] == "회전출고(판매부진)").sum())
        center_qty = int(to_int_safe(center_by_sku.loc[sku])) if isinstance(center_by_sku, pd.Series) and sku in center_by_sku.index else 0

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


def compute_weekly_summary_table(norm: pd.DataFrame, center_df: pd.DataFrame, reorder_df: pd.DataFrame, plc_weeks_threshold: float) -> pd.DataFrame:
    if norm is None or norm.empty:
        return pd.DataFrame(
            columns=["주차", "결품위험 매장수", "회전 출고 가능 매장수", "물류센터 보유량", "추가 입고량", "MOQ", "리드타임"]
        )

    weeks = sort_year_week_labels_asc(norm["_year_week"].tolist())
    center_by_sku = center_stock_by_sku(center_df)
    center_total = int(pd.to_numeric(center_by_sku, errors="coerce").fillna(0).sum()) if not center_by_sku.empty else 0
    moq_df = reorder_params_by_sku(reorder_df)

    out_rows: List[dict] = []
    for yw in weeks:
        store_level = compute_store_rows_for_week(norm, str(yw), plc_weeks_threshold)
        if store_level.empty:
            continue
        summary = aggregate_sku_summary(store_level, center_by_sku, moq_df)

        short_stores = int(store_level.loc[store_level["역할"] == "결품위험", "매장"].nunique())
        rotation_stores = int(store_level.loc[store_level["회전출고가능"] == True, "매장"].nunique())
        add_in = int(pd.to_numeric(summary.get("물류+회전_반영_추가발주", 0), errors="coerce").fillna(0).sum()) if not summary.empty else 0

        need = summary[summary["물류+회전_반영_추가발주"] > 0].copy() if not summary.empty else pd.DataFrame()
        moq_max = int(pd.to_numeric(need.get("MOQ(참고)", 0), errors="coerce").fillna(0).max()) if not need.empty else 0
        lt_max = int(pd.to_numeric(need.get("리드타임_일", 0), errors="coerce").fillna(0).max()) if not need.empty else 0

        out_rows.append(
            {
                "주차": str(yw),
                "결품위험 매장수": short_stores,
                "회전 출고 가능 매장수": rotation_stores,
                "물류센터 보유량": center_total,
                "추가 입고량": add_in,
                "MOQ": moq_max,
                "리드타임": lt_max,
            }
        )

    return pd.DataFrame(out_rows)


def _fmt_int(x) -> str:
    try:
        return f"{int(x):,}"
    except Exception:
        return "0"


def _df_to_html_table(df: pd.DataFrame, max_rows: int = 200) -> str:
    if df is None or df.empty:
        return "<div class='muted'>표시할 데이터가 없습니다.</div>"
    view = df.head(max_rows).copy()
    return view.to_html(index=False, escape=True, classes="tbl")


def _layout(title: str, body_html: str) -> str:
    return f"""<!doctype html>
<html lang="ko">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>{title}</title>
  <style>
    body {{ font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, "Noto Sans KR", Arial; margin: 0; background: #0b1220; color: #e5e7eb; }}
    .wrap {{ max-width: 1400px; margin: 0 auto; padding: 18px 16px 40px; }}
    .card {{ background: linear-gradient(145deg, #0f172a 0%, #111c33 100%); border: 1px solid #22314f; border-radius: 14px; padding: 14px 14px; }}
    .grid {{ display: grid; grid-template-columns: 1fr; gap: 12px; }}
    @media (min-width: 1000px) {{ .grid2 {{ grid-template-columns: 1.2fr 1fr; }} }}
    h1 {{ margin: 8px 0 12px; font-size: 22px; letter-spacing: -0.02em; }}
    h2 {{ margin: 0 0 10px; font-size: 16px; color: #cbd5e1; }}
    .muted {{ color: #94a3b8; font-size: 13px; }}
    .row {{ display: flex; gap: 10px; flex-wrap: wrap; align-items: end; }}
    label {{ display: block; font-size: 12px; color: #94a3b8; margin-bottom: 6px; }}
    select, input {{ background: #0b1220; border: 1px solid #22314f; border-radius: 10px; color: #e5e7eb; padding: 10px 10px; min-width: 220px; }}
    button {{ background: #2563eb; border: 1px solid #1d4ed8; color: white; padding: 10px 12px; border-radius: 10px; cursor: pointer; }}
    button:hover {{ background: #1d4ed8; }}
    .kpis {{ display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 10px; }}
    @media (min-width: 900px) {{ .kpis {{ grid-template-columns: repeat(4, minmax(0, 1fr)); }} }}
    .kpi {{ background: #0b1220; border: 1px solid #22314f; border-radius: 14px; padding: 12px 12px; }}
    .kpi .t {{ font-size: 12px; color: #94a3b8; }}
    .kpi .v {{ font-size: 20px; font-weight: 700; margin-top: 6px; }}
    .headline {{ border: 1px solid #22314f; border-radius: 14px; padding: 12px 12px; background: #0b1220; font-weight: 600; line-height: 1.45; }}
    .hl {{ color: #f87171; font-weight: 700; }}
    .tbl {{ width: 100%; border-collapse: collapse; font-size: 13px; overflow: hidden; border-radius: 12px; }}
    .tbl th, .tbl td {{ border-bottom: 1px solid #22314f; padding: 8px 8px; vertical-align: top; }}
    .tbl th {{ text-align: left; color: #cbd5e1; background: #0b1220; position: sticky; top: 0; }}
    .pill {{ display:inline-block; padding: 2px 8px; border-radius: 999px; border: 1px solid #22314f; background:#0b1220; }}
  </style>
</head>
<body>
  <div class="wrap">
    {body_html}
  </div>
</body>
</html>"""


def overview_kpis(store_level: pd.DataFrame, summary: pd.DataFrame, norm_df: Optional[pd.DataFrame] = None, stockout_horizon_weeks: float = 2.0) -> dict:
    out = {"분배된_매장수": 0, "누판율_퍼센트": 0.0, "주차내_결품위험_매장수": 0, "추가_발주량": 0}
    if store_level is None or store_level.empty:
        return out

    if norm_df is not None and not norm_df.empty and "_store" in norm_df.columns:
        stores = norm_df["_store"].astype(str).str.strip()
        stores = stores[(stores != "") & (stores != "_unknown")]
        out["분배된_매장수"] = int(stores.nunique())
    else:
        out["분배된_매장수"] = int(store_level["매장"].astype(str).str.strip().nunique())

    total_sales = float(pd.to_numeric(store_level["기판매량"], errors="coerce").fillna(0.0).sum())
    total_receipt = float(pd.to_numeric(store_level["기초재고"], errors="coerce").fillna(0.0).sum())
    out["누판율_퍼센트"] = round(100.0 * total_sales / max(total_receipt, 1.0), 2)

    d = pd.to_numeric(store_level["주간예측수요"], errors="coerce").fillna(0.0)
    s = pd.to_numeric(store_level["기초재고"], errors="coerce").fillna(0.0)
    eps = 1e-6
    risk = (d > eps) & (s / d < float(stockout_horizon_weeks))
    out["주차내_결품위험_매장수"] = int(store_level.loc[risk, "매장"].nunique())

    if summary is not None and not summary.empty and "물류+회전_반영_추가발주" in summary.columns:
        out["추가_발주량"] = int(pd.to_numeric(summary["물류+회전_반영_추가발주"], errors="coerce").fillna(0).sum())

    return out


def reorder_guidance_for_store_slice(
    sl: pd.DataFrame,
    center_by_sku: pd.Series,
    lead_time_days: int,
    minimum_capacity: int,
    sales_weeks: float,
    scope_prefix: str = "",
) -> str:
    if sl is None or sl.empty:
        return f"{scope_prefix}해당 조건의 매장 행이 없어 리오더 안내를 표시할 수 없습니다."

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
    sw = float(sales_weeks) if sales_weeks else float(DEFAULT_REORDER_SALES_WEEKS)
    if sw <= 0:
        sw = float(DEFAULT_REORDER_SALES_WEEKS)

    order_qty_raw = int(round(weekly_total * sw)) if weekly_total > 0 else 0
    order_qty = max(order_qty_raw, moq) if moq > 0 else order_qty_raw

    eps = 1e-6
    if weekly_total <= eps:
        sw_disp = int(sw) if abs(sw - round(sw)) < 1e-9 else sw
        return (
            f"{scope_prefix}리오더 리드타임 <span class='pill'>{lt}일</span> 기준, 주간 수요 합이 <span class='pill'>0</span>이라 "
            f"결품 방지 발주 기한을 산출할 수 없습니다. "
            f"(최소발주수량: <span class='pill'>{moq}장</span>, <span class='pill'>{sw_disp}주</span> 판매량 기준 발주 시 제안 수량: "
            f"<span class='pill'>{order_qty}장</span>)."
        )

    cover_days = 7.0 * float(total_inv) / weekly_total
    days_until_must_order = cover_days - float(lt)
    must_in_days = int(np.floor(max(0.0, days_until_must_order)))
    mo = int((pd.Timestamp.now().normalize() + pd.Timedelta(days=must_in_days)).month)
    dd = int((pd.Timestamp.now().normalize() + pd.Timedelta(days=must_in_days)).day)
    sw_disp = int(sw) if abs(sw - round(sw)) < 1e-9 else sw

    return (
        f"{scope_prefix}리오더 리드타임 <span class='pill'>{lt}일</span> 기준, <span class='pill'>{mo}월 {dd}일</span> 이내 "
        f"<span class='pill'>{order_qty}장</span> 리오더 발주 필요합니다 "
        f"(최소발주수량: <span class='pill'>{moq}장</span>, <span class='pill'>{sw_disp}주</span> 판매량 기준 발주)."
    )


@app.get("/health")
def health():
    msg = diagnose_env()
    if msg:
        return JSONResponse({"ok": False, "error": msg}, status_code=500)
    return {"ok": True}


@app.get("/debug-env")
def debug_env():
    """
    배포 디버깅용: 환경변수 존재 여부만 반환(키 값은 절대 노출하지 않음).
    """
    url = os.getenv("SUPABASE_URL", "").strip()
    key = os.getenv("SUPABASE_KEY", "").strip()
    return {
        "has_SUPABASE_URL": bool(url),
        "has_SUPABASE_KEY": bool(key),
        "SUPABASE_URL_host": (url.split("://", 1)[-1].split("/", 1)[0] if url else ""),
        "SUPABASE_KEY_len": (len(key) if key else 0),
        "note": "값은 숨기고 존재 여부만 표시합니다.",
    }


@app.get("/api/analyze")
def analyze(
    year_week: str = Query(..., description="분석 기준 주차"),
    offset: int = Query(0, ge=0, description="SKU 페이지네이션 오프셋(행 기준)"),
    limit: int = Query(100, ge=1, le=500, description="한 번에 계산할 SKU 수(권장 100~300)"),
    plc_weeks: float = Query(DEFAULT_PLC_WEEKS, gt=0, le=52, description="PLC(주) 임계값"),
):
    """
    서버리스(5분) 타임아웃 회피용 배치 API.
    - 한 번에 전체 SKU를 계산하지 않고, offset/limit 단위로 잘라서 처리합니다.
    - 반환값에는 next_offset(다음 호출용)과 rows(요약 표)가 포함됩니다.
    """
    msg = diagnose_env()
    if msg:
        return JSONResponse({"ok": False, "error": msg}, status_code=500)

    yw = str(year_week).strip()
    if not yw:
        return JSONResponse({"ok": False, "error": "`year_week`가 비어 있습니다."}, status_code=400)

    # 1) SKU 목록을 '행 기준'으로 페이징하며 unique SKU를 limit개 채움
    #    (Supabase distinct가 제한적이라, 행을 조금 넉넉히 읽고 unique로 압축합니다.)
    unique_skus: List[str] = []
    scan_offset = int(offset)
    scan_hard_limit = int(limit) * 30  # 중복이 많을 수 있어 여유
    scanned_rows = 0

    while len(unique_skus) < int(limit) and scanned_rows < scan_hard_limit:
        page = load_supabase_filtered(
            "sku_weekly_forecast",
            eq_filters={"year_week": yw},
            order_by="sku",
            ascending=True,
            range_from=scan_offset,
            range_to=scan_offset + 999,
            select_cols="sku",
        )
        if page.empty or "sku" not in page.columns:
            break
        scanned_rows += len(page)
        for s in page["sku"].astype(str).tolist():
            s = str(s).strip()
            if s and s not in unique_skus:
                unique_skus.append(s)
                if len(unique_skus) >= int(limit):
                    break
        scan_offset += len(page)
        if len(page) < 1000:
            break

    if not unique_skus:
        return {
            "ok": True,
            "year_week": yw,
            "offset": offset,
            "limit": limit,
            "next_offset": scan_offset,
            "skus": [],
            "rows": [],
            "note": "해당 주차에 데이터가 없거나, 더 이상 SKU가 없습니다.",
        }

    # 2) 선택된 SKU들에 대해서만 필요한 테이블을 로드
    weekly_raw = load_supabase_filtered(
        "sku_weekly_forecast",
        eq_filters={"year_week": yw},
        in_filters={"sku": unique_skus},
    )
    center_df = load_supabase_filtered("center_stock", in_filters={"sku": unique_skus})
    reorder_df = load_supabase_filtered("reorder", in_filters={"sku": unique_skus})

    norm = normalize_weekly_slice(weekly_raw.copy())
    norm = merge_weekly_actual_forecast(norm) if not norm.empty else norm

    store_level = compute_store_rows_for_week(norm, yw, float(plc_weeks))
    center_by_sku = center_stock_by_sku(center_df)
    moq_df = reorder_params_by_sku(reorder_df)
    summary = aggregate_sku_summary(store_level, center_by_sku, moq_df) if not store_level.empty else pd.DataFrame()

    rows = summary.to_dict(orient="records") if summary is not None and not summary.empty else []
    return {
        "ok": True,
        "year_week": yw,
        "offset": offset,
        "limit": limit,
        "next_offset": scan_offset,  # 다음 호출 때 offset으로 그대로 넣으면 됩니다(행 기준)
        "sku_count": len(unique_skus),
        "skus": unique_skus,
        "rows": rows,
        "note": "서버리스 타임아웃 회피를 위해 SKU를 청크로 계산합니다.",
    }


@app.get("/", response_class=HTMLResponse)
def home(
    year_week: Optional[str] = Query(default=None),
    style_code: Optional[str] = Query(default=None),
    detail_sku: Optional[str] = Query(default=None),
    role_filter: Optional[str] = Query(default="결품위험,회전출고(판매부진)"),
    view_store: Optional[str] = Query(default="전체"),
):
    env_msg = diagnose_env()
    if env_msg:
        body = f"""
        <h1>스파오 리오더 의사결정 Agent</h1>
        <div class="card">
          <h2>배포 설정이 필요합니다</h2>
          <div class="muted">{env_msg}</div>
          <div class="muted" style="margin-top:10px;">(참고) `/health`에서 상태를 확인할 수 있습니다.</div>
        </div>
        """
        return _layout("결품·발주 취합", body)

    weekly_raw = load_supabase_table("sku_weekly_forecast")
    center_df = load_supabase_table("center_stock")
    reorder_df = load_supabase_table("reorder")

    if weekly_raw.empty:
        body = """
        <h1>스파오 리오더 의사결정 Agent</h1>
        <div class="card"><div class="muted">`public.sku_weekly_forecast` 테이블에 데이터가 없습니다.</div></div>
        """
        return _layout("결품·발주 취합", body)

    norm = normalize_weekly_slice(weekly_raw.copy())
    if norm.empty:
        body = """
        <h1>스파오 리오더 의사결정 Agent</h1>
        <div class="card"><div class="muted">`sku`/`year_week` 등 필수 컬럼을 확인하세요.</div></div>
        """
        return _layout("결품·발주 취합", body)

    norm = merge_weekly_actual_forecast(norm)
    yw_list = sort_year_week_labels(norm["_year_week"].tolist())
    if not yw_list:
        body = """
        <h1>스파오 리오더 의사결정 Agent</h1>
        <div class="card"><div class="muted">유효한 `year_week` 값을 찾지 못했습니다.</div></div>
        """
        return _layout("결품·발주 취합", body)

    plc_thr = float(DEFAULT_PLC_WEEKS)
    if not year_week or str(year_week).strip() not in yw_list:
        year_week = yw_list[0]

    center_by_sku = center_stock_by_sku(center_df)
    moq_df = reorder_params_by_sku(reorder_df)

    store_level = compute_store_rows_for_week(norm, str(year_week), plc_thr)
    if store_level.empty:
        body = f"""
        <h1>스파오 리오더 의사결정 Agent</h1>
        <div class="card"><div class="muted">선택한 주차 `{year_week}`에 해당하는 행이 없습니다.</div></div>
        """
        return _layout("결품·발주 취합", body)

    summary = aggregate_sku_summary(store_level, center_by_sku, moq_df)
    weekly_summary = compute_weekly_summary_table(norm, center_df, reorder_df, plc_thr)

    # style_code 옵션
    has_style = "style_code" in store_level.columns and (store_level["style_code"].astype(str).str.strip() != "").any()
    style_opts = []
    if has_style:
        style_opts = sorted({str(x).strip() for x in store_level["style_code"].tolist() if str(x).strip()})
        if not style_code or str(style_code).strip() not in style_opts:
            style_code = style_opts[0] if style_opts else None

    sku_choices = sorted(store_level["sku"].astype(str).unique().tolist())
    if not detail_sku or str(detail_sku).strip() not in sku_choices:
        detail_sku = sku_choices[0] if sku_choices else ""

    # 리오더 헤드라인 슬라이스 (style_code 있으면 style 범위, 없으면 detail_sku 단일)
    scope_prefix = ""
    sl_headline = pd.DataFrame()
    lt = 0
    moq = 0
    if has_style and style_code:
        sl_headline = store_level[store_level["style_code"].astype(str) == str(style_code)].copy()
        scope_prefix = f"[style_code <strong>{style_code}</strong>] "
        # 보수적으로: style 내 SKU들의 최댓값
        sk_list = sorted(sl_headline["sku"].astype(str).unique().tolist())
        if not moq_df.empty and "sku" in moq_df.columns:
            sub = moq_df[moq_df["sku"].astype(str).isin(sk_list)]
            if not sub.empty:
                lt = int(pd.to_numeric(sub["lead_time_days"], errors="coerce").fillna(0).max())
                moq = int(pd.to_numeric(sub["minimum_capacity"], errors="coerce").fillna(0).max())
    else:
        sl_headline = store_level[store_level["sku"].astype(str) == str(detail_sku)].copy()
        gr = summary[summary["sku"].astype(str) == str(detail_sku)]
        if not gr.empty:
            lt = int(gr["리드타임_일"].iloc[0])
            moq = int(gr["MOQ(참고)"].iloc[0])

    headline_txt = reorder_guidance_for_store_slice(
        sl_headline, center_by_sku, lt, moq, float(DEFAULT_REORDER_SALES_WEEKS), scope_prefix=scope_prefix
    )

    ov = overview_kpis(store_level, summary, norm_df=norm, stockout_horizon_weeks=2.0)

    # 드릴다운 필터
    roles = [x.strip() for x in str(role_filter or "").split(",") if x.strip()]
    det = store_level[store_level["sku"].astype(str) == str(detail_sku)].copy()
    if roles:
        det = det[det["역할"].isin(roles)].copy()
    store_opts = sorted(store_level["매장"].astype(str).unique().tolist())
    view_det = det.copy()
    if view_store and view_store != "전체":
        view_det = view_det[view_det["매장"].astype(str) == str(view_store)].copy()

    det_display = view_det[
        ["매장", "기판매량", "예측판매량", "주간예측수요", "기초재고", "PLC_주", "역할", "결품부족분", "회전가능잉여"]
    ].sort_values(["역할", "결품부족분"], ascending=[True, False])

    # 폼 HTML
    yw_opts_html = "\n".join(
        [f"<option value='{pd.io.formats.format.escape_html(yw)}' {'selected' if yw == year_week else ''}>{pd.io.formats.format.escape_html(yw)}</option>" for yw in yw_list]
    )
    style_sel_html = ""
    if has_style and style_opts:
        opts = "\n".join(
            [f"<option value='{pd.io.formats.format.escape_html(sc)}' {'selected' if sc == style_code else ''}>{pd.io.formats.format.escape_html(sc)}</option>" for sc in style_opts]
        )
        style_sel_html = f"""
        <div>
          <label>style_code</label>
          <select name="style_code">{opts}</select>
        </div>
        """

    sku_opts_html = "\n".join(
        [f"<option value='{pd.io.formats.format.escape_html(s)}' {'selected' if s == detail_sku else ''}>{pd.io.formats.format.escape_html(s)}</option>" for s in sku_choices]
    )
    store_opts_html = "\n".join(
        ["<option value='전체' " + ("selected" if view_store == "전체" else "") + ">전체</option>"]
        + [f"<option value='{pd.io.formats.format.escape_html(s)}' {'selected' if s == view_store else ''}>{pd.io.formats.format.escape_html(s)}</option>" for s in store_opts]
    )

    # 테이블들
    keep_cols = ["주차", "결품위험 매장수", "회전 출고 가능 매장수", "물류센터 보유량", "추가 입고량", "MOQ", "리드타임"]
    weekly_summary_view = weekly_summary.reindex(columns=keep_cols) if not weekly_summary.empty else weekly_summary
    summary_cols = [
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
    summary_view = summary.reindex(columns=summary_cols) if not summary.empty else summary

    body = f"""
    <h1>스파오 리오더 의사결정 Agent</h1>

    <div class="card" style="margin-bottom:12px;">
      <h2>필터</h2>
      <form method="get" class="row">
        <div>
          <label>year_week</label>
          <select name="year_week">{yw_opts_html}</select>
        </div>
        {style_sel_html}
        <div>
          <label>상세 SKU</label>
          <select name="detail_sku">{sku_opts_html}</select>
        </div>
        <div>
          <label>역할 필터(콤마)</label>
          <input name="role_filter" value="{pd.io.formats.format.escape_html(role_filter or '')}" />
        </div>
        <div>
          <label>매장(상세표만)</label>
          <select name="view_store">{store_opts_html}</select>
        </div>
        <div>
          <button type="submit">적용</button>
        </div>
      </form>
      <div class="muted" style="margin-top:10px;">
        회전 출고 가능 매장수 정의: <span class="pill">기초재고 &gt; {MIN_STORE_RETAIN_QTY}</span> 이고 <span class="pill">PLC ≥ {DEFAULT_PLC_WEEKS}주</span>
      </div>
    </div>

    <div class="headline" style="margin-bottom:12px;">{headline_txt}</div>

    <div class="kpis" style="margin-bottom:12px;">
      <div class="kpi"><div class="t">분배된 매장수</div><div class="v">{_fmt_int(ov["분배된_매장수"])}</div></div>
      <div class="kpi"><div class="t">누판율</div><div class="v">{ov["누판율_퍼센트"]:.1f}%</div></div>
      <div class="kpi"><div class="t">2주 내 결품 위험 매장 수</div><div class="v">{_fmt_int(ov["주차내_결품위험_매장수"])}</div></div>
      <div class="kpi"><div class="t">추가 발주량</div><div class="v">{_fmt_int(ov["추가_발주량"])}</div></div>
    </div>

    <div class="grid grid2">
      <div class="card">
        <h2>주차별 취합</h2>
        {_df_to_html_table(weekly_summary_view, max_rows=300)}
      </div>
      <div class="card">
        <h2>SKU별 취합</h2>
        {_df_to_html_table(summary_view, max_rows=200)}
      </div>
    </div>

    <div class="card" style="margin-top:12px;">
      <h2>매장별 상세 (드릴다운)</h2>
      <div class="muted" style="margin-bottom:10px;">이 표만 필터(매장/역할)를 적용합니다. 나머지 계산은 전체 매장 기준입니다.</div>
      {_df_to_html_table(det_display, max_rows=500)}
    </div>
    """

    return _layout("결품·발주 취합", body)
