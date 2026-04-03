"""
결품 예측 · 매장간 이동 · 추가 발주
데이터 → 매장 부족/여유 판단 → 매장간 이동 → 센터 재고 → 최종 발주 자동 산출
"""

from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import streamlit as st
from supabase import create_client

st.set_page_config(
    page_title="결품 예측 · 매장간 이동 · 추가 발주 대시보드",
    layout="wide",
)

SUPABASE_URL = st.secrets["SUPABASE_URL"]
SUPABASE_KEY = st.secrets["SUPABASE_KEY"]
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)


# =========================================================
# 공통 유틸
# =========================================================
def first_existing_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    if df is None or df.empty:
        return None
    for col in candidates:
        if col in df.columns:
            return col
    lower_map = {str(c).lower(): c for c in df.columns}
    for col in candidates:
        if col.lower() in lower_map:
            return lower_map[col.lower()]
    return None


def normalize_text(v: Any) -> str:
    if pd.isna(v):
        return ""
    return str(v).strip()


def to_number(v: Any, default: float = 0.0) -> float:
    if pd.isna(v):
        return default
    s = str(v).strip().replace(",", "")
    if s == "":
        return default
    try:
        return float(s)
    except Exception:
        return default


def to_int(v: Any, default: int = 0) -> int:
    return int(round(to_number(v, default)))


@st.cache_data(ttl=300)
def load_table(table_name: str, page_size: int = 1000) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    start = 0

    while True:
        end = start + page_size - 1
        result = supabase.table(table_name).select("*").range(start, end).execute()
        data = result.data or []
        if not data:
            break
        rows.extend(data)
        if len(data) < page_size:
            break
        start += page_size

    return pd.DataFrame(rows)


@st.cache_data(ttl=300)
def load_data() -> Dict[str, pd.DataFrame]:
    return {
        "weekly": load_table("sku_weekly_forecast"),
        "center": load_table("center_stock"),
        "reorder": load_table("reorder"),
    }


# =========================================================
# 데이터 정규화
# =========================================================
def prepare_weekly_forecast(df: pd.DataFrame) -> pd.DataFrame:
    empty_cols = [
        "sku",
        "store_name",
        "plant",
        "sku_name",
        "style_code",
        "stage",
        "sale_qty",
        "begin_stock",
        "loss",
        "inbound_qty",
        "outbound_qty",
        "avg_discount_rate",
        "is_forecast",
        "is_peak_week",
        "year_week",
    ]
    if df.empty:
        return pd.DataFrame(columns=empty_cols)

    col_sku = first_existing_col(df, ["sku", "SKU"])
    if not col_sku:
        return pd.DataFrame(columns=empty_cols)

    col_store = first_existing_col(df, ["store_name", "store"])
    col_plant = first_existing_col(df, ["plant", "PLANT"])
    col_sku_name = first_existing_col(df, ["sku_name"])
    col_style = first_existing_col(df, ["sty", "style_code"])
    col_stage = first_existing_col(df, ["stage"])
    col_sale = first_existing_col(df, ["sale_qty", "forecast_qty"])
    col_begin = first_existing_col(df, ["begin_stock", "opening_stock", "base_stock"])
    col_loss = first_existing_col(df, ["loss"])
    col_inbound = first_existing_col(df, ["inbound_qty", "in_qty"])
    col_outbound = first_existing_col(df, ["outbound_qty", "out_qty"])
    col_discount = first_existing_col(df, ["avg_discount_rate", "avg_discount_ratio"])
    col_forecast = first_existing_col(df, ["is_forecast"])
    col_peak = first_existing_col(df, ["is_peak_week"])
    col_year_week = first_existing_col(df, ["year_week", "yearweek"])

    n = len(df)
    out = pd.DataFrame()
    out["sku"] = df[col_sku].map(normalize_text)
    out["store_name"] = df[col_store].map(normalize_text) if col_store else pd.Series([""] * n)
    out["plant"] = df[col_plant].map(normalize_text) if col_plant else pd.Series([""] * n)
    out["sku_name"] = df[col_sku_name].map(normalize_text) if col_sku_name else out["sku"]
    out["style_code"] = df[col_style].map(normalize_text) if col_style else pd.Series([""] * n)
    out["stage"] = df[col_stage].map(normalize_text) if col_stage else pd.Series([""] * n)
    out["sale_qty"] = df[col_sale].map(lambda x: to_number(x, 0.0)) if col_sale else pd.Series([0.0] * n)
    out["begin_stock"] = df[col_begin].map(lambda x: to_number(x, 0.0)) if col_begin else pd.Series([0.0] * n)
    out["loss"] = df[col_loss].map(lambda x: to_number(x, 0.0)) if col_loss else pd.Series([0.0] * n)
    out["inbound_qty"] = df[col_inbound].map(lambda x: to_number(x, 0.0)) if col_inbound else pd.Series([0.0] * n)
    out["outbound_qty"] = df[col_outbound].map(lambda x: to_number(x, 0.0)) if col_outbound else pd.Series([0.0] * n)
    out["avg_discount_rate"] = (
        df[col_discount].map(lambda x: to_number(x, 0.0)) if col_discount else pd.Series([0.0] * n)
    )
    if col_forecast:
        out["is_forecast"] = df[col_forecast].fillna(False).astype(bool)
    else:
        out["is_forecast"] = True
    if col_peak:
        out["is_peak_week"] = df[col_peak].fillna(False).astype(bool)
    else:
        out["is_peak_week"] = False
    out["year_week"] = df[col_year_week].map(normalize_text) if col_year_week else pd.Series([""] * n)

    out = out[out["sku"] != ""].copy()
    out["store_key"] = out["store_name"].where(out["store_name"] != "", out["plant"])
    return out


def prepare_center_stock(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["sku", "center", "stock_qty", "style_code"])

    col_sku = first_existing_col(df, ["sku", "SKU"])
    col_center = first_existing_col(df, ["center", "center_code"])
    col_stock = first_existing_col(df, ["stock_qty", "qty", "stock"])
    col_style = first_existing_col(df, ["style_code", "sty"])

    n = len(df)
    out = pd.DataFrame()
    out["sku"] = df[col_sku].map(normalize_text) if col_sku else pd.Series([""] * n)
    out["center"] = df[col_center].map(normalize_text) if col_center else pd.Series([""] * n)
    out["stock_qty"] = df[col_stock].map(lambda x: to_number(x, 0.0)) if col_stock else pd.Series([0.0] * n)
    out["style_code"] = df[col_style].map(normalize_text) if col_style else pd.Series([""] * n)
    out = out[out["sku"] != ""].copy()
    return out


def prepare_reorder(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["sku", "factory", "lead_time", "minimum_capacity", "style_code"])

    col_sku = first_existing_col(df, ["sku", "SKU"])
    col_factory = first_existing_col(df, ["factory"])
    col_lead = first_existing_col(df, ["lead_time"])
    col_min = first_existing_col(df, ["minimum_capacity"])
    col_style = first_existing_col(df, ["style_code", "sty"])

    n = len(df)
    out = pd.DataFrame()
    out["sku"] = df[col_sku].map(normalize_text) if col_sku else pd.Series([""] * n)
    out["factory"] = df[col_factory].map(normalize_text) if col_factory else pd.Series([""] * n)
    out["lead_time"] = df[col_lead].map(lambda x: to_number(x, 0.0)) if col_lead else pd.Series([0.0] * n)
    out["minimum_capacity"] = df[col_min].map(lambda x: to_number(x, 0.0)) if col_min else pd.Series([0.0] * n)
    out["style_code"] = df[col_style].map(normalize_text) if col_style else pd.Series([""] * n)
    out = out[out["sku"] != ""].copy()
    return out


# =========================================================
# 핵심 계산 로직
# =========================================================
def build_store_risk_table(weekly: pd.DataFrame) -> pd.DataFrame:
    if weekly.empty:
        return pd.DataFrame()

    grouped = (
        weekly.groupby(["sku", "sku_name", "style_code", "store_key", "plant"], dropna=False)
        .agg(
            begin_stock=("begin_stock", "sum"),
            forecast_sales=("sale_qty", "sum"),
            loss_qty=("loss", "sum"),
            inbound_qty=("inbound_qty", "sum"),
            outbound_qty=("outbound_qty", "sum"),
            peak_weeks=("is_peak_week", "sum"),
            avg_discount_rate=("avg_discount_rate", "mean"),
        )
        .reset_index()
    )

    grouped["expected_demand"] = grouped["forecast_sales"] + grouped["loss_qty"]
    grouped["available_stock_before_transfer"] = (
        grouped["begin_stock"] + grouped["inbound_qty"] - grouped["outbound_qty"]
    )
    grouped["projected_ending_stock"] = (
        grouped["available_stock_before_transfer"] - grouped["expected_demand"]
    )
    grouped["shortage_qty"] = grouped["projected_ending_stock"].apply(lambda x: abs(x) if x < 0 else 0)
    grouped["surplus_qty"] = grouped["projected_ending_stock"].apply(lambda x: x if x > 0 else 0)
    grouped["risk_flag"] = grouped["shortage_qty"] > 0

    def calc_weeks_of_cover(row: pd.Series) -> float:
        if row["forecast_sales"] <= 0:
            return 999.0
        return round(row["available_stock_before_transfer"] / row["forecast_sales"], 2)

    grouped["weeks_of_cover"] = grouped.apply(calc_weeks_of_cover, axis=1)
    return grouped


def allocate_transfer_and_center(
    store_df: pd.DataFrame,
    center_df: pd.DataFrame,
    reorder_df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if store_df.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    store_df = store_df.copy()
    center_stock_map: Dict[str, float] = (
        center_df.groupby("sku", dropna=False)["stock_qty"].sum().to_dict()
        if not center_df.empty
        else {}
    )
    reorder_info = (
        reorder_df.groupby("sku", dropna=False)
        .agg(
            lead_time=("lead_time", "max"),
            minimum_capacity=("minimum_capacity", "max"),
            factory=("factory", "first"),
        )
        .reset_index()
        if not reorder_df.empty
        else pd.DataFrame(columns=["sku", "lead_time", "minimum_capacity", "factory"])
    )

    summary_rows: List[Dict[str, Any]] = []
    transfer_rows: List[Dict[str, Any]] = []
    store_detail_rows: List[Dict[str, Any]] = []

    for sku, sku_df in store_df.groupby("sku", dropna=False):
        sku_df = sku_df.reset_index(drop=True)
        sku_name = str(sku_df["sku_name"].iloc[0])
        style_code = str(sku_df["style_code"].iloc[0])

        donors = sku_df[sku_df["surplus_qty"] > 0].copy().sort_values("surplus_qty", ascending=False)
        receivers = sku_df[sku_df["shortage_qty"] > 0].copy().sort_values("shortage_qty", ascending=False)

        donor_pool = {idx: float(row["surplus_qty"]) for idx, row in donors.iterrows()}
        transfer_total = 0.0

        sku_center_initial = (
            float(center_df[center_df["sku"] == sku]["stock_qty"].sum()) if not center_df.empty else 0.0
        )

        for _, recv_row in receivers.iterrows():
            shortage_start = float(recv_row["shortage_qty"])
            if shortage_start <= 0:
                continue
            needed = shortage_start

            for donor_idx, donor_row in donors.iterrows():
                available = donor_pool.get(donor_idx, 0.0)
                if available <= 0 or needed <= 0:
                    continue
                move_qty = min(available, needed)
                donor_pool[donor_idx] -= move_qty
                needed -= move_qty
                transfer_total += move_qty
                transfer_rows.append(
                    {
                        "sku": sku,
                        "sku_name": sku_name,
                        "style_code": style_code,
                        "from_store": donor_row["store_key"],
                        "to_store": recv_row["store_key"],
                        "transfer_qty": int(round(move_qty)),
                    }
                )

            needed_after_transfer = needed
            center_available = float(center_stock_map.get(sku, 0.0))
            center_alloc = min(center_available, needed)
            center_stock_map[sku] = center_available - center_alloc
            needed -= center_alloc
            final_order_qty = max(0.0, needed)
            transfer_in_qty = int(round(shortage_start - needed_after_transfer))

            store_detail_rows.append(
                {
                    "sku": sku,
                    "sku_name": sku_name,
                    "style_code": style_code,
                    "store_name": recv_row["store_key"],
                    "plant": recv_row["plant"],
                    "begin_stock": int(round(recv_row["begin_stock"])),
                    "forecast_sales": int(round(recv_row["forecast_sales"])),
                    "loss_qty": int(round(recv_row["loss_qty"])),
                    "shortage_before_action": int(round(shortage_start)),
                    "transfer_in_qty": transfer_in_qty,
                    "center_alloc_qty": int(round(center_alloc)),
                    "final_order_qty": int(round(final_order_qty)),
                    "weeks_of_cover": recv_row["weeks_of_cover"],
                    "risk_flag": True,
                }
            )

        sku_center_remaining = float(center_stock_map.get(sku, 0.0))
        used_center = sku_center_initial - sku_center_remaining
        total_shortage = float(sku_df["shortage_qty"].sum())
        total_surplus = float(sku_df["surplus_qty"].sum())

        additional_order_qty = max(0.0, total_shortage - transfer_total - used_center)

        reorder_row = reorder_info[reorder_info["sku"] == sku]
        lead_time = float(reorder_row["lead_time"].iloc[0]) if not reorder_row.empty else 0.0
        minimum_capacity = float(reorder_row["minimum_capacity"].iloc[0]) if not reorder_row.empty else 0.0
        factory = str(reorder_row["factory"].iloc[0]) if not reorder_row.empty else ""

        recommended_order_qty = additional_order_qty
        if minimum_capacity > 0 and recommended_order_qty > 0:
            recommended_order_qty = max(recommended_order_qty, minimum_capacity)

        summary_rows.append(
            {
                "sku": sku,
                "sku_name": sku_name,
                "style_code": style_code,
                "risk_store_count": int((sku_df["shortage_qty"] > 0).sum()),
                "surplus_store_count": int((sku_df["surplus_qty"] > 0).sum()),
                "total_forecast_sales": int(round(sku_df["forecast_sales"].sum())),
                "total_shortage_before_action": int(round(total_shortage)),
                "total_transfer_available": int(round(total_surplus)),
                "transfer_allocated_qty": int(round(transfer_total)),
                "center_stock_initial": int(round(sku_center_initial)),
                "center_stock_used": int(round(used_center)),
                "additional_order_qty": int(round(additional_order_qty)),
                "recommended_order_qty": int(round(recommended_order_qty)),
                "lead_time": int(round(lead_time)),
                "minimum_capacity": int(round(minimum_capacity)),
                "factory": factory,
            }
        )

        safe_rows = sku_df[sku_df["shortage_qty"] <= 0].copy()
        for _, safe_row in safe_rows.iterrows():
            store_detail_rows.append(
                {
                    "sku": sku,
                    "sku_name": sku_name,
                    "style_code": style_code,
                    "store_name": safe_row["store_key"],
                    "plant": safe_row["plant"],
                    "begin_stock": int(round(safe_row["begin_stock"])),
                    "forecast_sales": int(round(safe_row["forecast_sales"])),
                    "loss_qty": int(round(safe_row["loss_qty"])),
                    "shortage_before_action": 0,
                    "transfer_in_qty": 0,
                    "center_alloc_qty": 0,
                    "final_order_qty": 0,
                    "weeks_of_cover": safe_row["weeks_of_cover"],
                    "risk_flag": False,
                }
            )

    summary_df = pd.DataFrame(summary_rows)
    transfer_df = pd.DataFrame(transfer_rows)
    store_detail_df = pd.DataFrame(store_detail_rows)

    if not store_detail_df.empty:
        store_detail_df["stock_status"] = store_detail_df["risk_flag"].map(
            lambda x: "결품 위험" if x else "안정"
        )

    return summary_df, transfer_df, store_detail_df


# =========================================================
# 화면 컴포넌트
# =========================================================
def inject_css():
    st.markdown(
        """
        <style>
        div[data-testid="stMetric"] {
            background: linear-gradient(145deg, #0f172a 0%, #1e293b 100%);
            border: 1px solid #334155;
            border-radius: 10px;
            padding: 12px 14px;
        }
        div[data-testid="stMetric"] label { color: #94a3b8 !important; }
        div[data-testid="stMetric"] [data-testid="stMetricValue"] { color: #f8fafc !important; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def metric_row(summary_df: pd.DataFrame, store_detail_df: pd.DataFrame):
    total_sku = int(summary_df["sku"].nunique()) if not summary_df.empty else 0
    risky_sku = int((summary_df["additional_order_qty"] > 0).sum()) if not summary_df.empty else 0
    risky_store = (
        int(store_detail_df["risk_flag"].eq(True).sum()) if not store_detail_df.empty else 0
    )
    total_order = int(summary_df["recommended_order_qty"].sum()) if not summary_df.empty else 0

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("분석 대상 SKU", f"{total_sku:,}")
    c2.metric("추가 발주 필요 SKU", f"{risky_sku:,}")
    c3.metric("결품 위험 매장(건)", f"{risky_store:,}")
    c4.metric("총 추천 발주 수량", f"{total_order:,}")


def render_dashboard(summary_df: pd.DataFrame):
    st.subheader("전체 요약 · SKU 단위")
    st.caption("부족 → 매장 이동·센터 반영 후 **추가 발주**까지 한눈에 봅니다.")

    if summary_df.empty:
        st.info("표시할 요약 데이터가 없습니다.")
        return

    sort_option = st.selectbox(
        "정렬 기준",
        [
            "recommended_order_qty",
            "additional_order_qty",
            "risk_store_count",
            "total_shortage_before_action",
        ],
        index=0,
    )

    view_df = summary_df.sort_values(sort_option, ascending=False).copy()
    st.dataframe(view_df, use_container_width=True, hide_index=True)


def render_store_detail(store_detail_df: pd.DataFrame, summary_df: pd.DataFrame):
    st.subheader("매장 상세")
    st.caption("SKU별로 **왜 발주가 나왔는지** 매장 단위로 확인합니다.")

    if store_detail_df.empty:
        st.info("표시할 매장 상세 데이터가 없습니다.")
        return

    sku_options = ["전체"] + sorted(store_detail_df["sku"].dropna().unique().tolist())
    selected_sku = st.selectbox("SKU 선택", sku_options)

    filtered = store_detail_df.copy()
    if selected_sku != "전체":
        filtered = filtered[filtered["sku"] == selected_sku].copy()

    status_options = ["전체", "결품 위험", "안정"]
    selected_status = st.selectbox("상태", status_options)
    if selected_status != "전체":
        filtered = filtered[filtered["stock_status"] == selected_status].copy()

    st.dataframe(
        filtered.sort_values(["risk_flag", "final_order_qty"], ascending=[False, False]),
        use_container_width=True,
        hide_index=True,
    )

    if selected_sku != "전체":
        sku_summary = summary_df[summary_df["sku"] == selected_sku]
        if not sku_summary.empty:
            row = sku_summary.iloc[0]
            st.markdown("#### 선택 SKU 발주 요약")
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("총 부족(이동 전)", f"{int(row['total_shortage_before_action']):,}")
            m2.metric("매장간 이동 반영", f"{int(row['transfer_allocated_qty']):,}")
            m3.metric("물류센터 반영", f"{int(row['center_stock_used']):,}")
            m4.metric("최종 추천 발주", f"{int(row['recommended_order_qty']):,}")


def render_transfer_plan(transfer_df: pd.DataFrame):
    st.subheader("매장간 이동 계획")
    st.caption("실행 시 **출고 매장 → 입고 매장** 기준으로 활용하세요.")

    if transfer_df.empty:
        st.info("현재 기준으로 필요한 매장간 이동이 없습니다.")
        return

    st.dataframe(
        transfer_df.sort_values(["sku", "transfer_qty"], ascending=[True, False]),
        use_container_width=True,
        hide_index=True,
    )


def render_reorder_plan(summary_df: pd.DataFrame):
    st.subheader("추가 발주")
    st.caption("구매/발주용 **최종 수량 리스트**입니다.")

    if summary_df.empty:
        st.info("추가 발주 데이터가 없습니다.")
        return

    reorder_view = summary_df[summary_df["recommended_order_qty"] > 0].copy()
    if reorder_view.empty:
        st.success("현재 기준으로 추가 발주가 필요한 SKU가 없습니다.")
        return

    reorder_view = reorder_view.sort_values("recommended_order_qty", ascending=False)
    st.dataframe(
        reorder_view[
            [
                "sku",
                "sku_name",
                "style_code",
                "total_shortage_before_action",
                "transfer_allocated_qty",
                "center_stock_used",
                "additional_order_qty",
                "recommended_order_qty",
                "lead_time",
                "minimum_capacity",
                "factory",
            ]
        ],
        use_container_width=True,
        hide_index=True,
    )


# =========================================================
# 메인
# =========================================================
def main():
    inject_css()
    st.title("결품 예측 · 매장간 이동 · 추가 발주")
    st.caption(
        "① 매장별 결품(예측+손실 대비 가용재고) → ② 매장간 이동으로 메움 → "
        "③ `public.center_stock` 반영 → ④ **최종 추가 발주** 산출. "
        "안전재고·거리·리드타임 주차 환산 등은 미반영(확장 여지)."
    )

    try:
        raw = load_data()
    except Exception as e:
        st.error(f"Supabase 데이터 로드 실패: {e}")
        st.stop()

    weekly = prepare_weekly_forecast(raw["weekly"])
    center = prepare_center_stock(raw["center"])
    reorder = prepare_reorder(raw["reorder"])

    if weekly.empty:
        st.error(
            "`public.sku_weekly_forecast`에 분석 가능한 데이터가 없습니다. "
            "(`sku` 컬럼과 최소 1행이 필요합니다.)"
        )
        st.stop()

    with st.sidebar:
        st.markdown("### 필터 · 기준 데이터")
        if st.button("데이터 새로고침"):
            st.cache_data.clear()
            st.rerun()

        yw_vals = sorted([x for x in weekly["year_week"].dropna().unique().tolist() if str(x).strip()])
        if yw_vals:
            yw_choice = st.selectbox("기준 주차 (year_week)", ["전체"] + yw_vals, index=0)
        else:
            yw_choice = "전체"
            st.caption("`year_week` 값이 없어 전체 행을 합산합니다.")

        fc_only = st.checkbox("미래 예측만 (is_forecast=true)", value=False)
        if fc_only:
            weekly = weekly[weekly["is_forecast"] == True].copy()  # noqa: E712
            if weekly.empty:
                st.warning("`is_forecast=true` 행이 없어 필터를 적용하지 않았습니다.")
                weekly = prepare_weekly_forecast(raw["weekly"])

        if yw_choice != "전체":
            weekly = weekly[weekly["year_week"] == yw_choice].copy()

        all_styles = sorted([x for x in weekly["style_code"].dropna().unique().tolist() if x])
        selected_style = st.selectbox("스타일 코드", ["전체"] + all_styles)

        all_stores = sorted([x for x in weekly["store_key"].dropna().unique().tolist() if x])
        selected_store = st.selectbox("매장 (store_key)", ["전체"] + all_stores)

        risk_only = st.checkbox("결품 위험 SKU만 요약에 표시", value=False)

    filtered_weekly = weekly.copy()
    if selected_style != "전체":
        filtered_weekly = filtered_weekly[filtered_weekly["style_code"] == selected_style].copy()
    if selected_store != "전체":
        filtered_weekly = filtered_weekly[filtered_weekly["store_key"] == selected_store].copy()

    if filtered_weekly.empty:
        st.warning("필터 조건에 맞는 주간 예측 행이 없습니다. 사이드바를 조정하세요.")
        st.stop()

    store_risk_df = build_store_risk_table(filtered_weekly)
    summary_df, transfer_df, store_detail_df = allocate_transfer_and_center(store_risk_df, center, reorder)

    if risk_only and not summary_df.empty:
        skus_risk = summary_df.loc[summary_df["total_shortage_before_action"] > 0, "sku"].tolist()
        summary_df = summary_df[summary_df["total_shortage_before_action"] > 0].copy()
        store_detail_df = store_detail_df[store_detail_df["sku"].isin(skus_risk)].copy()
        transfer_df = transfer_df[transfer_df["sku"].isin(skus_risk)].copy()

    metric_row(summary_df, store_detail_df)

    tab1, tab2, tab3, tab4 = st.tabs(
        ["전체 요약", "매장 상세", "매장간 이동", "추가 발주"]
    )

    with tab1:
        render_dashboard(summary_df)
    with tab2:
        render_store_detail(store_detail_df, summary_df)
    with tab3:
        render_transfer_plan(transfer_df)
    with tab4:
        render_reorder_plan(summary_df)

    with st.expander("계산 정의 (요약)"):
        st.markdown(
            """
            - **expected_demand** = `sale_qty` 합 + `loss` 합  
            - **available_stock** = `begin_stock` + `inbound_qty` − `outbound_qty`  
            - **projected_ending_stock** = available − expected_demand  
            - **shortage / surplus** = projected 가 음수면 부족분·양수면 여유분  
            - SKU 안에서 여유 매장 → 부족 매장 순으로 이동 수량 배정 후, 남은 부족에 **센터 재고** 배정  
            - **additional_order_qty** = 총 부족 − 이동 합 − 센터 사용 합 (0 미만은 0)  
            - **recommended_order_qty** = MOQ(`minimum_capacity`)가 있으면 발주가 필요할 때만 max(추가발주, MOQ)
            """
        )


if __name__ == "__main__":
    main()
