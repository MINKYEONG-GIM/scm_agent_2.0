import math
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import streamlit as st
from supabase import create_client


# =========================================================
# 기본 설정
# =========================================================
SUPABASE_URL = st.secrets["SUPABASE_URL"]
SUPABASE_KEY = st.secrets["SUPABASE_KEY"]
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

st.set_page_config(
    page_title="결품 예측 · 매장간 이동 · 추가 발주 대시보드",
    layout="wide",
)


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
        "run": load_table("sku_forecast_run"),
        "weekly": load_table("sku_weekly_forecast"),
        "center": load_table("center_stock"),
        "reorder": load_table("reorder"),
    }


# =========================================================
# 데이터 정규화
# =========================================================
def prepare_weekly_forecast(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(
            columns=[
                "sku", "store_name", "plant", "sku_name", "style_code", "stage",
                "sale_qty", "begin_stock", "loss", "inbound_qty", "outbound_qty",
                "avg_discount_rate", "is_forecast", "is_peak_week", "year_week"
            ]
        )

    col_sku = first_existing_col(df, ["sku", "SKU"])
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
    col_year_week = first_existing_col(df, ["year_week"])

    out = pd.DataFrame()
    out["sku"] = df[col_sku].map(normalize_text) if col_sku else ""
    out["store_name"] = df[col_store].map(normalize_text) if col_store else ""
    out["plant"] = df[col_plant].map(normalize_text) if col_plant else ""
    out["sku_name"] = df[col_sku_name].map(normalize_text) if col_sku_name else out["sku"]
    out["style_code"] = df[col_style].map(normalize_text) if col_style else ""
    out["stage"] = df[col_stage].map(normalize_text) if col_stage else ""
    out["sale_qty"] = df[col_sale].map(lambda x: to_number(x, 0.0)) if col_sale else 0.0
    out["begin_stock"] = df[col_begin].map(lambda x: to_number(x, 0.0)) if col_begin else 0.0
    out["loss"] = df[col_loss].map(lambda x: to_number(x, 0.0)) if col_loss else 0.0
    out["inbound_qty"] = df[col_inbound].map(lambda x: to_number(x, 0.0)) if col_inbound else 0.0
    out["outbound_qty"] = df[col_outbound].map(lambda x: to_number(x, 0.0)) if col_outbound else 0.0
    out["avg_discount_rate"] = df[col_discount].map(lambda x: to_number(x, 0.0)) if col_discount else 0.0
    out["is_forecast"] = df[col_forecast].fillna(False) if col_forecast else False
    out["is_peak_week"] = df[col_peak].fillna(False) if col_peak else False
    out["year_week"] = df[col_year_week].map(normalize_text) if col_year_week else ""

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

    out = pd.DataFrame()
    out["sku"] = df[col_sku].map(normalize_text) if col_sku else ""
    out["center"] = df[col_center].map(normalize_text) if col_center else ""
    out["stock_qty"] = df[col_stock].map(lambda x: to_number(x, 0.0)) if col_stock else 0.0
    out["style_code"] = df[col_style].map(normalize_text) if col_style else ""
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

    out = pd.DataFrame()
    out["sku"] = df[col_sku].map(normalize_text) if col_sku else ""
    out["factory"] = df[col_factory].map(normalize_text) if col_factory else ""
    out["lead_time"] = df[col_lead].map(lambda x: to_number(x, 0.0)) if col_lead else 0.0
    out["minimum_capacity"] = df[col_min].map(lambda x: to_number(x, 0.0)) if col_min else 0.0
    out["style_code"] = df[col_style].map(normalize_text) if col_style else ""
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
    grouped["available_stock_before_transfer"] = grouped["begin_stock"] + grouped["inbound_qty"] - grouped["outbound_qty"]
    grouped["projected_ending_stock"] = grouped["available_stock_before_transfer"] - grouped["expected_demand"]
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
    center_stock_map = center_df.groupby("sku", dropna=False)["stock_qty"].sum().to_dict() if not center_df.empty else {}
    reorder_info = reorder_df.groupby("sku", dropna=False).agg(
        lead_time=("lead_time", "max"),
        minimum_capacity=("minimum_capacity", "max"),
        factory=("factory", "first"),
    ).reset_index() if not reorder_df.empty else pd.DataFrame(columns=["sku", "lead_time", "minimum_capacity", "factory"])

    summary_rows: List[Dict[str, Any]] = []
    transfer_rows: List[Dict[str, Any]] = []
    store_detail_rows: List[Dict[str, Any]] = []

    for sku, sku_df in store_df.groupby("sku", dropna=False):
        sku_df = sku_df.copy().reset_index(drop=True)
        sku_name = sku_df["sku_name"].iloc[0]
        style_code = sku_df["style_code"].iloc[0]

        donors = sku_df[sku_df["surplus_qty"] > 0].copy().sort_values("surplus_qty", ascending=False)
        receivers = sku_df[sku_df["shortage_qty"] > 0].copy().sort_values("shortage_qty", ascending=False)

        donor_pool = {idx: float(row["surplus_qty"]) for idx, row in donors.iterrows()}
        transfer_total = 0.0

        for recv_idx, recv_row in receivers.iterrows():
            needed = float(recv_row["shortage_qty"])
            recv_store = recv_row["store_key"]

            if needed <= 0:
                continue

            for donor_idx, donor_row in donors.iterrows():
                available = donor_pool.get(donor_idx, 0.0)
                if available <= 0 or needed <= 0:
                    continue

                move_qty = min(available, needed)
                donor_pool[donor_idx] -= move_qty
                needed -= move_qty
                transfer_total += move_qty

                transfer_rows.append({
                    "sku": sku,
                    "sku_name": sku_name,
                    "style_code": style_code,
                    "from_store": donor_row["store_key"],
                    "to_store": recv_store,
                    "transfer_qty": int(round(move_qty)),
                })

            center_available = float(center_stock_map.get(sku, 0.0))
            center_alloc = min(center_available, needed)
            center_stock_map[sku] = center_available - center_alloc
            needed -= center_alloc

            final_order_qty = max(0.0, needed)

            store_detail_rows.append({
                "sku": sku,
                "sku_name": sku_name,
                "style_code": style_code,
                "store_name": recv_store,
                "plant": recv_row["plant"],
                "begin_stock": int(round(recv_row["begin_stock"])),
                "forecast_sales": int(round(recv_row["forecast_sales"])),
                "loss_qty": int(round(recv_row["loss_qty"])),
                "shortage_before_action": int(round(recv_row["shortage_qty"])),
                "transfer_in_qty": int(round(recv_row["shortage_qty"] - needed - center_alloc if recv_row["shortage_qty"] - needed - center_alloc > 0 else recv_row["shortage_qty"] - max(needed + center_alloc, 0))),
                "center_alloc_qty": int(round(center_alloc)),
                "final_order_qty": int(round(final_order_qty)),
                "weeks_of_cover": recv_row["weeks_of_cover"],
                "risk_flag": True,
            })

        sku_center_remaining = float(center_stock_map.get(sku, 0.0))
        sku_center_initial = float(center_df[center_df["sku"] == sku]["stock_qty"].sum()) if not center_df.empty else 0.0
        total_shortage = float(sku_df["shortage_qty"].sum())
        total_surplus = float(sku_df["surplus_qty"].sum())

        used_center = sku_center_initial - sku_center_remaining
        additional_order_qty = max(0.0, total_shortage - transfer_total - used_center)

        reorder_row = reorder_info[reorder_info["sku"] == sku]
        lead_time = float(reorder_row["lead_time"].iloc[0]) if not reorder_row.empty else 0.0
        minimum_capacity = float(reorder_row["minimum_capacity"].iloc[0]) if not reorder_row.empty else 0.0
        factory = str(reorder_row["factory"].iloc[0]) if not reorder_row.empty else ""

        recommended_order_qty = additional_order_qty
        if minimum_capacity > 0 and recommended_order_qty > 0:
            recommended_order_qty = max(recommended_order_qty, minimum_capacity)

        summary_rows.append({
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
        })

        safe_rows = sku_df[sku_df["shortage_qty"] <= 0].copy()
        for _, safe_row in safe_rows.iterrows():
            store_detail_rows.append({
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
            })

    summary_df = pd.DataFrame(summary_rows)
    transfer_df = pd.DataFrame(transfer_rows)
    store_detail_df = pd.DataFrame(store_detail_rows)

    if not store_detail_df.empty:
        store_detail_df["stock_status"] = store_detail_df["risk_flag"].map(lambda x: "결품 위험" if x else "안정")

    return summary_df, transfer_df, store_detail_df


# =========================================================
# 화면 컴포넌트
# =========================================================
def metric_row(summary_df: pd.DataFrame, store_detail_df: pd.DataFrame):
    total_sku = int(summary_df["sku"].nunique()) if not summary_df.empty else 0
    risky_sku = int((summary_df["additional_order_qty"] > 0).sum()) if not summary_df.empty else 0
    risky_store = int((store_detail_df["risk_flag"] == True).sum()) if not store_detail_df.empty else 0
    total_order = int(summary_df["recommended_order_qty"].sum()) if not summary_df.empty else 0

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("분석 대상 SKU", f"{total_sku:,}")
    c2.metric("추가 발주 필요 SKU", f"{risky_sku:,}")
    c3.metric("결품 위험 매장", f"{risky_store:,}")
    c4.metric("총 추천 발주 수량", f"{total_order:,}")


def render_dashboard(summary_df: pd.DataFrame):
    st.subheader("SKU 요약")

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
    st.dataframe(
        view_df,
        use_container_width=True,
        hide_index=True,
    )


def render_store_detail(store_detail_df: pd.DataFrame, summary_df: pd.DataFrame):
    st.subheader("매장 상세")

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

    st.dataframe(filtered.sort_values(["risk_flag", "final_order_qty"], ascending=[False, False]), use_container_width=True, hide_index=True)

    if selected_sku != "전체":
        sku_summary = summary_df[summary_df["sku"] == selected_sku]
        if not sku_summary.empty:
            row = sku_summary.iloc[0]
            st.markdown("#### 선택한 SKU 발주 요약")
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("결품 부족 수량", f"{int(row['total_shortage_before_action']):,}")
            m2.metric("매장간 이동 반영", f"{int(row['transfer_allocated_qty']):,}")
            m3.metric("물류센터 반영", f"{int(row['center_stock_used']):,}")
            m4.metric("최종 추천 발주", f"{int(row['recommended_order_qty']):,}")


def render_transfer_plan(transfer_df: pd.DataFrame):
    st.subheader("매장간 이동 계획")

    if transfer_df.empty:
        st.info("현재 계산 기준으로 필요한 매장간 이동 계획이 없습니다.")
        return

    st.dataframe(
        transfer_df.sort_values(["sku", "transfer_qty"], ascending=[True, False]),
        use_container_width=True,
        hide_index=True,
    )


def render_reorder_plan(summary_df: pd.DataFrame):
    st.subheader("추가 발주 계획")

    if summary_df.empty:
        st.info("추가 발주 데이터가 없습니다.")
        return

    reorder_df = summary_df[summary_df["recommended_order_qty"] > 0].copy()
    if reorder_df.empty:
        st.success("현재 계산 기준으로 추가 발주가 필요한 SKU가 없습니다.")
        return

    reorder_df = reorder_df.sort_values("recommended_order_qty", ascending=False)
    st.dataframe(
        reorder_df[
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
    st.title("결품 예측 · 매장간 이동 · 추가 발주 웹")
    st.caption("SKU별, 매장별 결품 위험을 계산하고 매장간 이동 가능 수량과 물류센터 재고를 먼저 반영한 뒤 최종 추가 발주 수량을 계산합니다.")

    raw = load_data()
    weekly = prepare_weekly_forecast(raw["weekly"])
    center = prepare_center_stock(raw["center"])
    reorder = prepare_reorder(raw["reorder"])

    if weekly.empty:
        st.error("sku_weekly_forecast 테이블에 분석 가능한 데이터가 없습니다.")
        st.stop()

    with st.sidebar:
        st.markdown("### 필터")
        all_styles = sorted([x for x in weekly["style_code"].dropna().unique().tolist() if x])
        selected_style = st.selectbox("스타일 코드", ["전체"] + all_styles)

        all_stores = sorted([x for x in weekly["store_key"].dropna().unique().tolist() if x])
        selected_store = st.selectbox("매장", ["전체"] + all_stores)

        risk_only = st.checkbox("결품 위험만 보기", value=False)

    filtered_weekly = weekly.copy()
    if selected_style != "전체":
        filtered_weekly = filtered_weekly[filtered_weekly["style_code"] == selected_style].copy()
    if selected_store != "전체":
        filtered_weekly = filtered_weekly[filtered_weekly["store_key"] == selected_store].copy()

    store_risk_df = build_store_risk_table(filtered_weekly)
    summary_df, transfer_df, store_detail_df = allocate_transfer_and_center(store_risk_df, center, reorder)

    if risk_only:
        summary_df = summary_df[summary_df["total_shortage_before_action"] > 0].copy()
        store_detail_df = store_detail_df[store_detail_df["risk_flag"] == True].copy()

    metric_row(summary_df, store_detail_df)

    tab1, tab2, tab3, tab4 = st.tabs([
        "전체 요약",
        "매장 상세",
        "매장간 이동",
        "추가 발주",
    ])

    with tab1:
        render_dashboard(summary_df)

    with tab2:
        render_store_detail(store_detail_df, summary_df)

    with tab3:
        render_transfer_plan(transfer_df)

    with tab4:
        render_reorder_plan(summary_df)


if __name__ == "__main__":
    main()
