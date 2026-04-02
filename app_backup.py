
import re
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from supabase import create_client


SUPABASE_URL = st.secrets["SUPABASE_URL"]
SUPABASE_KEY = st.secrets["SUPABASE_KEY"]
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)


# =========================
# 공통 유틸
# =========================
def first_existing_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    """Supabase/Postgres 컬럼 대소문자 차이를 흡수합니다."""
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


def clean_number(value):
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


def parse_year_month_to_timestamp(ym: str) -> pd.Timestamp:
    """year_month: '2025-01', '2025-03', '202503' 등."""
    s = str(ym).strip().replace(".", "")
    if re.match(r"^\d{6}$", s):
        s = f"{s[:4]}-{s[4:6]}"
    ts = pd.to_datetime(s + "-01", errors="coerce")
    return ts if pd.notna(ts) else pd.NaT


def iso_week_monday_month_day(year: int, week_no: int) -> Optional[Tuple[int, int]]:
    ts = pd.to_datetime(f"{year}-W{int(week_no):02d}-1", format="%G-W%V-%u", errors="coerce")
    if pd.isna(ts):
        return None
    return int(ts.month), int(ts.day)


def format_calendar_week_label(calendar_year: int, iso_week_no: int) -> str:
    ts = pd.to_datetime(f"{calendar_year}-W{int(iso_week_no):02d}-1", format="%G-W%V-%u", errors="coerce")
    if pd.isna(ts):
        return f"{iso_week_no}주차"
    yy = calendar_year % 100
    m = int(ts.month)
    week_in_month = (int(ts.day) - 1) // 7 + 1
    return f"{yy:02d}년 {m}월 {week_in_month}주차"


# =========================
# Supabase 로딩
# =========================
def load_supabase_table(table_name: str, page_size: int = 1000) -> pd.DataFrame:
    all_rows = []
    start = 0
    while True:
        end = start + page_size - 1
        res = (
            supabase.table(table_name)
            .select("*")
            .range(start, end)
            .execute()
        )
        rows = res.data or []
        if not rows:
            break
        all_rows.extend(rows)
        if len(rows) < page_size:
            break
        start += page_size
    return pd.DataFrame(all_rows)


@st.cache_data(ttl=300)
def load_sku_forecast_run_df() -> pd.DataFrame:
    return load_supabase_table("sku_forecast_run")


@st.cache_data(ttl=300)
def load_sku_weekly_forecast_df() -> pd.DataFrame:
    return load_supabase_table("sku_weekly_forecast")


@st.cache_data(ttl=300)
def load_sku_monthly_forecast_df() -> pd.DataFrame:
    return load_supabase_table("sku_monthly_forecast")


def infer_run_batch_key(runs_df: pd.DataFrame) -> str:
    """자식 테이블 forecast_run_id와 맞출 부모 쪽 키 컬럼명."""
    fr = first_existing_col(runs_df, ["forecast_run_id", "forecast_runid"])
    if fr and runs_df[fr].notna().any():
        return fr
    return "id"


def list_run_batches(
    runs_df: pd.DataFrame,
    weekly_df: pd.DataFrame,
) -> List[Tuple[object, pd.Timestamp, int]]:
    """
    (batch_key, 대표 run_date, 해당 배치 주간 행 수) 목록. 최신 run_date 우선.
    sku_forecast_run이 비어 있으면 weekly의 forecast_run_id만으로 배치를 만듭니다.
    """
    wk_fr = first_existing_col(weekly_df, ["forecast_run_id", "forecast_runid"])
    if weekly_df.empty or wk_fr is None:
        return []

    keys_in_weekly = weekly_df[wk_fr].dropna().astype(object).unique().tolist()
    parts: List[Tuple[object, pd.Timestamp, int]] = []

    if runs_df.empty:
        for k in keys_in_weekly:
            n_w = int((weekly_df[wk_fr] == k).sum())
            parts.append((k, pd.Timestamp.now(), n_w))
        parts.sort(key=lambda x: x[2], reverse=True)
        return parts

    parent_key = infer_run_batch_key(runs_df)
    rd_col = first_existing_col(runs_df, ["run_date", "rundate"])
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

    parts.sort(key=lambda x: (x[1], x[2]), reverse=True)
    return parts


def filter_by_run_key(df: pd.DataFrame, run_key_col: str, batch_key: object) -> pd.DataFrame:
    if df.empty or run_key_col not in df.columns:
        return pd.DataFrame()
    return df[df[run_key_col] == batch_key].copy()


STAGE_COLORS = {
    "도입": "#1f77b4",
    "성장": "#2ca02c",
    "피크": "#d62728",
    "피크2": "#d62728",
    "성숙": "#9467bd",
    "비시즌": "#7f7f7f",
    "쇠퇴": "#8c564b",
}


def build_dual_line_chart(
    title_name: str,
    weekly_df: pd.DataFrame,
    monthly_df: pd.DataFrame,
) -> go.Figure:
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=weekly_df["week_start"],
            y=weekly_df["sales"],
            mode="lines",
            name="주차별 예측(연결선)",
            line=dict(color="#b0b0b0", width=2),
            hoverinfo="skip",
            showlegend=False,
            connectgaps=True,
        )
    )

    stage_df = weekly_df.copy().reset_index(drop=True)
    stage_df["week_no"] = stage_df["week_start"].dt.isocalendar().week.astype(int)

    if "stage" in stage_df.columns:
        current_stage = None
        segment_x, segment_y, segment_week = [], [], []

        for _, row in stage_df.iterrows():
            stage = row["stage"]
            x, y, w = row["week_start"], row["sales"], int(row["week_no"])

            if current_stage is None:
                current_stage = stage
                segment_x, segment_y, segment_week = [x], [y], [w]
            elif stage == current_stage:
                segment_x.append(x)
                segment_y.append(y)
                segment_week.append(w)
            else:
                st_name = str(current_stage)
                fig.add_trace(
                    go.Scatter(
                        x=segment_x,
                        y=segment_y,
                        customdata=segment_week,
                        mode="lines+markers",
                        name=st_name,
                        line=dict(color=STAGE_COLORS.get(st_name, "#333"), width=3),
                        marker=dict(size=7),
                        hovertemplate=(
                            "주차: %{customdata}주차<br>주차 시작일: %{x|%Y-%m-%d}<br>예측 수량: %{y:,.0f}<br>단계: "
                            + st_name
                            + "<extra></extra>"
                        ),
                        showlegend=True,
                    )
                )
                current_stage = stage
                segment_x, segment_y, segment_week = [x], [y], [w]

        if segment_x:
            st_name = str(current_stage)
            fig.add_trace(
                go.Scatter(
                    x=segment_x,
                    y=segment_y,
                    customdata=segment_week,
                    mode="lines+markers",
                    name=st_name,
                    line=dict(color=STAGE_COLORS.get(st_name, "#333"), width=3),
                    marker=dict(size=7),
                    hovertemplate=(
                        "주차: %{customdata}주차<br>주차 시작일: %{x|%Y-%m-%d}<br>예측 수량: %{y:,.0f}<br>단계: "
                        + st_name
                        + "<extra></extra>"
                    ),
                    showlegend=True,
                )
            )

    fig.add_trace(
        go.Scatter(
            x=monthly_df["month"],
            y=monthly_df["sales"],
            customdata=monthly_df["month"].dt.isocalendar().week.astype(int),
            mode="lines+markers",
            name="월별 예측",
            line=dict(width=3, color="#bfbfbf"),
            marker=dict(size=7, color="#bfbfbf"),
            fill="tozeroy",
            fillcolor="rgba(191, 191, 191, 0.25)",
            connectgaps=True,
            yaxis="y2",
            hovertemplate="월: %{x|%Y-%m}<br>(참고) %{customdata}주차<br>예측 수량: %{y:,.0f}<extra></extra>",
        )
    )

    fig.update_layout(
        title=f"{title_name} 주차별 단계 / 월별 예측 추이",
        xaxis_title="날짜",
        yaxis_title="주차별 예측 수량",
        yaxis2=dict(title="월별 예측 수량", overlaying="y", side="right", showgrid=False),
        height=650,
        hovermode="x unified",
        margin=dict(l=30, r=30, t=70, b=30),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    fig.update_yaxes(tickformat=",.0f", rangemode="tozero")
    fig.update_layout(yaxis=dict(rangemode="tozero"), yaxis2=dict(rangemode="tozero"))
    return fig


def weekly_rows_to_timeseries(weekly_slice: pd.DataFrame) -> pd.DataFrame:
    """sku_weekly_forecast 행 → week_start, sales(=forecast_qty), stage."""
    yw = first_existing_col(weekly_slice, ["year_week", "yearweek"])
    fq = first_existing_col(weekly_slice, ["forecast_qty", "forecastqty"])
    stg = first_existing_col(weekly_slice, ["stage"])
    if not yw or not fq:
        return pd.DataFrame(columns=["week_start", "sales", "stage"])

    rows = []
    for _, r in weekly_slice.iterrows():
        ws = parse_yearweek_to_date(r[yw])
        if pd.isna(ws):
            continue
        qty = clean_number(r[fq])
        rows.append(
            {
                "week_start": ws,
                "sales": 0.0 if pd.isna(qty) else float(qty),
                "stage": str(r[stg]).strip() if stg and pd.notna(r.get(stg)) else "",
            }
        )
    out = pd.DataFrame(rows).sort_values("week_start").reset_index(drop=True)
    return out


def monthly_rows_to_timeseries(monthly_slice: pd.DataFrame) -> pd.DataFrame:
    ym = first_existing_col(monthly_slice, ["year_month", "yearmonth"])
    fq = first_existing_col(monthly_slice, ["forecast_qty", "forecastqty"])
    if not ym or not fq:
        return pd.DataFrame(columns=["month", "sales"])

    rows = []
    for _, r in monthly_slice.iterrows():
        m = parse_year_month_to_timestamp(r[ym])
        if pd.isna(m):
            continue
        qty = clean_number(r[fq])
        rows.append({"month": m, "sales": 0.0 if pd.isna(qty) else float(qty)})
    return pd.DataFrame(rows).sort_values("month").reset_index(drop=True)


def build_forecast_weekly_table(
    weekly_ts: pd.DataFrame,
    sku: str,
    sku_label: str,
    label_year: int,
) -> pd.DataFrame:
    """주차별 예측 비중·수량·단계 표 (DB 예측만 사용)."""
    if weekly_ts.empty:
        return pd.DataFrame()

    df = weekly_ts.copy()
    df["week_no"] = df["week_start"].dt.isocalendar().week.astype(int)
    total = float(df["sales"].sum())
    if total > 0:
        df["ratio_pct"] = df["sales"] / total * 100.0
    else:
        df["ratio_pct"] = 0.0

    df["주차"] = df["week_no"].astype(int).map(lambda w: format_calendar_week_label(label_year, int(w)))
    df["예측 단계"] = df["stage"].replace({"피크": "성숙", "피크2": "성숙"}).fillna("").astype(str)

    out = df.rename(columns={"sales": "주차별 예측 수량"}).copy()
    out["SKU"] = sku
    out["SKU_NAME"] = sku_label
    out["주차별 예측 비중(%)"] = out["ratio_pct"].round(1)
    return out[
        ["SKU", "SKU_NAME", "week_no", "주차", "주차별 예측 비중(%)", "주차별 예측 수량", "예측 단계"]
    ].reset_index(drop=True)


def get_run_meta_for_sku(
    runs_df: pd.DataFrame,
    batch_key: object,
    sku: str,
    plant: Optional[str],
) -> Tuple[str, str]:
    """(shape_type, shape_reason) — sku_forecast_run 행 매칭."""
    parent_key = infer_run_batch_key(runs_df)
    sku_c = first_existing_col(runs_df, ["SKU", "sku"])
    pl_c = first_existing_col(runs_df, ["plant", "PLANT"])
    st_c = first_existing_col(runs_df, ["shape_type", "shapetype"])
    sr_c = first_existing_col(runs_df, ["shape_reason", "shapereason"])

    if not sku_c:
        return "—", "sku_forecast_run에 SKU 컬럼이 없습니다."

    m = runs_df[runs_df[parent_key] == batch_key].copy()
    if m.empty:
        m = runs_df.copy()

    m = m[m[sku_c].astype(str).str.strip() == str(sku).strip()]
    if plant and pl_c and pl_c in m.columns:
        m = m[m[pl_c].astype(str).str.strip() == str(plant).strip()]

    if m.empty:
        return "—", "해당 배치·SKU의 sku_forecast_run 메타를 찾지 못했습니다."

    row = m.iloc[0]
    sl = str(row[st_c]).strip() if st_c and pd.notna(row.get(st_c)) else "—"
    reason = str(row[sr_c]).strip() if sr_c and pd.notna(row.get(sr_c)) else ""
    return sl, reason or "(사유 없음)"


# =========================
# 메인 화면
# =========================
def main():
    st.set_page_config(page_title="SKU 예측 대시보드", layout="wide")

    try:
        runs_df = load_sku_forecast_run_df()
        weekly_df = load_sku_weekly_forecast_df()
        monthly_df = load_sku_monthly_forecast_df()
    except Exception as e:
        st.error(f"Supabase 테이블을 불러오지 못했습니다: {e}")
        return

    wk_fr = first_existing_col(weekly_df, ["forecast_run_id", "forecast_runid"])
    mo_fr = first_existing_col(monthly_df, ["forecast_run_id", "forecast_runid"])

    if weekly_df.empty:
        st.warning("sku_weekly_forecast 테이블에 데이터가 없습니다.")
        return
    if wk_fr is None:
        st.warning("sku_weekly_forecast에 forecast_run_id 컬럼이 없습니다.")
        return

    batches = list_run_batches(runs_df, weekly_df)
    if not batches:
        st.warning(
            "예측 배치를 식별할 수 없습니다. sku_weekly_forecast.forecast_run_id와 "
            "sku_forecast_run의 id 또는 forecast_run_id가 일치하는지 확인하세요."
        )
        return

    batch_labels = {
        str(k): f"{pd.Timestamp(rd).strftime('%Y-%m-%d %H:%M')} · batch={k} · 주간행 {n}건"
        for k, rd, n in batches
    }
    batch_keys_ordered = [b[0] for b in batches]

    st.sidebar.markdown("### 예측 배치")
    selected_batch_str = st.sidebar.selectbox(
        "실행 배치 선택",
        options=[batch_labels[str(k)] for k in batch_keys_ordered],
        index=0,
    )
    inv_lbl = {v: k for k, v in batch_labels.items()}
    selected_batch_key = inv_lbl[selected_batch_str]
    try:
        selected_batch_key = type(batch_keys_ordered[0])(selected_batch_key)
    except (ValueError, TypeError, IndexError):
        pass

    weekly_run = filter_by_run_key(weekly_df, wk_fr, selected_batch_key)
    monthly_run = (
        filter_by_run_key(monthly_df, mo_fr, selected_batch_key) if mo_fr else pd.DataFrame()
    )

    sku_w = first_existing_col(weekly_run, ["sku", "SKU"])
    pl_w = first_existing_col(weekly_run, ["plant", "PLANT"])
    sty_w = first_existing_col(weekly_run, ["sty", "style_code", "stylecode"])

    if not sku_w:
        st.warning("sku_weekly_forecast에서 sku 컬럼을 찾을 수 없습니다.")
        return

    opt_rows = (
        weekly_run[[c for c in [sku_w, pl_w, sty_w] if c]]
        .drop_duplicates()
        .sort_values([sku_w] + ([pl_w] if pl_w else []) + ([sty_w] if sty_w else []))
        .reset_index(drop=True)
    )

    if pl_w:
        opt_rows["plant_name"] = (
            opt_rows[pl_w].fillna("").astype(str).str.strip().replace("", "전체")
        )
    else:
        opt_rows["plant_name"] = "전체"
    opt_rows["style_code"] = (
        opt_rows[sty_w].astype(str).str.strip() if sty_w else opt_rows[sku_w].astype(str).str.slice(0, 10)
    )
    opt_rows["sku"] = opt_rows[sku_w].astype(str).str.strip()
    opt_rows["sku_name"] = opt_rows["sku"]
    opt_rows["option_id"] = opt_rows.apply(
        lambda r: f"{r['plant_name']}||{r['sku']}",
        axis=1,
    )
    opt_rows["display_label"] = opt_rows.apply(
        lambda r: f"{r['sku_name']} | 매장:{r['plant_name']} | 스타일:{r['style_code']}",
        axis=1,
    )

    if opt_rows.empty:
        st.warning("선택한 배치에 SKU 행이 없습니다.")
        return

    col_a, col_b, col_c = st.columns([1, 1, 2])

    with col_a:
        plants = sorted(opt_rows["plant_name"].dropna().astype(str).unique().tolist())
        plant_options = ["전체"] + [p for p in plants if p and p != "전체"]
        if not plant_options:
            plant_options = ["전체"]
        selected_plant = st.selectbox("매장 선택", options=plant_options)

    pf = opt_rows.copy()
    if selected_plant != "전체":
        pf = pf[pf["plant_name"] == selected_plant].copy()

    with col_b:
        styles = sorted(pf["style_code"].dropna().astype(str).str.strip().unique().tolist())
        style_options = ["전체"] + [s for s in styles if s]
        selected_style = st.selectbox("스타일(sty / 앞 10자)", options=style_options)

    with col_c:
        fo = pf.copy()
        if selected_style != "전체":
            fo = fo[fo["style_code"].astype(str).str.strip() == selected_style].copy()
        if fo.empty:
            st.warning("선택한 매장·스타일에 해당하는 상품이 없습니다.")
            return
        oid = st.selectbox(
            "상품 선택",
            options=fo["option_id"].tolist(),
            format_func=lambda x: fo.loc[fo["option_id"] == x, "display_label"].iloc[0],
        )

    sel = fo[fo["option_id"] == oid].iloc[0]
    selected_sku = str(sel["sku"]).strip()
    selected_plant_val = str(sel["plant_name"]).strip()
    plant_for_filter = None if selected_plant_val in ("", "전체") else selected_plant_val

    w_slice = weekly_run[weekly_run[sku_w].astype(str).str.strip() == selected_sku]
    if pl_w and plant_for_filter:
        w_slice = w_slice[w_slice[pl_w].astype(str).str.strip() == plant_for_filter]

    m_slice = pd.DataFrame()
    sku_m = first_existing_col(monthly_run, ["sku", "SKU"]) if not monthly_run.empty else None
    pl_m = first_existing_col(monthly_run, ["plant", "PLANT"]) if not monthly_run.empty else None
    if mo_fr and not monthly_run.empty and sku_m:
        m_slice = monthly_run[monthly_run[sku_m].astype(str).str.strip() == selected_sku]
        if pl_m and plant_for_filter:
            m_slice = m_slice[m_slice[pl_m].astype(str).str.strip() == plant_for_filter]

    weekly_ts = weekly_rows_to_timeseries(w_slice)
    monthly_ts = monthly_rows_to_timeseries(m_slice)

    if weekly_ts.empty:
        st.warning("선택한 SKU의 주간 예측 데이터가 없습니다.")
        return

    if monthly_ts.empty:
        monthly_ts = (
            weekly_ts.assign(month=weekly_ts["week_start"].dt.to_period("M").dt.to_timestamp())
            .groupby("month", as_index=False)["sales"]
            .sum()
            .sort_values("month")
            .reset_index(drop=True)
        )

    shape_label, shape_reason = get_run_meta_for_sku(
        runs_df, selected_batch_key, selected_sku, plant_for_filter
    )

    label_year = int(pd.Timestamp.today().year)
    compare_table_df = build_forecast_weekly_table(
        weekly_ts, selected_sku, selected_sku, label_year
    )

    st.markdown("### 주차별 예측 비중 · 예측 수량")
    st.dataframe(
        compare_table_df.drop(columns=["SKU", "SKU_NAME"], errors="ignore"),
        use_container_width=True,
        hide_index=True,
        column_config={
            "주차별 예측 비중(%)": st.column_config.NumberColumn(
                "주차별 예측 비중(%)",
                format="%.2f%%",
            ),
            "주차별 예측 수량": st.column_config.NumberColumn(
                "주차별 예측 수량",
                format="%.0f",
            ),
        },
    )

    st.markdown(f"### SKU: **{selected_sku}**")
    st.markdown(f"### 형태(메타): **{shape_label}**")
    st.caption(shape_reason)

    if shape_label in ("단봉형",):
        st.markdown("**참고 단계 순서(단봉):** 도입 > 성장 > 피크 > 성숙 > 쇠퇴")
    elif shape_label in ("쌍봉형",):
        st.markdown("**참고 단계 순서(쌍봉):** 도입 > 성장 > 피크 > 성숙 > 비시즌 > 성숙 > 피크2 > 성숙 > 쇠퇴")
    elif shape_label not in ("—", "판단불가"):
        st.markdown("**참고 단계 순서(올시즌 등):** 도입 > 성장 > 성숙 > 쇠퇴")

    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown("### 주차·월별 예측 추이")
        fig1 = build_dual_line_chart(selected_sku, weekly_ts, monthly_ts)
        st.plotly_chart(fig1, use_container_width=True)

    with col2:
        st.markdown("### 주차별 예측 수량")
        fig2 = go.Figure()
        tw = weekly_ts.copy()
        tw["week_no"] = tw["week_start"].dt.isocalendar().week.astype(int)
        fig2.add_trace(
            go.Scatter(
                x=tw["week_start"],
                y=tw["sales"],
                customdata=tw["week_no"].values.reshape(-1, 1),
                name="주간 예측",
                mode="lines+markers",
                hovertemplate=(
                    "주차: %{customdata[0]}주차<br>날짜: %{x|%Y-%m-%d}<br>예측 수량: %{y:,.0f}<extra></extra>"
                ),
            )
        )
        y0, y1 = tw["week_start"].min(), tw["week_start"].max()
        fig2.update_layout(
            title=f"{selected_sku} 주차별 예측 수량",
            xaxis_title="날짜",
            yaxis_title="예측 수량",
            height=650,
            hovermode="x unified",
            xaxis=dict(range=[y0, y1] if pd.notna(y0) else None),
            yaxis=dict(rangemode="tozero"),
        )
        st.plotly_chart(fig2, use_container_width=True)


if __name__ == "__main__":
    main()
