from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from engine.benchmark import (
    BENCHMARK_RATINGS,
    MMD_BUCKET_MAP,
    detect_mmd_date_column as _detect_mmd_date_column,
    get_benchmark_curve,
)
from engine.scoring import add_workflow_spread_bps as _add_workflow_spread_bps
from ui.common import safe_dataframe, safe_plotly_chart, section_anchor


def render_focused_core_charts(
    market_df: pd.DataFrame,
    issuer_trades: pd.DataFrame,
    mmd_df: pd.DataFrame,
    selected_issuer: str,
    comparison_issuers: list[str],
    selected_sector: str,
):
    section_anchor("workflow-core-charts", "Core Charts")
    st.markdown(
        "<div class='focus-band'>Core visual analysis only: spread trend, yield trend, trading volume, and maturity-year curve. Use this page when you want charts without the long audit/admin sections.</div>",
        unsafe_allow_html=True,
    )
    if market_df.empty or "issuer" not in market_df.columns:
        st.info("No uploaded market data is available for charts.")
        return

    all_issuers = sorted(market_df["issuer"].dropna().astype(str).unique().tolist())
    chart_issuers = [selected_issuer] + [x for x in comparison_issuers if x != selected_issuer]
    chart_issuers = [x for x in chart_issuers if x in all_issuers]
    if not chart_issuers:
        chart_issuers = [selected_issuer]

    chart_base_all = _add_workflow_spread_bps(market_df.copy())
    chart_base_all["trade_date"] = pd.to_datetime(chart_base_all.get("trade_date"), errors="coerce")
    if "trade_amount" in chart_base_all.columns:
        chart_base_all["trade_amount"] = pd.to_numeric(chart_base_all["trade_amount"], errors="coerce").fillna(0)
    else:
        chart_base_all["trade_amount"] = 0
    if "yield" in chart_base_all.columns:
        chart_base_all["yield"] = pd.to_numeric(chart_base_all["yield"], errors="coerce")

    chart_dates = chart_base_all["trade_date"].dropna() if "trade_date" in chart_base_all.columns else pd.Series(dtype="datetime64[ns]")
    if chart_dates.empty:
        st.warning("Core charts require valid trade_date values.")
        return

    ctrl1, ctrl2, ctrl3 = st.columns([1, 1, 1.4])
    with ctrl1:
        trend_frequency = st.selectbox(
            "Trend Frequency",
            ["Daily", "Weekly", "Monthly"],
            index=1,
            key="focused_core_trend_frequency",
        )
    with ctrl2:
        curve_benchmark_rating = st.selectbox(
            "Curve Benchmark",
            BENCHMARK_RATINGS,
            index=0,
            key="focused_core_curve_benchmark",
        )
    with ctrl3:
        reference_lines = st.multiselect(
            "Reference Lines",
            ["Sector median", "All uploaded median", "AAA/MMD baseline", "MMD benchmark curve"],
            default=["Sector median", "All uploaded median"],
            key="focused_core_reference_lines",
            help="MMD benchmark curve applies to the issuer curve when benchmark data is available. AAA/MMD baseline is 0 bps on spread charts.",
        )

    date_min = chart_dates.min().date()
    date_max = chart_dates.max().date()
    selected_chart_dates = st.date_input(
        "Chart Date Range",
        value=(date_min, date_max),
        min_value=date_min,
        max_value=date_max,
        key="focused_core_chart_date_range",
        help="Filters the focused core charts. Sidebar zoom only changes the visual viewport; this changes the chart data.",
    )
    chart_start_date, chart_end_date = date_min, date_max
    if isinstance(selected_chart_dates, tuple) and len(selected_chart_dates) == 2:
        chart_start_date, chart_end_date = selected_chart_dates
        chart_base_all = chart_base_all[
            (chart_base_all["trade_date"].dt.date >= chart_start_date)
            & (chart_base_all["trade_date"].dt.date <= chart_end_date)
        ].copy()

    visible_charts = st.multiselect(
        "Visible Chart Modules",
        ["Spread Trend", "Volume & Activity", "Issuer Curve"],
        default=["Spread Trend"],
        key="focused_core_visible_charts",
        help="Keep the default short while reviewing. Add volume or curve only when that evidence is needed.",
    )

    freq_map = {"Daily": "D", "Weekly": "W", "Monthly": "M"}
    period_freq = freq_map.get(trend_frequency, "W")
    chart_base = chart_base_all[chart_base_all["issuer"].astype(str).isin(chart_issuers)].copy()

    if not visible_charts:
        st.info("Select at least one chart module to display.")
        return

    if "Spread Trend" in visible_charts:
        st.subheader("Spread Trend")
    if "Spread Trend" in visible_charts and not chart_base.empty and {"trade_date", "spread_bps", "issuer"}.issubset(chart_base.columns):
        spread_points = chart_base.dropna(subset=["trade_date", "spread_bps"]).copy()
        spread_trend = (
            spread_points.groupby([pd.Grouper(key="trade_date", freq=period_freq), "issuer"], as_index=False)
            .agg(
                spread_bps=("spread_bps", "median"),
                avg_yield=("yield", "mean") if "yield" in spread_points.columns else ("spread_bps", "count"),
                trade_count=("spread_bps", "count"),
                total_par=("trade_amount", "sum"),
            )
            .dropna(subset=["spread_bps"])
            .sort_values("trade_date")
        )
        fig = go.Figure()
        for issuer_name in chart_issuers:
            tmp = spread_trend[spread_trend["issuer"].astype(str) == str(issuer_name)]
            if tmp.empty:
                continue
            fig.add_trace(
                go.Scatter(
                    x=tmp["trade_date"],
                    y=tmp["spread_bps"],
                    mode="lines+markers",
                    name=issuer_name,
                    line=dict(width=3.2 if issuer_name == selected_issuer else 2.1),
                    customdata=np.stack(
                        [
                            tmp["trade_count"].fillna(0),
                            tmp["total_par"].fillna(0),
                            tmp["avg_yield"].fillna(np.nan),
                        ],
                        axis=-1,
                    ),
                    hovertemplate=(
                        "%{x|%m/%d/%Y}<br>"
                        "Spread: %{y:.1f} bps<br>"
                        "Trades: %{customdata[0]:,.0f}<br>"
                        "Par: $%{customdata[1]:,.0f}<br>"
                        "Avg yield: %{customdata[2]:.3f}%"
                        "<extra>%{fullData.name}</extra>"
                    ),
                )
            )

        reference_base = chart_base_all.dropna(subset=["trade_date", "spread_bps"]).copy()
        if "Sector median" in reference_lines and selected_sector and selected_sector != "Unknown" and "sector" in reference_base.columns:
            sector_ref = reference_base[reference_base["sector"].astype(str) == str(selected_sector)].copy()
            sector_ref = (
                sector_ref.groupby(pd.Grouper(key="trade_date", freq=period_freq), as_index=False)
                .agg(spread_bps=("spread_bps", "median"), trade_count=("spread_bps", "count"))
                .dropna(subset=["spread_bps"])
            )
            if not sector_ref.empty:
                fig.add_trace(
                    go.Scatter(
                        x=sector_ref["trade_date"],
                        y=sector_ref["spread_bps"],
                        mode="lines",
                        name=f"{selected_sector} median",
                        line=dict(width=2, dash="dash"),
                        hovertemplate="%{x|%m/%d/%Y}<br>Spread: %{y:.1f} bps<extra>%{fullData.name}</extra>",
                    )
                )
        if "All uploaded median" in reference_lines:
            all_ref = (
                reference_base.groupby(pd.Grouper(key="trade_date", freq=period_freq), as_index=False)
                .agg(spread_bps=("spread_bps", "median"), trade_count=("spread_bps", "count"))
                .dropna(subset=["spread_bps"])
            )
            if not all_ref.empty:
                fig.add_trace(
                    go.Scatter(
                        x=all_ref["trade_date"],
                        y=all_ref["spread_bps"],
                        mode="lines",
                        name="All uploaded median",
                        line=dict(width=2, dash="dot"),
                        hovertemplate="%{x|%m/%d/%Y}<br>Spread: %{y:.1f} bps<extra>%{fullData.name}</extra>",
                    )
                )
        if "AAA/MMD baseline" in reference_lines and not spread_trend.empty:
            fig.add_hline(y=0, line_dash="longdash", line_width=1.5, annotation_text="AAA/MMD baseline")

        if fig.data:
            fig.update_layout(
                title=f"{selected_issuer} Spread Trend with Reference Lines",
                xaxis_title="Trade Date",
                yaxis_title="Spread (bps)",
                hovermode="x unified",
                height=520,
                legend_title_text="Line Item",
                margin=dict(l=40, r=40, t=70, b=45),
            )
            safe_plotly_chart(fig, width="stretch")
            with st.expander("Spread trend data", expanded=False):
                safe_dataframe(spread_trend, hide_index=True, top_rows=8)
        else:
            st.info("No spread trend traces were available for the selected filters.")
    elif "Spread Trend" in visible_charts:
        st.info("Spread trend needs issuer, trade_date, and spread or index_rate/yield fields.")

    if "Volume & Activity" in visible_charts:
        st.subheader("Volume & Activity")
    if "Volume & Activity" in visible_charts and not chart_base.empty and {"trade_date", "trade_amount", "issuer"}.issubset(chart_base.columns):
        vol = chart_base.copy()
        vol = vol.dropna(subset=["trade_date"])
        vol["period"] = vol["trade_date"].dt.to_period(period_freq).dt.to_timestamp()
        vol["volume_group"] = vol["issuer"].astype(str)
        volume_by_group = (
            vol.groupby(["period", "volume_group"], as_index=False)
            .agg(volume=("trade_amount", "sum"), trade_count=("trade_amount", "count"))
            .sort_values("period")
        )
        volume_total = (
            vol.groupby("period", as_index=False)
            .agg(total_volume=("trade_amount", "sum"), total_trade_count=("trade_amount", "count"))
            .sort_values("period")
        )
        selected_volume = (
            vol[vol["issuer"].astype(str) == str(selected_issuer)]
            .groupby("period", as_index=False)
            .agg(selected_volume=("trade_amount", "sum"))
        )
        volume_total = volume_total.merge(selected_volume, on="period", how="left")
        volume_total["selected_volume"] = volume_total["selected_volume"].fillna(0)
        volume_total["selected_issuer_share"] = np.where(
            volume_total["total_volume"] > 0,
            volume_total["selected_volume"] / volume_total["total_volume"] * 100,
            np.nan,
        )

        fig_vol = make_subplots(
            rows=2,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.08,
            row_heights=[0.68, 0.32],
            specs=[[{}], [{"secondary_y": True}]],
            subplot_titles=("Trading Volume", "Activity & Selected Issuer Share"),
        )
        for group_name in chart_issuers:
            tmp = volume_by_group[volume_by_group["volume_group"].astype(str) == str(group_name)]
            if tmp.empty:
                continue
            fig_vol.add_trace(
                go.Bar(
                    x=tmp["period"],
                    y=tmp["volume"] / 1_000_000,
                    name=group_name,
                    customdata=np.stack([tmp["trade_count"].fillna(0)], axis=-1),
                    hovertemplate="%{x|%m/%d/%Y}<br>Volume: $%{y:,.1f}M<br>Trades: %{customdata[0]:,.0f}<extra>%{fullData.name}</extra>",
                ),
                row=1,
                col=1,
            )
        fig_vol.add_trace(
            go.Scatter(
                x=volume_total["period"],
                y=volume_total["total_trade_count"],
                mode="lines+markers",
                name="Total trade count",
                line=dict(width=2.4),
                hovertemplate="%{x|%m/%d/%Y}<br>Total trades: %{y:,.0f}<extra>Total trade count</extra>",
            ),
            row=2,
            col=1,
            secondary_y=False,
        )
        fig_vol.add_trace(
            go.Scatter(
                x=volume_total["period"],
                y=volume_total["selected_issuer_share"],
                mode="lines+markers",
                name=f"{selected_issuer} volume share",
                line=dict(width=2.4, dash="dash"),
                hovertemplate="%{x|%m/%d/%Y}<br>Share: %{y:.1f}%<extra>%{fullData.name}</extra>",
            ),
            row=2,
            col=1,
            secondary_y=True,
        )
        fig_vol.update_layout(
            title=f"{trend_frequency} Volume, Trade Count, and {selected_issuer} Share",
            barmode="stack",
            height=680,
            hovermode="x unified",
            legend_title_text="Series",
            margin=dict(l=40, r=50, t=85, b=45),
        )
        fig_vol.update_yaxes(title_text="Volume ($MM)", row=1, col=1)
        fig_vol.update_yaxes(title_text="Trade Count", row=2, col=1, secondary_y=False)
        fig_vol.update_yaxes(title_text=f"{selected_issuer} Share", ticksuffix="%", row=2, col=1, secondary_y=True)
        safe_plotly_chart(fig_vol, width="stretch")

        volume_table = volume_by_group.merge(volume_total[["period", "total_volume", "total_trade_count", "selected_issuer_share"]], on="period", how="left")
        with st.expander("Volume and activity data", expanded=False):
            safe_dataframe(volume_table, hide_index=True, top_rows=8)
    elif "Volume & Activity" in visible_charts:
        st.info("Volume chart needs trade_date, trade_amount, and issuer fields.")

    if "Issuer Curve" not in visible_charts:
        return

    st.subheader("Issuer Curve")
    curve_source = issuer_trades.copy()
    if curve_source.empty or not {"maturity_year", "yield"}.issubset(curve_source.columns):
        st.info("Issuer curve needs maturity_year and yield fields.")
        return

    curve_lookback = st.select_slider(
        "Issuer Curve Lookback",
        options=[7, 14, 30, 60, 90, 180, 365],
        value=60,
        format_func=lambda x: f"{x} days",
        key="focused_core_curve_lookback",
        help="Uses selected issuer trades inside this lookback window ending on the latest selected trade date.",
    )
    curve_source["trade_date"] = pd.to_datetime(curve_source.get("trade_date"), errors="coerce")
    if "trade_date" in curve_source.columns:
        curve_source = curve_source[
            (curve_source["trade_date"].dt.date >= chart_start_date)
            & (curve_source["trade_date"].dt.date <= chart_end_date)
        ].copy()
    latest_curve_date = curve_source["trade_date"].dropna().max()
    if pd.notna(latest_curve_date):
        curve_source = curve_source[curve_source["trade_date"] >= latest_curve_date - pd.Timedelta(days=int(curve_lookback))].copy()
    curve_source["maturity_year"] = pd.to_numeric(curve_source["maturity_year"], errors="coerce")
    curve_source["yield"] = pd.to_numeric(curve_source["yield"], errors="coerce")
    issuer_curve = (
        curve_source.dropna(subset=["maturity_year", "yield"])
        .groupby("maturity_year", as_index=False)
        .agg(avg_yield=("yield", "mean"), trade_count=("yield", "count"), total_par=("trade_amount", "sum"), latest_trade=("trade_date", "max"))
        .sort_values("maturity_year")
    )
    if issuer_curve.empty:
        st.info("No curve observations were available for the selected lookback.")
        return

    curve_fig = go.Figure()
    curve_fig.add_trace(
        go.Scatter(
            x=issuer_curve["maturity_year"],
            y=issuer_curve["avg_yield"],
            mode="lines+markers",
            name=f"{selected_issuer} issuer curve",
            line=dict(width=3.2),
            customdata=np.stack([issuer_curve["trade_count"].fillna(0), issuer_curve["total_par"].fillna(0)], axis=-1),
            hovertemplate=(
                "%{x:.0f}Y<br>Yield: %{y:.3f}%<br>"
                "Trades: %{customdata[0]:,.0f}<br>Par: $%{customdata[1]:,.0f}"
                "<extra>%{fullData.name}</extra>"
            ),
        )
    )

    curve_universe = chart_base_all.copy()
    if pd.notna(latest_curve_date):
        curve_universe = curve_universe[curve_universe["trade_date"] >= latest_curve_date - pd.Timedelta(days=int(curve_lookback))].copy()
    if {"sector", "maturity_year", "yield"}.issubset(curve_universe.columns) and "Sector median" in reference_lines and selected_sector != "Unknown":
        sector_curve = curve_universe[curve_universe["sector"].astype(str) == str(selected_sector)].copy()
        sector_curve["maturity_year"] = pd.to_numeric(sector_curve["maturity_year"], errors="coerce")
        sector_curve["yield"] = pd.to_numeric(sector_curve["yield"], errors="coerce")
        sector_curve = (
            sector_curve.dropna(subset=["maturity_year", "yield"])
            .groupby("maturity_year", as_index=False)
            .agg(avg_yield=("yield", "median"), trade_count=("yield", "count"))
            .sort_values("maturity_year")
        )
        if not sector_curve.empty:
            curve_fig.add_trace(
                go.Scatter(
                    x=sector_curve["maturity_year"],
                    y=sector_curve["avg_yield"],
                    mode="lines+markers",
                    name=f"{selected_sector} median curve",
                    line=dict(width=2, dash="dash"),
                    hovertemplate="%{x:.0f}Y<br>Yield: %{y:.3f}%<extra>%{fullData.name}</extra>",
                )
            )
    if {"maturity_year", "yield"}.issubset(curve_universe.columns) and "All uploaded median" in reference_lines:
        all_curve = curve_universe.copy()
        all_curve["maturity_year"] = pd.to_numeric(all_curve["maturity_year"], errors="coerce")
        all_curve["yield"] = pd.to_numeric(all_curve["yield"], errors="coerce")
        all_curve = (
            all_curve.dropna(subset=["maturity_year", "yield"])
            .groupby("maturity_year", as_index=False)
            .agg(avg_yield=("yield", "median"), trade_count=("yield", "count"))
            .sort_values("maturity_year")
        )
        if not all_curve.empty:
            curve_fig.add_trace(
                go.Scatter(
                    x=all_curve["maturity_year"],
                    y=all_curve["avg_yield"],
                    mode="lines+markers",
                    name="All uploaded median curve",
                    line=dict(width=2, dash="dot"),
                    hovertemplate="%{x:.0f}Y<br>Yield: %{y:.3f}%<extra>%{fullData.name}</extra>",
                )
            )

    benchmark_curve = pd.DataFrame()
    if "MMD benchmark curve" in reference_lines and isinstance(mmd_df, pd.DataFrame) and not mmd_df.empty:
        date_col = _detect_mmd_date_column(mmd_df)
        if date_col:
            mmd_work = mmd_df.copy()
            mmd_work[date_col] = pd.to_datetime(mmd_work[date_col], errors="coerce").dt.normalize()
            mmd_work = mmd_work.dropna(subset=[date_col])
            if pd.notna(latest_curve_date):
                mmd_work = mmd_work[mmd_work[date_col] <= latest_curve_date.normalize()].copy()
            if not mmd_work.empty:
                rows = []
                for year in sorted(issuer_curve["maturity_year"].dropna().astype(int).unique().tolist()):
                    bucket = f"{year}Y"
                    tenor = MMD_BUCKET_MAP.get(bucket, "10Y")
                    y, meta = get_benchmark_curve(mmd_work, tenor, curve_benchmark_rating)
                    if y is None:
                        continue
                    bench_tmp = pd.DataFrame({"benchmark_yield": pd.to_numeric(y, errors="coerce")}).dropna()
                    if bench_tmp.empty:
                        continue
                    rows.append(
                        {
                            "maturity_year": year,
                            "benchmark_yield": float(bench_tmp["benchmark_yield"].iloc[-1]),
                            "benchmark_rating": curve_benchmark_rating,
                            "mmd_tenor": tenor,
                            "benchmark_source": meta.get("benchmark_source"),
                            "source_column": meta.get("source_column"),
                            "rating_spread_bps": meta.get("rating_spread_bps"),
                        }
                    )
                benchmark_curve = pd.DataFrame(rows)
                if not benchmark_curve.empty:
                    curve_fig.add_trace(
                        go.Scatter(
                            x=benchmark_curve["maturity_year"],
                            y=benchmark_curve["benchmark_yield"],
                            mode="lines+markers",
                            name=f"{curve_benchmark_rating} benchmark curve",
                            line=dict(width=2.4, dash="longdash"),
                            hovertemplate="%{x:.0f}Y<br>Yield: %{y:.3f}%<extra>%{fullData.name}</extra>",
                        )
                    )

    curve_fig.update_layout(
        title=f"{selected_issuer} Issuer Curve vs References",
        xaxis_title="Maturity Year",
        yaxis_title="Yield (%)",
        hovermode="x unified",
        height=540,
        legend_title_text="Curve",
        margin=dict(l=40, r=40, t=70, b=45),
    )
    safe_plotly_chart(curve_fig, width="stretch")

    curve_table = issuer_curve.copy()
    if not benchmark_curve.empty:
        curve_table = curve_table.merge(benchmark_curve, on="maturity_year", how="left")
        curve_table["spread_to_benchmark_bps"] = (
            pd.to_numeric(curve_table["avg_yield"], errors="coerce")
            - pd.to_numeric(curve_table["benchmark_yield"], errors="coerce")
        ) * 100
    display_cols = [
        "maturity_year", "avg_yield", "benchmark_rating", "benchmark_yield",
        "spread_to_benchmark_bps", "trade_count", "total_par", "latest_trade",
        "mmd_tenor", "benchmark_source", "source_column", "rating_spread_bps",
    ]
    with st.expander("Issuer curve data", expanded=False):
        safe_dataframe(curve_table[[c for c in display_cols if c in curve_table.columns]], hide_index=True, top_rows=12)
