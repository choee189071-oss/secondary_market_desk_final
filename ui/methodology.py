from __future__ import annotations

import pandas as pd
import streamlit as st

from engine.methodology import methodology_evidence_summary, methodology_trust_layers
from ui.common import _render_card_grid, safe_dataframe


def render_methodology_trust_panel(
    market_df: pd.DataFrame,
    issuer_df: pd.DataFrame | None,
    mmd_df: pd.DataFrame,
    benchmark_source_mode: str,
    benchmark_priority: str,
    benchmark_conflict_policy: str,
    *,
    title: str = "Methodology Evidence",
    expanded: bool = False,
    show_cards: bool = True,
    show_trust_layers: bool = True,
    render_benchmark_methodology_block=None,
):
    """Render the shared trust/evidence layer used by Upload, Snapshot, and Export."""
    summary = methodology_evidence_summary(
        market_df=market_df,
        issuer_df=issuer_df,
        mmd_df=mmd_df,
        benchmark_source_mode=benchmark_source_mode,
        benchmark_priority=benchmark_priority,
        benchmark_conflict_policy=benchmark_conflict_policy,
    )

    if show_cards:
        _render_card_grid(summary["cards"], "status-card-grid")

    with st.expander(title, expanded=expanded):
        st.subheader("Evidence Summary")
        safe_dataframe(summary["evidence"], hide_index=True, auto_collapse=False)

        if show_trust_layers:
            trust_layers = methodology_trust_layers(
                benchmark_source_mode=benchmark_source_mode,
                benchmark_priority=benchmark_priority,
                benchmark_conflict_policy=benchmark_conflict_policy,
            )
            trust_tabs = st.tabs(list(trust_layers.keys()))
            for tab, (layer_name, layer_df) in zip(trust_tabs, trust_layers.items()):
                with tab:
                    safe_dataframe(layer_df, hide_index=True, auto_collapse=False)

        if render_benchmark_methodology_block is not None:
            st.subheader("Benchmark Detail")
            render_benchmark_methodology_block(
                mmd_df,
                benchmark_source_mode,
                benchmark_priority,
                benchmark_conflict_policy,
            )
