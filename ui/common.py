from __future__ import annotations

import pandas as pd
import streamlit as st

from app_state import (
    LARGE_TABLE_COL_THRESHOLD,
    LARGE_TABLE_ROW_THRESHOLD,
    MAX_TABLE_ROWS,
    TABLE_PREVIEW_ROWS,
)


def _make_unique_columns(columns) -> list[str]:
    """Return unique, human-readable column names for Streamlit/Arrow display."""
    seen: dict[str, int] = {}
    unique_cols: list[str] = []
    for col in columns:
        base = str(col)
        if base not in seen:
            seen[base] = 0
            unique_cols.append(base)
        else:
            seen[base] += 1
            unique_cols.append(f"{base}_{seen[base]}")
    return unique_cols


def prepare_display_dataframe(df: pd.DataFrame, max_rows: int | None = TABLE_PREVIEW_ROWS) -> pd.DataFrame:
    """Prepare a dataframe for safe Streamlit display."""
    if df is None:
        return pd.DataFrame()
    if not isinstance(df, pd.DataFrame):
        try:
            df = pd.DataFrame(df)
        except Exception:
            return pd.DataFrame({"value": [str(df)]})

    out = df.copy()
    out.columns = _make_unique_columns(out.columns)

    if max_rows is not None and len(out) > max_rows:
        out = out.head(int(max_rows)).copy()

    for col in out.columns:
        try:
            if pd.api.types.is_datetime64_any_dtype(out[col]):
                out[col] = pd.to_datetime(out[col], errors="coerce").dt.strftime("%m/%d/%Y")
            elif "date" in str(col).lower() or "maturity" in str(col).lower():
                converted = pd.to_datetime(out[col], errors="coerce")
                if len(converted) == 0 or converted.notna().mean() >= 0.6:
                    out[col] = converted.dt.strftime("%m/%d/%Y")
        except Exception:
            pass
    return out


def safe_dataframe(
    df: pd.DataFrame,
    *args,
    expander_label: str | None = None,
    expanded: bool = False,
    max_rows: int | None = TABLE_PREVIEW_ROWS,
    auto_collapse: bool = True,
    top_rows: int = 10,
    **kwargs,
):
    """Render dataframes safely for focused workflow modules."""
    if "use_container_width" in kwargs and "width" not in kwargs:
        kwargs["width"] = "stretch" if kwargs.pop("use_container_width") else "content"
    else:
        kwargs.pop("use_container_width", None)
    kwargs.setdefault("width", "stretch")

    effective_max_rows = max_rows
    try:
        if effective_max_rows == TABLE_PREVIEW_ROWS:
            effective_max_rows = min(TABLE_PREVIEW_ROWS, int(MAX_TABLE_ROWS))
    except Exception:
        pass

    display_df = prepare_display_dataframe(df, max_rows=effective_max_rows)
    row_count = len(df) if isinstance(df, pd.DataFrame) else len(display_df)
    col_count = len(display_df.columns)
    is_large = row_count >= LARGE_TABLE_ROW_THRESHOLD or col_count >= LARGE_TABLE_COL_THRESHOLD

    if expander_label is None:
        expander_label = f"View data table ({row_count:,} rows x {col_count:,} cols)"
        if effective_max_rows is not None and row_count > effective_max_rows:
            expander_label += f" - preview capped at {effective_max_rows:,} rows"

    if auto_collapse and is_large:
        preview_rows = min(max(int(top_rows), 1), len(display_df))
        st.caption(f"Showing top {preview_rows:,} rows. Expand for a larger preview.")
        st.dataframe(display_df.head(preview_rows), *args, **kwargs)
        with st.expander(expander_label, expanded=expanded):
            if effective_max_rows is not None and row_count > effective_max_rows:
                st.caption(f"Large-table protection: showing first {effective_max_rows:,} of {row_count:,} rows.")
            return st.dataframe(display_df, *args, **kwargs)

    return st.dataframe(display_df, *args, **kwargs)


def safe_plotly_chart(fig, *args, **kwargs):
    """Central wrapper for Plotly charts and deprecated Streamlit kwargs."""
    if "use_container_width" in kwargs and "width" not in kwargs:
        kwargs["width"] = "stretch" if kwargs.pop("use_container_width") else "content"
    else:
        kwargs.pop("use_container_width", None)
    kwargs.setdefault("width", "stretch")
    try:
        if fig is not None:
            fig.update_layout(uirevision="keep")
    except Exception:
        pass
    return st.plotly_chart(fig, *args, **kwargs)


def section_anchor(anchor_id: str, title: str, level: int = 2):
    """Create a stable HTML anchor plus a Streamlit header/subheader."""
    st.markdown(f"<a id='{anchor_id}'></a>", unsafe_allow_html=True)
    if level == 1:
        st.title(title)
    elif level == 2:
        st.header(title)
    else:
        st.subheader(title)


def clean_metric_card(label: str, value: object, size: str = "large", note: str | None = None):
    """Compact custom metric card that gives long text more room than st.metric."""
    value_class = "clean-card-value-large" if size == "large" else "clean-card-value-small"
    safe_value = "-" if value is None else str(value)
    note_html = f"<div class='clean-card-note'>{note}</div>" if note else ""
    st.markdown(
        f"""
<div class="clean-card">
  <div class="clean-card-label">{label}</div>
  <div class="{value_class}">{safe_value}</div>
  {note_html}
</div>
""",
        unsafe_allow_html=True,
    )


def _fmt_bps(x, digits: int = 1) -> str:
    try:
        if pd.isna(x):
            return "N/A"
        return f"{float(x):+.{digits}f} bp"
    except Exception:
        return "N/A"


def _fmt_num(x, digits: int = 1) -> str:
    try:
        if pd.isna(x):
            return "N/A"
        return f"{float(x):,.{digits}f}"
    except Exception:
        return "N/A"


def _fmt_pct(x, digits: int = 1) -> str:
    try:
        if pd.isna(x):
            return "N/A"
        return f"{float(x):.{digits}f}%"
    except Exception:
        return "N/A"


def _fmt_mm(x, digits: int = 1) -> str:
    try:
        if pd.isna(x):
            return "N/A"
        return f"${float(x) / 1_000_000:,.{digits}f}M"
    except Exception:
        return "N/A"


def _fmt_date(x) -> str:
    try:
        return pd.to_datetime(x).strftime("%m/%d/%Y")
    except Exception:
        return str(x)


def _first_existing_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    if not isinstance(df, pd.DataFrame):
        return None
    for c in candidates:
        if c in df.columns:
            return c
    return None
