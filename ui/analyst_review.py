from __future__ import annotations

import json

import pandas as pd
import streamlit as st

from engine.methodology import analyst_review_items
from reports.export_center import focused_report_filename


REVIEW_STATUSES = ["Correct", "Needs Review", "Wrong", "Not Reviewed"]


def _review_records() -> dict:
    if "analyst_review_records" not in st.session_state:
        st.session_state["analyst_review_records"] = {}
    return st.session_state["analyst_review_records"]


def _current_review_dataframe(context: dict) -> pd.DataFrame:
    base = analyst_review_items(context)
    records = _review_records()
    if base.empty:
        return pd.DataFrame()

    rows = []
    for _, row in base.iterrows():
        key = str(row["Review item"])
        saved = records.get(key, {})
        rows.append(
            {
                "Review item": key,
                "Current output": row.get("Current output", ""),
                "Suggested expected value": row.get("Suggested expected value", ""),
                "Status": saved.get("status", "Not Reviewed"),
                "Analyst expected value": saved.get("expected_value", row.get("Suggested expected value", "")),
                "Analyst note": saved.get("note", ""),
                "Reviewer": saved.get("reviewer", ""),
                "Updated": saved.get("updated_at", ""),
                "Why review": row.get("Why review", ""),
            }
        )
    return pd.DataFrame(rows)


def render_analyst_review_mode(context: dict, selected_issuer: str, safe_dataframe) -> pd.DataFrame:
    """Render analyst validation workflow and return current review table."""
    st.subheader("Analyst Review Mode")
    st.markdown(
        "<div class='focus-band'><b>Lock:</b> status, expected value, note.</div>",
        unsafe_allow_html=True,
    )

    review_df = _current_review_dataframe(context)
    if review_df.empty:
        st.info("No review checklist is available for the current report context.")
        return review_df

    reviewer_col, item_col = st.columns([1, 1.4])
    with reviewer_col:
        reviewer = st.text_input("Reviewer", key="analyst_review_reviewer", placeholder="Name or team")
    with item_col:
        selected_item = st.selectbox("Review item", review_df["Review item"].tolist(), key="analyst_review_item")

    selected_row = review_df[review_df["Review item"] == selected_item].iloc[0]
    st.caption(f"Why review: {selected_row.get('Why review', '')}")

    c1, c2 = st.columns([1, 1])
    with c1:
        st.text_input("Current output", value=str(selected_row.get("Current output", "")), disabled=True)
    with c2:
        expected_value = st.text_input(
            "Analyst expected value",
            value=str(selected_row.get("Analyst expected value", "")),
            key=f"analyst_expected_{selected_item}",
        )

    c3, c4 = st.columns([1, 2])
    with c3:
        current_status = str(selected_row.get("Status", "Not Reviewed"))
        status = st.selectbox(
            "Validation status",
            REVIEW_STATUSES,
            index=REVIEW_STATUSES.index(current_status) if current_status in REVIEW_STATUSES else REVIEW_STATUSES.index("Not Reviewed"),
            key=f"analyst_status_{selected_item}",
        )
    with c4:
        note = st.text_area(
            "Analyst note",
            value=str(selected_row.get("Analyst note", "")),
            key=f"analyst_note_{selected_item}",
            height=86,
            placeholder="What should be checked, corrected, or locked as expected output?",
        )

    save_col, clear_col = st.columns([1, 1])
    with save_col:
        if st.button("Save Review Item", key=f"save_review_{selected_item}"):
            records = _review_records()
            records[selected_item] = {
                "review_item": selected_item,
                "current_output": selected_row.get("Current output", ""),
                "expected_value": expected_value,
                "status": status,
                "note": note,
                "reviewer": reviewer,
                "updated_at": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M"),
            }
            st.success(f"Saved review for {selected_item}.")
            review_df = _current_review_dataframe(context)
    with clear_col:
        if st.button("Clear Review Feedback", key="clear_analyst_review_feedback"):
            st.session_state["analyst_review_records"] = {}
            st.info("Analyst review feedback cleared.")
            review_df = _current_review_dataframe(context)

    st.caption("Current review table")
    safe_dataframe(review_df, hide_index=True, auto_collapse=False)

    records_df = pd.DataFrame(_review_records().values())
    export_df = review_df if records_df.empty else records_df
    e1, e2 = st.columns(2)
    with e1:
        st.download_button(
            "Download Review CSV",
            data=export_df.to_csv(index=False).encode("utf-8"),
            file_name=focused_report_filename(selected_issuer, "analyst_review.csv"),
            mime="text/csv",
        )
    with e2:
        st.download_button(
            "Download Review JSON",
            data=json.dumps(export_df.to_dict("records"), indent=2, default=str).encode("utf-8"),
            file_name=focused_report_filename(selected_issuer, "analyst_review.json"),
            mime="application/json",
        )

    with st.expander("Expected output JSON template", expanded=False):
        template = {
            "issuer": selected_issuer,
            "expected": {
                str(row["Review item"]): {
                    "current_output": row.get("Current output", ""),
                    "expected_value": row.get("Analyst expected value", row.get("Suggested expected value", "")),
                    "status": row.get("Status", "Not Reviewed"),
                }
                for _, row in review_df.iterrows()
            },
        }
        st.code(json.dumps(template, indent=2, default=str), language="json")

    return review_df
