from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st


STATE_MAP = {
    "view": "Browse",
    "addtocart": "Add_to_Cart",
    "transaction": "Purchase",
}


@st.cache_data
def load_and_prepare_data() -> pd.DataFrame:
    csv_path = Path(__file__).resolve().parents[1] / "Data" / "events.csv"
    df = pd.read_csv(csv_path, usecols=["visitorid", "timestamp", "event"])
    df = df[df["event"].isin(["view", "addtocart", "transaction"])].copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    df = df.sort_values(["visitorid", "timestamp"]).reset_index(drop=True)
    df["state"] = df["event"].map(STATE_MAP)
    return df


@st.cache_data
def build_markov_artifacts(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    transitions_df = df[["visitorid", "state"]].copy()
    transitions_df["to_state"] = transitions_df.groupby("visitorid")["state"].shift(-1).fillna("Exit")
    transition_counts = pd.crosstab(transitions_df["state"], transitions_df["to_state"])
    transition_probs = transition_counts.div(transition_counts.sum(axis=1), axis=0)
    return transitions_df, transition_counts, transition_probs


def main() -> None:
    st.set_page_config(page_title="Customer Behavior Dashboard", layout="wide")
    st.title("Customer Purchase Behavior Dashboard")
    st.caption("Markov Chain analysis of customer journeys from browse to purchase.")

    df = load_and_prepare_data()
    transitions_df, transition_counts, transition_probs = build_markov_artifacts(df)

    total_customers = int(df["visitorid"].nunique())
    total_events = int(len(df))
    total_purchases = int((df["event"] == "transaction").sum())
    customers_with_purchase = int(df.loc[df["event"] == "transaction", "visitorid"].nunique())
    purchase_customer_rate = (customers_with_purchase / total_customers * 100) if total_customers else 0.0

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Customers", f"{total_customers:,}")
    c2.metric("Total Events", f"{total_events:,}")
    c3.metric("Transactions", f"{total_purchases:,}")
    c4.metric("Customers Who Purchased", f"{purchase_customer_rate:.2f}%")

    left, right = st.columns([1, 1])

    with left:
        st.subheader("Event Mix")
        event_mix = (
            df["event"]
            .value_counts()
            .rename_axis("event")
            .reset_index(name="count")
            .sort_values("count", ascending=False)
        )
        st.bar_chart(event_mix.set_index("event"))
        st.dataframe(event_mix, use_container_width=True)

    with right:
        st.subheader("Daily Event Trend")
        daily_events = (
            df.set_index("timestamp")
            .resample("D")
            .size()
            .rename("events")
            .reset_index()
        )
        st.line_chart(daily_events.set_index("timestamp")["events"])
        st.caption("Shows traffic trend over time across all event types.")

    st.subheader("Transition Probability Heatmap")
    heatmap_df = (
        transition_probs.reset_index()
        .melt(id_vars="state", var_name="to_state", value_name="probability")
        .rename(columns={"state": "from_state"})
    )
    heatmap = (
        alt.Chart(heatmap_df)
        .mark_rect()
        .encode(
            x=alt.X("to_state:N", title="To State"),
            y=alt.Y("from_state:N", title="From State"),
            color=alt.Color("probability:Q", title="Probability", scale=alt.Scale(scheme="blues")),
            tooltip=[
                alt.Tooltip("from_state:N", title="From"),
                alt.Tooltip("to_state:N", title="To"),
                alt.Tooltip("probability:Q", format=".4f", title="Probability"),
            ],
        )
        .properties(height=280)
    )
    st.altair_chart(heatmap, use_container_width=True)
    st.dataframe(transition_probs.round(4), use_container_width=True)

    st.subheader("Top Transitions by Count")
    top_transitions = (
        transitions_df.groupby(["state", "to_state"])
        .size()
        .rename("count")
        .reset_index()
        .sort_values("count", ascending=False)
        .head(12)
    )
    top_transitions["transition"] = top_transitions["state"] + " → " + top_transitions["to_state"]
    st.bar_chart(top_transitions.set_index("transition")["count"])
    st.dataframe(top_transitions[["state", "to_state", "count"]], use_container_width=True)


if __name__ == "__main__":
    main()
