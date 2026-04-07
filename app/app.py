from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st


@st.cache_data
def load_transition_matrix() -> pd.DataFrame:
    app_dir = Path(__file__).resolve().parent
    pkl_path = app_dir / "transition_probs.pkl"
    csv_path = app_dir / "transition_probs.csv"

    if pkl_path.exists():
        return pd.read_pickle(pkl_path)
    if csv_path.exists():
        return pd.read_csv(csv_path, index_col=0)
    return pd.DataFrame()


def predict_next_state(current_state: str, prob_matrix: pd.DataFrame) -> pd.Series:
    if current_state not in prob_matrix.index:
        raise ValueError(
            f"Unknown state '{current_state}'. Valid states: {list(prob_matrix.index)}"
        )
    return prob_matrix.loc[current_state].sort_values(ascending=False)


def main() -> None:
    st.set_page_config(page_title="Customer Behavior Predictor", layout="centered")

    transition_probs = load_transition_matrix()
    if transition_probs.empty:
        st.error(
            "Transition matrix not found. Run `notebooks/analysis.ipynb` to generate "
            "`transition_probs.pkl` (and CSV) in the `app/` folder, then refresh this page."
        )
        st.stop()

    st.title("Customer Behavior Predictor")
    st.caption("Next-step probabilities from a first-order Markov model (Retailrocket events).")

    # --- Section 1: Next-State Prediction ---
    st.subheader("1. Next-State Prediction")
    state_options = list(transition_probs.index)
    current_state = st.selectbox("Current state", state_options)

    if st.button("Predict Next State →", type="primary"):
        probs = predict_next_state(current_state, transition_probs)
        probs = probs.dropna()

        fig, ax = plt.subplots(figsize=(8, max(3, 0.45 * len(probs))))
        labels = probs.index.astype(str).tolist()
        values = probs.values.astype(float)
        max_idx = int(values.argmax()) if len(values) else -1
        colors = ["#2171b5" if i == max_idx else "#9ecae9" for i in range(len(values))]

        y_pos = range(len(values))
        ax.barh(list(y_pos), values, color=colors, edgecolor="white", linewidth=0.5)
        ax.set_yticks(list(y_pos))
        ax.set_yticklabels(labels)
        ax.set_xlabel("Probability")
        ax.set_xlim(0, max(values) * 1.15 if len(values) else 1)
        ax.set_title(f"Next states from “{current_state}”")

        for i, v in enumerate(values):
            ax.text(
                v + max(values) * 0.01 if len(values) else 0.01,
                i,
                f"{v:.3f}",
                va="center",
                fontsize=10,
            )

        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        out_df = probs.reset_index()
        out_df.columns = ["Next state", "Probability"]
        st.dataframe(out_df, use_container_width=True)

        top_state = probs.index[0]
        if top_state == "Exit":
            st.info(
                "Interpretation: the most likely next step is **Exit** — high immediate "
                "drop-off risk from this state."
            )
        elif top_state == "Purchase":
            st.info(
                "Interpretation: **Purchase** is most likely next — strong conversion signal "
                "from this state."
            )
        elif top_state == "Browse":
            st.info(
                "Interpretation: **Browse** is most likely next — users tend to keep exploring."
            )
        elif top_state == "Add_to_Cart":
            st.info(
                "Interpretation: **Add_to_Cart** is most likely next — rising purchase intent."
            )
        else:
            st.info(f"Interpretation: **{top_state}** is the most probable next state.")

    st.divider()

    # --- Section 2: Full Heatmap ---
    st.subheader("2. Full Transition Matrix (heatmap)")
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    sns.heatmap(
        transition_probs,
        annot=True,
        fmt=".3f",
        cmap="Blues",
        ax=ax2,
    )
    ax2.set_title("Markov Transition Probability Heatmap")
    plt.tight_layout()
    st.pyplot(fig2)
    plt.close(fig2)

    st.divider()

    # --- Section 3: Raw Matrix ---
    with st.expander("View Raw Transition Matrix"):
        styled = (
            transition_probs.style.format("{:.4f}")
            .background_gradient(cmap="Blues")
        )
        st.dataframe(styled, use_container_width=True)

    st.divider()

    # --- Section 4: Funnel Insights ---
    st.subheader("4. Funnel insights (from matrix)")
    c1, c2, c3 = st.columns(3)

    p_browse_exit = float(transition_probs.loc["Browse", "Exit"])
    p_browse_cart = float(transition_probs.loc["Browse", "Add_to_Cart"])
    p_cart_purchase = float(transition_probs.loc["Add_to_Cart", "Purchase"])

    with c1:
        st.metric(
            label="Browse → Exit (drop-off)",
            value=f"{p_browse_exit * 100:.2f}%",
            delta=f"{p_browse_exit * 100:.2f}%",
            delta_color="inverse",
        )
    with c2:
        st.metric(
            label="Browse → Add_to_Cart (engagement)",
            value=f"{p_browse_cart * 100:.2f}%",
            delta=f"{p_browse_cart * 100:.2f}%",
        )
    with c3:
        st.metric(
            label="Add_to_Cart → Purchase (conversion)",
            value=f"{p_cart_purchase * 100:.2f}%",
            delta=f"{p_cart_purchase * 100:.2f}%",
        )

    st.caption("Customer Purchase Behavior Analysis — Markov chain on Retailrocket-style events.")


if __name__ == "__main__":
    main()
