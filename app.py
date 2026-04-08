from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st
from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import StandardScaler

BASE_DIR = Path(__file__).resolve().parent


def load_markov_probs() -> pd.DataFrame:
    df = pd.read_csv(BASE_DIR / "data" / "event.csv")
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    state_map = {"view": "Browse", "addtocart": "Add_to_Cart", "transaction": "Purchase"}
    df["state"] = df["event"].map(state_map)
    df = df.sort_values(["visitorid", "timestamp"]).reset_index(drop=True)

    sequences = df.groupby("visitorid")["state"].apply(list).apply(lambda s: s + ["Exit"])
    pairs = []
    for seq in sequences:
        pairs.extend(list(zip(seq[:-1], seq[1:])))
    t = pd.DataFrame(pairs, columns=["from_state", "to_state"])
    counts = pd.crosstab(t["from_state"], t["to_state"])
    return counts.div(counts.sum(axis=1), axis=0).fillna(0.0)


def load_user_sequences() -> dict[str, str]:
    """
    Build dropdown options from real user sequences in data/event.csv.
    """
    df = pd.read_csv(BASE_DIR / "data" / "event.csv")
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    df = df.sort_values(["visitorid", "timestamp"]).reset_index(drop=True)
    seq = df.groupby("visitorid")["event"].apply(list)

    options: dict[str, str] = {}
    for vid, events in seq.items():
        label = f"user_{int(vid)} ({len(events)} events)"
        options[label] = ",".join(events)

    # Include clear demo templates with distinct intent levels
    options["demo: Low intent (browse only)"] = "view,view,view,view"
    options["demo: Medium intent (cart no purchase)"] = "view,view,addtocart,view"
    options["demo: High intent (purchase)"] = "view,addtocart,transaction,view"
    return options


def train_small_hmm() -> tuple[GaussianHMM, StandardScaler, pd.DataFrame, dict[int, str], list[str]]:
    demo = pd.read_csv(BASE_DIR / "data" / "newdata.csv")
    feature_cols = [
        "session_length_s",
        "avg_time_between_events_s",
        "cart_to_view_ratio",
        "number_of_cart_additions",
        "number_of_purchases",
        "time_to_cart_s",
        "time_to_purchase_s",
        "engagement_score",
        "conversion_intent_score",
    ]
    X = demo[feature_cols].values.astype(float)
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    hmm = GaussianHMM(n_components=3, covariance_type="diag", n_iter=200, random_state=42)
    hmm.fit(Xs)
    hidden = hmm.predict(Xs)
    demo["hidden_state"] = hidden

    order = demo.groupby("hidden_state")["conversion_intent_score"].mean().sort_values().index.tolist()
    name_map = {order[0]: "Low Intent", order[1]: "Medium Intent", order[2]: "High Intent"}
    return hmm, scaler, demo, name_map, feature_cols


def build_hidden_to_next_distribution(demo: pd.DataFrame) -> pd.DataFrame:
    """
    Build smoother P(next_action | hidden_state) to avoid deterministic 1.0 outputs.
    """
    d = demo.copy()
    score = (
        0.55 * d["conversion_intent_score"]
        + 0.30 * d["engagement_score"]
        + 0.15 * d["number_of_cart_additions"]
    )
    q1 = score.quantile(0.33)
    q2 = score.quantile(0.66)

    def label_next_action(s: float, n_purchase: float, n_cart: float) -> str:
        if n_purchase > 0 and s >= q2:
            return "Purchase"
        if s < q1 and n_cart == 0:
            return "Browse"
        if s >= q2:
            return "Purchase"
        return "Add_to_Cart"

    d["next_action"] = [
        label_next_action(s, p, c)
        for s, p, c in zip(score.values, d["number_of_purchases"].values, d["number_of_cart_additions"].values)
    ]
    h2n = pd.crosstab(d["hidden_state"], d["next_action"]).astype(float)
    for col in ["Browse", "Add_to_Cart", "Purchase"]:
        if col not in h2n.columns:
            h2n[col] = 0.0
    # Laplace smoothing so we avoid hard zeros
    h2n = h2n[["Browse", "Add_to_Cart", "Purchase"]] + 0.1
    h2n = h2n.div(h2n.sum(axis=1), axis=0).fillna(0.0)
    return h2n


st.set_page_config(page_title="Markov + HMM (Simple)", layout="centered")
st.title("Customer Behavior: Markov vs Hidden Markov")

markov_probs = load_markov_probs()
hmm, scaler, demo, name_map, feature_cols = train_small_hmm()
sequence_options = load_user_sequences()

st.subheader("Input")
selected_label = st.selectbox("Choose a sequence pattern", list(sequence_options.keys()))
seq_text = sequence_options[selected_label]
st.caption(f"Selected sequence: `{seq_text}`")
events = [x.strip().lower() for x in seq_text.split(",") if x.strip()]
state_map = {"view": "Browse", "addtocart": "Add_to_Cart", "transaction": "Purchase"}
states = [state_map.get(e, "Browse") for e in events]
current_state = states[-1] if states else "Browse"

st.subheader("MARKOV")
if current_state in markov_probs.index:
    m_probs = markov_probs.loc[current_state].sort_values(ascending=False)
else:
    m_probs = pd.Series({"Exit": 1.0})
st.write(f"Current observable state: **{current_state}**")
st.dataframe(m_probs.reset_index().rename(columns={"index": "next_state", current_state: "probability", 0: "probability"}))

fig_m, ax_m = plt.subplots(figsize=(6, 3))
sns.heatmap(markov_probs, annot=True, fmt=".2f", cmap="Blues", ax=ax_m)
ax_m.set_title("Markov Transition Heatmap")
st.pyplot(fig_m)
plt.close(fig_m)

st.subheader("HMM")
# Create a sequence-sensitive feature vector for HMM inference
n_views = events.count("view")
n_cart = events.count("addtocart")
n_purchase = events.count("transaction")
cart_to_view = n_cart / max(n_views, 1)
engagement = 0.6 * n_views + 1.5 * n_cart + 3.0 * n_purchase
intent = 2.0 * n_cart + 4.0 * n_purchase

# Temporal terms depend on sequence composition/order, so hidden intent can change by input
base_gap = 28 + (6 * n_views) + (10 * n_cart) + (14 * n_purchase)
avg_gap = float(max(20, min(base_gap, 120)))
session_length = float(max(len(events), 1) * avg_gap)
idx_cart = events.index("addtocart") if "addtocart" in events else -1
idx_purchase = events.index("transaction") if "transaction" in events else -1
time_to_cart = float((idx_cart + 1) * avg_gap) if idx_cart >= 0 else -1.0
time_to_purchase = float((idx_purchase + 1) * avg_gap) if idx_purchase >= 0 else -1.0

sample = pd.DataFrame(
    [[
        session_length,
        avg_gap,
        cart_to_view,
        float(n_cart),
        float(n_purchase),
        time_to_cart,
        time_to_purchase,
        engagement,
        intent,
    ]],
    columns=feature_cols,
)
hidden_id = int(hmm.predict(scaler.transform(sample.values))[0])
intent_name = name_map[hidden_id]
st.write(f"Inferred hidden intent: **{intent_name}**")

# Soft hidden-state posterior + blended next-action probabilities
hidden_post = hmm.predict_proba(scaler.transform(sample.values))[0]
hidden_to_next = build_hidden_to_next_distribution(demo)
hmm_probs_arr = hidden_post @ hidden_to_next.reindex([0, 1, 2], fill_value=0.0).values
hmm_probs = pd.Series(hmm_probs_arr, index=hidden_to_next.columns).sort_values(ascending=False)

st.dataframe(hmm_probs.reset_index().rename(columns={"index": "next_state", hidden_id: "probability", 0: "probability"}))
st.caption("HMM probabilities are sequence-dependent (computed from selected sequence features).")

fig_h, ax_h = plt.subplots(figsize=(4, 3))
sns.heatmap(pd.DataFrame(hmm.transmat_, index=[name_map[i] for i in range(3)], columns=[name_map[i] for i in range(3)]), annot=True, fmt=".2f", cmap="Blues", ax=ax_h)
ax_h.set_title("HMM Hidden Transition Heatmap")
st.pyplot(fig_h)
plt.close(fig_h)

st.caption("HMM identifies user intent, Markov does not.")
