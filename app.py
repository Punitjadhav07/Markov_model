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


st.set_page_config(page_title="Markov + HMM (Simple)", layout="centered")
st.title("Customer Behavior: Markov vs Hidden Markov")

markov_probs = load_markov_probs()
hmm, scaler, demo, name_map, feature_cols = train_small_hmm()

st.subheader("Input")
sequence_options = {
    "Browse-heavy": "view,view,view",
    "Cart-intent": "view,view,addtocart",
    "Fast-conversion": "view,addtocart,transaction",
    "Mixed": "view,addtocart,view,addtocart",
}
selected_label = st.selectbox("Choose a sequence pattern", list(sequence_options.keys()))
seq_text = st.text_input("Sequence text (editable)", value=sequence_options[selected_label])
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
# Create a quick feature vector from typed sequence (simple approximation for demo)
n_views = events.count("view")
n_cart = events.count("addtocart")
n_purchase = events.count("transaction")
cart_to_view = n_cart / max(n_views, 1)
engagement = 0.6 * n_views + 1.5 * n_cart + 3.0 * n_purchase
intent = 2.0 * n_cart + 4.0 * n_purchase

sample = pd.DataFrame(
    [[
        max(len(events), 1) * 60.0,
        45.0,
        cart_to_view,
        float(n_cart),
        float(n_purchase),
        120.0 if n_cart > 0 else -1.0,
        300.0 if n_purchase > 0 else -1.0,
        engagement,
        intent,
    ]],
    columns=feature_cols,
)
hidden_id = int(hmm.predict(scaler.transform(sample.values))[0])
intent_name = name_map[hidden_id]
st.write(f"Inferred hidden intent: **{intent_name}**")

# Map hidden state to likely next action from demo rows
demo_next = []
for _, row in demo.iterrows():
    if row["number_of_purchases"] > 0:
        demo_next.append("Purchase")
    elif row["number_of_cart_additions"] > 0:
        demo_next.append("Add_to_Cart")
    else:
        demo_next.append("Browse")
demo = demo.copy()
demo["next_action"] = demo_next
hidden_to_next = pd.crosstab(demo["hidden_state"], demo["next_action"])
hidden_to_next = hidden_to_next.div(hidden_to_next.sum(axis=1), axis=0).fillna(0.0)
if hidden_id in hidden_to_next.index:
    hmm_probs = hidden_to_next.loc[hidden_id].sort_values(ascending=False)
else:
    hmm_probs = pd.Series({"Browse": 1.0})

st.dataframe(hmm_probs.reset_index().rename(columns={"index": "next_state", hidden_id: "probability", 0: "probability"}))

fig_h, ax_h = plt.subplots(figsize=(4, 3))
sns.heatmap(pd.DataFrame(hmm.transmat_, index=[name_map[i] for i in range(3)], columns=[name_map[i] for i in range(3)]), annot=True, fmt=".2f", cmap="Blues", ax=ax_h)
ax_h.set_title("HMM Hidden Transition Heatmap")
st.pyplot(fig_h)
plt.close(fig_h)

st.caption("HMM identifies user intent, Markov does not.")
