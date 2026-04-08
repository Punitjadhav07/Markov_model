from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import StandardScaler


@dataclass(frozen=True)
class HMMArtifacts:
    model: GaussianHMM
    scaler: StandardScaler
    hidden_state_names: list[str]
    hidden_to_next_action: pd.DataFrame  # index=hidden_state, columns=observable next actions


def train_gaussian_hmm(
    X: np.ndarray,
    lengths: list[int],
    n_hidden_states: int = 4,
    random_state: int = 42,
    n_iter: int = 200,
) -> GaussianHMM:
    model = GaussianHMM(
        n_components=n_hidden_states,
        covariance_type="diag",
        n_iter=n_iter,
        random_state=random_state,
        verbose=False,
    )
    model.fit(X, lengths)
    return model


def fit_scaler(X: np.ndarray) -> tuple[StandardScaler, np.ndarray]:
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    return scaler, Xs


def infer_hidden_states_viterbi(model: GaussianHMM, X: np.ndarray, lengths: list[int]) -> np.ndarray:
    # hmmlearn uses Viterbi for .predict()
    return model.predict(X, lengths)


def estimate_hidden_to_next_action(
    hidden_states: np.ndarray,
    next_observable: pd.Series,
    n_hidden: int,
) -> pd.DataFrame:
    """
    Empirical mapping: P(next_observable | hidden_state) based on inferred hidden state per timestep.
    """
    df = pd.DataFrame({"hidden": hidden_states, "next_obs": next_observable.values})
    counts = pd.crosstab(df["hidden"], df["next_obs"])
    probs = counts.div(counts.sum(axis=1), axis=0).fillna(0.0)
    probs.index.name = "hidden_state"
    return probs


def name_hidden_states(
    hidden_states: np.ndarray,
    conversion_intent_score: pd.Series,
    n_hidden: int,
) -> list[str]:
    """
    Produce stable, interpretable hidden state names by ranking states on mean conversion intent.
    """
    tmp = pd.DataFrame({"hidden": hidden_states, "intent": conversion_intent_score.values})
    intent_by_state = tmp.groupby("hidden")["intent"].mean().reindex(range(n_hidden)).fillna(tmp["intent"].mean())
    ordering = list(intent_by_state.sort_values().index)

    # Lowest → highest intent mapping
    palette = [
        "Casual Browsing",
        "Interested",
        "High Purchase Intent",
        "Impulsive Buyer",
        "Drop-off Risk",
    ]
    # pick first n_hidden, but keep the top name meaningful
    base = palette[:n_hidden]
    # reorder names according to intent ordering
    names = [None] * n_hidden
    for rank, state_id in enumerate(ordering):
        names[state_id] = base[min(rank, len(base) - 1)]
    return [str(n) for n in names]


def hidden_state_profile_table(
    df_events: pd.DataFrame,
    features_df: pd.DataFrame,
    hidden_states: np.ndarray,
    hidden_state_names: list[str] | None = None,
) -> pd.DataFrame:
    """
    Data-driven validation table for hidden states.
    Computes the requested metrics on the provided (typically TEST) set.
    """
    if len(df_events) != len(features_df) or len(df_events) != len(hidden_states):
        raise ValueError("df_events, features_df, and hidden_states must align row-wise.")

    d = df_events.copy()
    d["hidden_state"] = hidden_states
    if hidden_state_names is not None:
        d["hidden_name"] = d["hidden_state"].map({i: n for i, n in enumerate(hidden_state_names)})
    else:
        d["hidden_name"] = d["hidden_state"].astype(str)

    cart_rate = (d["state"] == "Add_to_Cart").groupby(d["hidden_state"]).mean()
    purchase_rate = (d["state"] == "Purchase").groupby(d["hidden_state"]).mean()

    # Use session_length_s from engineered features (seconds)
    session_length = features_df["session_length_s"].groupby(d["hidden_state"]).mean()
    engagement = features_df["engagement_score"].groupby(d["hidden_state"]).mean()
    intent = features_df["conversion_intent_score"].groupby(d["hidden_state"]).mean()

    table = pd.DataFrame(
        {
            "Hidden State": pd.Index(cart_rate.index, name="hidden_state").astype(int),
            "Cart Rate": cart_rate.values,
            "Purchase Rate": purchase_rate.values,
            "Session Length (s)": session_length.reindex(cart_rate.index).values,
            "Engagement": engagement.reindex(cart_rate.index).values,
            "Conversion Intent": intent.reindex(cart_rate.index).values,
        }
    )
    if hidden_state_names is not None:
        table["Interpretation"] = table["Hidden State"].map({i: n for i, n in enumerate(hidden_state_names)})
    else:
        table["Interpretation"] = ""
    return table.sort_values("Hidden State").reset_index(drop=True)


def save_artifacts(app_dir: Path, artifacts: HMMArtifacts) -> None:
    app_dir.mkdir(parents=True, exist_ok=True)
    with open(app_dir / "hmm_model.pkl", "wb") as f:
        pickle.dump(artifacts.model, f)
    with open(app_dir / "hmm_scaler.pkl", "wb") as f:
        pickle.dump(artifacts.scaler, f)
    with open(app_dir / "hmm_hidden_state_names.pkl", "wb") as f:
        pickle.dump(artifacts.hidden_state_names, f)
    artifacts.hidden_to_next_action.to_csv(app_dir / "hmm_hidden_to_next_action.csv")

