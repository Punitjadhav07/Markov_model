from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class SessionConfig:
    """How we split a visitor's events into sessions."""

    inactivity_threshold_minutes: int = 30


def sessionize_events(df: pd.DataFrame, cfg: SessionConfig = SessionConfig()) -> pd.DataFrame:
    """
    Add `session_id` by splitting each visitor's timeline into sessions where the
    time gap between consecutive events exceeds cfg.inactivity_threshold_minutes.
    """
    out = df.sort_values(["visitorid", "timestamp"]).copy()
    gap = out.groupby("visitorid")["timestamp"].diff()
    new_session = gap.isna() | (gap > pd.Timedelta(minutes=cfg.inactivity_threshold_minutes))
    out["session_id"] = new_session.groupby(out["visitorid"]).cumsum().astype("int64")
    return out


def add_state_column(df: pd.DataFrame) -> pd.DataFrame:
    state_map = {"view": "Browse", "addtocart": "Add_to_Cart", "transaction": "Purchase"}
    out = df.copy()
    out["state"] = out["event"].map(state_map)
    return out


def reduced_feature_spec() -> tuple[list[str], dict[str, str]]:
    """
    10–12 high-signal observables used for the academically-defensible HMM:
    - minimize redundancy / correlation
    - focus on intent, engagement, conversion likelihood
    """
    cols = [
        # temporal dynamics / friction
        "time_since_last_event_s",
        "avg_time_between_events_s",
        "time_since_session_start_s",
        "session_length_s",
        # engagement / depth
        "sequence_length",
        "number_of_views",
        # intent / conversion-relevant actions
        "number_of_cart_additions",
        "cart_to_view_ratio",
        "time_to_cart_s",
        "time_to_purchase_s",
        # derived (interpretable, compact summary signal)
        "engagement_score",
        "conversion_intent_score",
    ]

    justification = {
        "time_since_last_event_s": "Captures hesitation/friction; long gaps often signal drop-off risk or low intent.",
        "avg_time_between_events_s": "Summarizes pacing; fast interaction bursts vs slow drifting behavior.",
        "time_since_session_start_s": "Provides within-session phase information (early exploration vs late-stage intent).",
        "session_length_s": "Long sessions can indicate exploration; extremely short sessions often indicate low engagement.",
        "sequence_length": "Depth of interaction in the session; longer sequences typically reflect higher engagement.",
        "number_of_views": "Core browsing intensity; baseline engagement signal before cart/purchase actions.",
        "number_of_cart_additions": "Strong intent signal; cart activity is a key precursor to purchase.",
        "cart_to_view_ratio": "Normalizes cart actions by browsing volume; higher ratio implies stronger intent per exposure.",
        "time_to_cart_s": "Speed to cart from session start; faster carting often indicates higher intent.",
        "time_to_purchase_s": "Speed to purchase from session start; proxies decisiveness and conversion likelihood.",
        "engagement_score": "Compact engagement summary combining interaction depth and action severity.",
        "conversion_intent_score": "Compact intent summary emphasizing cart/purchase signals and penalizing bounces/stalls.",
    }
    return cols, justification


def select_reduced_features(features_df: pd.DataFrame) -> tuple[pd.DataFrame, list[str], pd.DataFrame]:
    """
    Return reduced feature matrix (10–12 cols) plus a justification table.
    """
    cols, justification = reduced_feature_spec()
    missing = [c for c in cols if c not in features_df.columns]
    if missing:
        raise ValueError(f"Missing required reduced features: {missing}")

    out = features_df[cols].copy()
    justification_df = (
        pd.DataFrame({"feature": cols, "why_it_matters": [justification[c] for c in cols]})
        .set_index("feature")
    )
    return out, cols, justification_df


def build_event_level_features(df: pd.DataFrame, feature_set: str = "full") -> tuple[pd.DataFrame, list[str]]:
    """
    Build >=20 engineered observable features per event (behavioral + temporal).
    Returns:
      - features_df: numeric feature matrix (rows align with df rows)
      - feature_cols: list of feature column names in order
    """
    d = df.copy()
    d = d.sort_values(["visitorid", "session_id", "timestamp"]).reset_index(drop=True)

    # Basic timing features
    d["time_since_last_event_s"] = (
        d.groupby(["visitorid", "session_id"])["timestamp"].diff().dt.total_seconds().fillna(0.0)
    )
    session_start = d.groupby(["visitorid", "session_id"])["timestamp"].transform("min")
    session_end = d.groupby(["visitorid", "session_id"])["timestamp"].transform("max")
    d["time_since_session_start_s"] = (d["timestamp"] - session_start).dt.total_seconds()
    d["remaining_time_in_session_s"] = (session_end - d["timestamp"]).dt.total_seconds()

    # Session aggregates repeated per row (useful as observables)
    d["sequence_length"] = d.groupby(["visitorid", "session_id"])["state"].transform("size").astype("int64")
    d["event_idx_in_session"] = d.groupby(["visitorid", "session_id"]).cumcount().astype("int64")
    d["bounce"] = (d["sequence_length"] == 1).astype("int64")
    d["session_length_s"] = (session_end - session_start).dt.total_seconds()
    d["avg_time_between_events_s"] = d["session_length_s"] / np.maximum(d["sequence_length"] - 1, 1)

    # Counts and ratios (cumulative within session)
    is_view = (d["state"] == "Browse").astype("int64")
    is_cart = (d["state"] == "Add_to_Cart").astype("int64")
    is_purchase = (d["state"] == "Purchase").astype("int64")
    d["number_of_views"] = is_view.groupby([d["visitorid"], d["session_id"]]).cumsum()
    d["number_of_cart_additions"] = is_cart.groupby([d["visitorid"], d["session_id"]]).cumsum()
    d["number_of_purchases"] = is_purchase.groupby([d["visitorid"], d["session_id"]]).cumsum()
    d["cart_to_view_ratio"] = d["number_of_cart_additions"] / d["number_of_views"].clip(lower=1)

    # Flags
    d["purchase_flag"] = is_purchase
    d["exit_flag"] = (
        d.groupby(["visitorid", "session_id"])["event_idx_in_session"].transform("max") == d["event_idx_in_session"]
    ).astype("int64")

    # Temporal calendar features
    d["hour"] = d["timestamp"].dt.hour.astype("int64")
    d["day_of_week"] = d["timestamp"].dt.dayofweek.astype("int64")  # 0=Mon
    d["is_morning"] = d["hour"].between(5, 11).astype("int64")
    d["is_afternoon"] = d["hour"].between(12, 17).astype("int64")
    d["is_night"] = (~(d["is_morning"].astype(bool) | d["is_afternoon"].astype(bool))).astype("int64")

    # Last event type (one-hot of previous state)
    d["prev_state"] = d.groupby(["visitorid", "session_id"])["state"].shift(1).fillna("START")
    prev_dummies = pd.get_dummies(d["prev_state"], prefix="prev", dtype="int64")

    # Revisit flag: visitor has >1 session overall
    sessions_per_visitor = d.groupby("visitorid")["session_id"].nunique()
    d["revisit_flag"] = d["visitorid"].map((sessions_per_visitor > 1).astype("int64"))

    # Time-to-cart and time-to-purchase (session-level, repeated)
    first_cart_time = d.loc[d["state"] == "Add_to_Cart"].groupby(["visitorid", "session_id"])["timestamp"].min()
    first_purchase_time = d.loc[d["state"] == "Purchase"].groupby(["visitorid", "session_id"])["timestamp"].min()
    d["time_to_cart_s"] = (
        (d.set_index(["visitorid", "session_id"]).index.map(first_cart_time) - session_start).dt.total_seconds()
    )
    d["time_to_purchase_s"] = (
        (d.set_index(["visitorid", "session_id"]).index.map(first_purchase_time) - session_start).dt.total_seconds()
    )
    d["time_to_cart_s"] = d["time_to_cart_s"].fillna(-1.0)
    d["time_to_purchase_s"] = d["time_to_purchase_s"].fillna(-1.0)

    # Derived metrics: engagement + conversion intent (simple, interpretable)
    d["total_events"] = d["sequence_length"]
    d["engagement_score"] = (
        0.2 * d["total_events"]
        + 0.6 * d["number_of_views"]
        + 1.5 * d["number_of_cart_additions"]
        + 3.0 * d["number_of_purchases"]
    )
    d["conversion_intent_score"] = (
        2.0 * d["number_of_cart_additions"]
        + 4.0 * d["number_of_purchases"]
        - 0.5 * d["bounce"]
        - 0.2 * (d["time_since_last_event_s"] > 600).astype("int64")
    )

    # Assemble final feature matrix
    base_cols = [
        "session_length_s",
        "time_since_last_event_s",
        "time_since_session_start_s",
        "remaining_time_in_session_s",
        "avg_time_between_events_s",
        "sequence_length",
        "event_idx_in_session",
        "total_events",
        "number_of_views",
        "number_of_cart_additions",
        "number_of_purchases",
        "cart_to_view_ratio",
        "purchase_flag",
        "exit_flag",
        "bounce",
        "revisit_flag",
        "time_to_cart_s",
        "time_to_purchase_s",
        "hour",
        "day_of_week",
        "is_morning",
        "is_afternoon",
        "is_night",
        "engagement_score",
        "conversion_intent_score",
    ]

    features_df = pd.concat([d[base_cols].astype("float64"), prev_dummies.astype("float64")], axis=1)
    if feature_set == "full":
        return features_df, list(features_df.columns)
    if feature_set == "reduced":
        reduced_df, cols, _ = select_reduced_features(features_df)
        return reduced_df, cols
    raise ValueError("feature_set must be 'full' or 'reduced'")

