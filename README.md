# Customer Purchase Behavior Analysis with Markov Chains

This project models customer journey behavior from event logs and uses a first-order Markov Chain to estimate the probability of the next user action.

## Project Structure

- `notebooks/analysis.ipynb` - end-to-end data analysis and Markov model walkthrough.
- `app/app.py` - Streamlit dashboard for KPI and transition insights.
- `Data/events.csv` - input dataset (local, not tracked in git by default).

## Problem Statement

Given user events (`view`, `addtocart`, `transaction`) with timestamps, the objective is to:

1. Build ordered event sequences per customer.
2. Convert events into behavior states.
3. Learn transition probabilities between states.
4. Use those probabilities for customer behavior insights and next-step prediction.

## How Markov Model Is Used Here

### 1) State Definition

Raw events are mapped to states:

- `view` -> `Browse`
- `addtocart` -> `Add_to_Cart`
- `transaction` -> `Purchase`

Each user journey is terminated with an `Exit` state.

### 2) Sequence Construction

For each `visitorid`, events are sorted by timestamp to form a sequence:

`[Browse, Browse, Add_to_Cart, Purchase, Exit]`

### 3) Transition Extraction

Adjacent state pairs are counted:

- `(Browse -> Add_to_Cart)`
- `(Add_to_Cart -> Purchase)`
- `(Purchase -> Exit)`, etc.

This creates a transition count matrix.

### 4) Probability Matrix (Markov Transition Matrix)

Each row in the count matrix is normalized so row sum = 1:

`P(next_state | current_state)`

This matrix powers:

- next-state prediction,
- dropout analysis (`Browse -> Exit`),
- conversion flow analysis (`Browse -> Add_to_Cart -> Purchase`).

### 5) Dashboard Insights

The Streamlit dashboard surfaces:

- customer and event KPIs,
- event mix distribution,
- daily event trend,
- transition probability heatmap,
- most frequent transitions.

## Setup

From project root:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run Notebook

```bash
jupyter notebook
```

Open `notebooks/analysis.ipynb` and run cells top-to-bottom.

## Run Dashboard

```bash
streamlit run app/app.py
```

## Notes on Data

- Notebook expects: `../Data/events.csv` relative to `notebooks/analysis.ipynb`.
- Dashboard expects: `Data/events.csv` relative to project root.
- If data file is missing, place the dataset at `Data/events.csv`.

## Expected Outcomes

- A reproducible data pipeline for customer event processing.
- A valid Markov transition matrix for behavior modeling.
- A professional dashboard for business-facing interpretation.
