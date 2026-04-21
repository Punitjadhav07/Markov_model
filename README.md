# Customer Behavior: Markov vs Hidden Markov 

This project compares:
- **Markov Chain** for observable next-state prediction
- **Hidden Markov Model (HMM)** for latent intent prediction

## Structure:-

```text
customer_prediction/
├── data/
│   ├── event.csv        # 100 rows (observable events)
│   └── newdata.csv      # 100 rows (engineered features for HMM)
├── notebook/
│   ├── markov.ipynb
│   └── hmm.ipynb
├── app.py               # Streamlit UI
└── README.md
```

## Markov baseline:-

From `data/event.csv`:
1. Map events: `view -> Browse`, `addtocart -> Add_to_Cart`, `transaction -> Purchase`
2. Build visitor sequences and append `Exit`
3. Compute transition probabilities with crosstab + row normalization
4. Predict next state from current observable state

## HMM intent model

From `data/newdata.csv` (100 rows), we use:
- `session_length_s`
- `avg_time_between_events_s`
- `cart_to_view_ratio`
- `number_of_cart_additions`
- `number_of_purchases`
- `time_to_cart_s`
- `time_to_purchase_s`
- `engagement_score`
- `conversion_intent_score`

Model:
- `GaussianHMM(n_components=3)` + standard scaling
- Hidden states are named by average `conversion_intent_score`:
  - Low Intent
  - Medium Intent
  - High Intent

## How intent is calculated

The app estimates intent from selected sequence events using:
- `engagement_score = 0.6 * views + 1.5 * cart_adds + 3.0 * purchases`
- `conversion_intent_score = 2.0 * cart_adds + 4.0 * purchases`

These values form part of the HMM feature vector for hidden-state inference.

## Streamlit UI

Input:
- Dropdown sequence pattern + editable sequence text

Output:
- Markov next-state probabilities
- HMM inferred intent (Low/Medium/High)
- HMM next likely action probabilities
- Markov and HMM heatmaps

## Run

```bash
cd customer_prediction
source .venv/bin/activate
streamlit run app.py
```

## Notebooks

```bash
jupyter notebook notebook/markov.ipynb
jupyter notebook notebook/hmm.ipynb
```
