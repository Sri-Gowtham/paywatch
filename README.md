# PayWatch

UPI-style transaction fraud scoring: causal feature pipeline, calibrated XGBoost scorer, SHAP explanations, precision-target alert tiers. Built and evaluated honestly on the IEEE-CIS Fraud Detection data (time-ordered split, no look-ahead).

> Status: the **model layer is complete and audited**. The MLOps, API, monitoring, dashboard and RAG layers are **not built yet** (see Roadmap status).

## Results (held-out latest 15% of time, 88,581 transactions, 3,083 fraud)

| Model | PR-AUC | ROC-AUC | Recall @1% FPR | Precision, top 1% alerts | Amount-weighted recall |
|---|---|---|---|---|---|
| Baseline (no entity history) | 0.581 | 0.897 | 0.519 | 0.892 | 0.387 |
| Frozen design, train only, 5 seeds | 0.674 +/- 0.006 (ensemble 0.685, 95% CI 0.669-0.700) | 0.941 | 0.632 | 0.947 | 0.526 |
| Production refit (train+val), 452 features | 0.711 | 0.944 | 0.643 | 0.972 | 0.466 |
| Production refit, compact (120 features) | 0.706 | 0.948 | 0.637 | 0.968 | 0.465 |

Calibrated tiers (isotonic fitted on the first half of the test period, reported on the second half, compact model):

| Tier | Calibrated threshold | Precision | Recall | Alert rate |
|---|---|---|---|---|
| Soft flag | 0.108 | 0.601 | 0.675 | 4.4% |
| Challenge | 0.233 | 0.747 | 0.598 | 3.1% |
| Hard block | 0.732 | 0.903 | 0.465 | 2.0% |

Calibration (second half): Brier 0.0188 vs 0.0377 for a constant base rate, ECE 0.007.

## How it is built

- Everything heavy runs on **Kaggle kernels** (no local compute). Source modules in `src/features/` are uploaded as the Kaggle dataset `paywatch-src` and imported by the kernels.
- `src/features/upi_fingerprint.py`: causal per-user behaviour features (amount ratio, merchant entropy, hour deviation, new-merchant flag, ...). User/merchant are **proxies** (card fields + address + account-age anchor; payee email domain).
- `src/features/entity_history.py`: lagged fraud history per entity (uid, card, address, email, device), using only labels older than 7 days. Verified by brute-force recomputation (0 mismatches on 3,000 sampled rows).
- `src/features/engineer.py`: time-ordered 70/15/15 split, train-only frequency encoding and pruning, anchored D columns.
- Model: single XGBoost, `scale_pos_weight`, recency weights (tau 120 days). No SMOTE.
- Kernels: `kaggle_kernel_pipeline/` (audited retrain), `kaggle_kernel_b2/` (SHAP + compact model), `kaggle_kernel_verify/` (independent test of saved artifacts, all round-trip checks pass).

## Limitations (read before quoting the numbers)

- **Proxy features on non-UPI data.** IEEE-CIS is Western e-commerce. User and merchant IDs are reconstructed, not real UPI fields.
- **Base rate.** 3.5% fraud is far above real UPI. Alert rates and precision-at-alert-rate do not transfer.
- **Label feedback dependence.** The entity-history features assume fraud labels arrive within about 7 days: test PR-AUC is ~0.69 at 7 days, ~0.65 at 14, ~0.61 at 30 (baseline without history: 0.58).
- **First-time fraud is the weak spot.** 73% of test fraud is on entities with no labeled history; recall there is 0.53 (PR-AUC 0.56), vs 0.91 recall on entities with history. Repeat offenders (27% of fraud) are caught ~94%.
- **Large amounts.** Recall at 1% FPR is 0.21 for amounts >= 1000 (43 fraud rows) and 0.46 for 200-1000.
- **Short test window** (~5 weeks): weekly PR-AUC ranges 0.65-0.76.
- **Explanations.** The top SHAP drivers are human-readable entity-history features, but 17% of SHAP mass sits on opaque Vesta `V` columns.

## Negative results (tested and dropped)

Isolation Forest second layer and score fusion (alone PR-AUC 0.097; validation gave it zero weight), adversarial feature pruning (cost ~0.04 PR-AUC), velocity features, recency-decayed counts, extra entity keys (tied within seed noise), LightGBM/CatBoost blends, bagging, SMOTE (replaced by `scale_pos_weight`).

## Roadmap status

Done: scaffold, data, feature engineering, UPI fingerprint (causal proxy), XGBoost, SHAP, compact serving model, calibration + tiers.
Rejected on evidence: Isolation Forest, score fusion. Skipped: EDA notebook.
Pending: adversarial stress test, MLflow + registry, FastAPI, Docker, CI/CD, drift simulation + detection, Streamlit dashboard, calibration dashboard, LLM explanations, RAG.

## Kaggle setup

1. kaggle.com -> Settings -> API -> create a token; save it as `~/.kaggle/access_token` (never commit it).
2. `kaggle datasets version -p src/features -m "..."`, wait until `kaggle datasets files srigowthamm/paywatch-src` shows the update, then `kaggle kernels push -p kaggle_kernel_pipeline`.
