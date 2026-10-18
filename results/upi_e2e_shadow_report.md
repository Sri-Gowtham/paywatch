# Shadow pilot report: FAIL

Decisions 29,107, mature 20,319, confirmed frauds 207. Label window 7.0 days. a decision older than 7 days with no fraud label is treated as legitimate

| Metric | Value |
|---|---|
| PR-AUC | 0.810 (95% CI 0.745 to 0.861) |
| ROC-AUC | 0.997 |
| Recall at 1% false-positive rate | 0.874 |
| Precision / recall at a 1% alert budget | 0.764 / 0.749 |
| Brier (model / constant base rate) | 0.0036 / 0.0101 |

| Tier target | Threshold | Alert rate | Precision | Recall |
|---|---|---|---|---|
| 0.5 | 0.083 | 2.11% | 0.437 | 0.903 |
| 0.7 | 0.300 | 1.09% | 0.724 | 0.773 |

Criteria (fixed before the pilot): pr_auc met, recall_at_fpr_1pct met, hard_block_precision NOT met

Drift: probability PSI 0.000; most shifted features: h7_payer_fraud_rate (1.46), h7_device_fraud_rate (1.34), h7_payee_fraud_rate (0.69), payee_distinct_payers (0.25), h7_pair_fraud_rate (0.19)