# PayWatch shadow-mode pilot guide

This guide is for a bank, payment app or fintech that wants to test PayWatch on its **own** UPI traffic. In shadow mode PayWatch scores every transaction and records what it would have recommended, but it never blocks, delays or changes a payment. After the labels (confirmed fraud) arrive, the pilot is judged against criteria written down beforehand.

**What exists today:** the toolkit in `src/upi/` (training kit, shadow-mode service, evaluation) and a synthetic end-to-end run that proves the plumbing. **What does not exist:** any result on real UPI traffic. The model in the live demo was trained on a public e-commerce dataset and cannot score UPI traffic; a pilot trains a new model on the partner's data, inside the partner's environment.

## 1. What the partner provides

One table of historical transactions for training, then a live feed of new ones.

| Field | Required | Notes |
|---|---|---|
| transaction id | yes | any unique string; used to join labels later |
| timestamp | yes | Unix seconds, time of the payment |
| amount | yes | rupees |
| payer id | yes | **pseudonymised** (see section 3) |
| payee id | yes | **pseudonymised** |
| device id | optional | **pseudonymised**; strongly useful against account takeover and mule rings |
| channel | optional | `app`, `qr`, `collect`, `intent` |
| confirmed fraud label | training and evaluation | 0/1 from the partner's case system |

**Volume.** What matters is the number of confirmed frauds, not rows. As a rule of thumb the width of a confidence interval shrinks with the square root of the number of frauds. On IEEE-CIS, about 3,000 frauds in the test period gave a PR-AUC interval of roughly plus or minus 0.015; with a few hundred frauds expect an interval several times wider, and with fewer than about 100 the evaluation reports INCONCLUSIVE instead of a verdict. The training kit refuses a time slice with too few frauds (default 20 per slice). Aim for several months of history so each time slice has enough.

**Label delay.** Fraud confirmation is slow. The features use fraud history only from labels older than a lag (default 7 days), and the evaluation only counts decisions whose label window has closed. Tell us your real label delay and set `label_lag_days` and `--window-days` to it.

## 2. What runs where

Everything runs inside the partner's environment. No data is sent to Kaggle or to the PayWatch authors.

1. **Train** (once, then periodically): `python -m src.upi.train --data transactions.csv --out models/upi --config config.json` (see `src/upi/config.example.json`). It writes the model, calibration, tiers, a warm state and a results file. Needs Python, numpy and lightgbm.
2. **Serve in shadow mode**: `uvicorn --factory src.upi.main:create_upi_app --workers 1` with the environment variables below. One worker only, because the history state lives in memory and is saved to disk.
3. **Send each transaction** to `POST /score` as it happens (or from a replayed copy of the feed). Send each confirmed-fraud label to `POST /feedback` when it is known.
4. **Evaluate**: `python -m src.upi.shadow_eval --log "logs/decisions.jsonl*" --model-dir models/upi --criteria criteria.json --out report/`.

| Variable | Purpose |
|---|---|
| `PAYWATCH_UPI_MODEL_DIR` | folder written by the training kit |
| `PAYWATCH_API_KEYS` | comma-separated API keys; the service refuses to start without them |
| `PAYWATCH_LOG_SALT` | secret that keys the transaction-id hash in the log; keep it constant |
| `PAYWATCH_SHADOW_LOG` | path of the decision log |
| `PAYWATCH_STATE_PATH` | where the state snapshot is saved (autosave every 300 s and on shutdown) |
| `PAYWATCH_LOG_FEATURES` | `1` to also log the feature vectors (off by default, see section 3) |

## 3. Privacy design (and what you must still decide)

- **Pseudonymise before anything reaches PayWatch.** `src/upi/schema.py` has `pseudonymise(value, salt)` (a keyed hash). The API and the training loader reject identifiers that look raw (a VPA containing `@`, a phone number). A rejected request's error message never echoes the submitted value.
- **The decision log contains** a keyed hash of the transaction id, the model's score, calibrated probability and tier, and timestamps. It does **not** contain payer, payee or device identifiers, or the amount.
- **Feature vectors are off by default.** If you turn `PAYWATCH_LOG_FEATURES` on (needed only for feature-level drift analysis) the log will contain the amount, the hour and behavioural aggregates. Turn it on only if your data policy allows it.
- **The state snapshot contains your pseudonymous identifiers** and counters built from them. Treat it as personal data under your own classification: restrict access, encrypt the disk, set a retention policy.
- **Authentication:** every endpoint except `/health` needs the `X-API-Key` header. Put the service behind your network controls and TLS; PayWatch does not terminate TLS.

## 4. Rollout stages

1. **Offline replay.** Train on history, read the holdout report (`results.json`). Nothing is deployed.
2. **Shadow, at least 30 days.** Live scoring with no effect on payments. Collect labels as they confirm.
3. **Analyst review.** Analysts look at the top-tier alerts alongside their normal process and record whether the explanations help. Still no automatic action.
4. **Limited action, only if the criteria were met.** A human decision, outside this toolkit.

**Kill switch:** stop sending transactions to `/score` (or stop the service). Because shadow mode never sits in the payment path, nothing else needs to change.

## 5. Judging the pilot

Write `criteria.json` before stage 2 and do not change it afterwards:

```json
{"min_frauds": 150, "min_pr_auc": 0.35, "min_recall_at_fpr_1pct": 0.40, "min_hard_block_precision": 0.80}
```

The report states PASS, FAIL or INCONCLUSIVE with the numbers, a bootstrap interval, tier precision and recall, calibration and probability drift. A decision older than the label window with no fraud label is counted as legitimate; if fraud reports arrive later than your window, precision is overstated, so choose the window from your real data.

## 6. Known limits (from this project's own results)

- On IEEE-CIS, recall on **first-time fraud was about 57%**, and a **fresh identity evaded about 31%** of flagged fraud with a single probe. Entity-history features are the main signal, so rings that rotate identities will be harder.
- Performance depends on label delay: PR-AUC fell from about 0.69 at 7 days of delay to about 0.61 at 30.
- Alert rates and precision depend heavily on the fraud base rate, which differs between datasets and from UPI.
- Fairness across customer groups has not been assessed.
- The end-to-end run in `results/upi_e2e.json` uses synthetic traffic written by us, with fraud patterns we injected. It shows that the pipeline works, not that PayWatch detects real UPI fraud.

## 7. Questions for the partner's legal and compliance team

These are questions to ask, not answers: PayWatch's authors do not give legal advice.

- Which RBI, NPCI and data-protection requirements apply to using transaction data for model development and shadow scoring, and does pseudonymised data still count as personal data for them?
- Must the data and the scoring service stay in India or inside the partner's own infrastructure, and may any third party (a cloud provider, a contractor) touch it?
- What is the lawful basis and the customer-notice position for scoring transactions this way?
- How long may the decision log and the state snapshot be kept, and who may read them?
- What approval is needed before any score influences a payment?
