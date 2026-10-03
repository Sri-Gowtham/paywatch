# PayWatch

UPI-style transaction fraud scoring: causal feature pipeline, calibrated XGBoost scorer, SHAP explanations, precision-target alert tiers. Built and evaluated honestly on the IEEE-CIS Fraud Detection data (time-ordered split, no look-ahead).

> Status: model, MLOps registry, serving API, drift monitoring, dashboard, adversarial stress test and a retrieval-grounded explanation layer are **built and tested** (on Kaggle). Docker and GitHub CI are written; the CI failure found on the first push is fixed locally but not yet pushed, so neither has been seen passing on GitHub (see Roadmap status).

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
- Kaggle scripts: `src/models/train_pipeline.py` (audited retrain), `shap_compact.py` (SHAP + compact model), `verify_artifacts.py` (independent test of saved artifacts, all round-trip checks pass). Kernel metadata lives in `kaggle/kernels.json`; run one with `python tools/kaggle_tools.py push <kernel> --wait`. The small result files behind the numbers in this README are in `results/`.

## Repository layout

```
src/features/   causal feature pipeline (uploaded to Kaggle as dataset `paywatch-src`)
src/models/     Kaggle training / evaluation / study scripts, MLflow registry code, archive/ = superseded experiments
src/api/        FastAPI scoring service, streaming state, warm-start snapshot
src/monitor/    drift detector, alert rules, simulation, Streamlit dashboard
src/rag/        RBI knowledge base: ingest, BM25 retriever, answers, explanations
tests/          pytest suite; tests/kaggle/ = runner scripts executed on Kaggle (API replay, dashboard check, CI reproduction)
models/compact/ production model files + encoders served by the API
kaggle/         kernel and dataset registries;  tools/kaggle_tools.py  push / fetch / version helper
results/        small result JSONs from the Kaggle runs (provenance of the numbers in this README)
mlflow/         MLflow experiment store and model registry;  docker/  image + compose;  data/sim/  simulation results
```

## Serving, warm start, monitoring and dashboard

- **API** (`src/api/`): `/predict`, `/feedback`, `/health`, `/model-info`; calibrated tiers; SHAP reasons. The streaming user-behaviour state and entity fraud history are parity-tested against the batch pipeline (12 tests pass on Kaggle). Server latency about 3 ms per request (p99 4.5 ms), about 24 ms with SHAP reasons. State lives in process memory, so run one worker. Models and encoders are in `models/compact/`.
- **Warm start** (`src/api/snapshot.py`): an empty state scores every user as new (replay PR-AUC 0.548); loading the state built from all train+val history gives 0.690 (offline 0.706). The 7.4 MB snapshot is derived from competition data and is kept on Kaggle only; rebuild it with `src/models/build_warm_state.py`.
- **Drift monitoring** (`src/monitor/`): PSI + KS on the 84 client-input features (state that grows by design is excluded), score drift, flag-rate shift, and label-delayed performance with bootstrap noise. 30-day simulation with drift injected on day 15 (`src/monitor/run_simulation.py`, results in `data/sim/sim_results.json`): the no-drift control raised 0 alerts at the chosen 3-day window (the window was chosen on the control only; 1-day and 2-day windows raised 6 and 4); the injected drift raised a warning on day 17 and a critical alert on day 18, with 0 alerts before day 15. Score-drift and performance alerts never fired: the injected drift did not measurably hurt PR-AUC. Rolling PR-AUC decays from 0.79 to 0.63 in both runs and is not caught by the 15% rule. Thresholds come from one control stream with overlapping windows, a weak false-alarm estimate.
- **Dashboard** (`streamlit run src/monitor/dashboard.py`): live scoring against the API, drift timeline with alerts, performance and calibration, and a threshold slider with a cost trade-off. Passes 13 automated checks on Kaggle (`tests/kaggle/dashboard_check.py`), including the assistant tab.

## High-confidence alerts: can precision reach 0.99?

Pre-registered test (`src/models/high_precision_study.py`, results in `results/hp.json`): pick the deepest alert threshold on the first half of the test period whose precision has a Clopper-Pearson 95% lower bound of at least 0.98, then require precision >= 0.99 and lower bound >= 0.98 on the held-out second half. **Not met.** The best variant (minimum over the 3 seeds) reached precision 0.970 (95% CI 0.946-0.986) at 18.8% recall on the held-out half. Held-out operating points: top 0.25% of alerts, 111 of 111 correct (CI 0.967-1.000, recall 6.4%); top 0.5%, precision 0.982 (0.955-0.995, recall 12.6%); top 1%, 0.971 (recall 24.8%). About 79% of the highest-confidence alerts are repeat offenders; recall on first-time fraud at this precision is 4.5%, and precision dipped to 0.936 in the fourth week. Do not quote 0.99 precision for this system.

## Retrieval-grounded explanations and analyst assistant (RAG)

`src/rag/` indexes six public RBI documents (FREE-AI report, Digital Payment Security Controls, Limiting Liability circular, Fraud Risk Management Master Directions and FAQs, Payments Vision 2025; 445 passages; list in `src/rag/sources.py`). The index is built on Kaggle (`src/rag/build_index.py`, results in `results/rag.json`) and kept out of the repo (`data/rag/`, gitignored).

- **Retriever: BM25, pure Python.** On 18 hand-labeled questions (a hit means a top-k passage from an expected document contains an accepted key phrase): BM25 finds it in the top 1 / 3 / 5 for 61% / 94% / 100%; a dense MiniLM retriever gets 61% / 78% / 89%; BM25+dense hybrid 67% / 94% / 100%. The dense model is not worth its weight, so serving needs no torch.
- **Assistant answers are extractive by default** (best matching sentences with numbered citations; no API key needed). 14 of the 18 answers contain the key phrase (78%), and all 4 off-topic questions (GST on gold, biryani, ...) are refused: a question is answered only when the retrieved passage covers at least half of its terms. With `ANTHROPIC_API_KEY` set the answer is composed by an LLM constrained to the excerpts; that path is unit-tested with a stub only, never run against the real API.
- **Decision explanations** (`src/rag/explainer.py`): a fixed template over the API response (action, calibrated probability, the top SHAP reasons, history flags) plus up to three on-topic RBI passages (a passage must contain the key term for the action, e.g. "authentication" for a challenge). Nothing is generated freely; the template states that attributions are not proof of intent.
- Dashboard: fifth tab "Analyst assistant" (question box with citations, and an explanation of the last transaction scored in the first tab). Limits: 18 questions is a small, hand-made evaluation; extractive answers can quote the right passage but not the exact sentence (the citation always shows the source); ChromaDB was not used because 445 passages do not need a vector database.

## Exploratory data analysis

`notebooks/01_eda.ipynb` (generated and executed on Kaggle by `src/models/eda_notebook.py`, so every number and plot is computed): 590,540 transactions over 182 days, 3.5% fraud; weekly fraud rate swings between 1.9% and 5.1% (time-ordered splits are required); credit cards (6.7% fraud) and mobile devices (10.2%) are riskier than debit (2.4%) and rows without identity data (2.1% vs 7.9% with it); only 24% of rows carry identity information and 214 of 438 columns are more than half missing; 98% of fraud sits on a card composite that had an earlier transaction and 81% follows an earlier fraud on the same card by at least 7 days, which is what motivated the lagged entity-history features.

## Adversarial stress test

Black-box evasion test (`src/models/stress_test.py`, results in `results/stress_test.json`): an attacker who can probe the scorer 1, 5 or 20 times and change only fields they control (amount, payee e-mail domain, device/browser, product and card type, plus the state those drive) tries to push the 1,500 test-period frauds that the model currently flags below the alert tiers. Each move copies a legitimate donor row's values for the affected features.

| Attack (share of flagged frauds pushed below SOFT_FLAG) | 1 probe | 5 probes | 20 probes |
|---|---|---|---|
| Amount mimicry | 4.2% | 7.1% | 8.5% |
| Payee e-mail rotation | 6.1% | 13.7% | 20.7% |
| Device / browser rotation | 1.3% | 4.4% | 9.4% |
| Behaviour mimicry | 1.5% | 3.4% | 5.2% |
| Product / card swap | 3.5% | 5.1% | 5.9% |
| **Fresh identity (no entity history)** | **31.7%** | 36.4% | 40.6% |
| All moves combined | 42.9% | 52.7% | **58.4%** |

With all moves and 20 probes, 68.7% fall below CHALLENGE and 92.0% below HARD_BLOCK, so the model's overall recall at SOFT_FLAG could fall from about 69% to roughly 29% against a strong adaptive attacker. The dominant weakness is the entity-history dependence: a fresh identity alone evades 32% with a single probe. Probing helps the attacker (31.7% to 40.6% from 1 to 20 queries), so do not expose raw scores to clients.

**The roadmap's "two layers are necessary" claim is not supported.** An Isolation Forest second layer at 1% false-positive rate flagged 0.0% of the model-flagged frauds, 0.0% of the attacked frauds and 0.0% of those that evaded; an OR rule with the model leaves recall unchanged and raises the false-positive rate from 1.9% to 2.9%.

**Adversarial training against the fresh-identity weakness** (`src/models/adversarial_training.py`, results in `results/adv_training.json`). Pre-registered criterion: fresh-identity evasion at 1 probe below 20% with test PR-AUC loss of at most 0.01 and recall@1%FPR loss of at most 0.02. Adding rotated copies of every training fraud met it: evasion fell from 32.9% to 0.0% (all moves combined, 20 probes: 69.5% to 0.4%) while test PR-AUC moved 0.675 to 0.679, recall@1%FPR 0.617 to 0.616 and the precision of flagged alerts rose by 1.1 points. Limits: the defence is specific to the trained transformation (payee e-mail rotation still evades 26% of flagged fraud at 20 probes, baseline 30%) and recall on genuinely first-time fraud did not improve (0.570 vs 0.569); the production models in `models/compact/` do not include this training yet.

Limits of the stress test: random-search attacker in feature space (no gradient attack, no real attacker data); it cannot directly alter the opaque Vesta `V`/`C`/`D` features; the 31% of fraud the model already misses is not included in the targets.

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

Done: scaffold, data, feature engineering, UPI fingerprint (causal proxy), XGBoost, SHAP, compact serving model, calibration + tiers, MLflow + registry, FastAPI, production simulation, drift detection, Streamlit dashboard (the threshold-tuning tab covers the calibration dashboard).
Also done: adversarial stress test, RAG ingestion and query (BM25), retrieval-grounded explanations. The LLM explanation step is optional and only template mode has run against real data (done differently).
Written but not yet seen passing on GitHub: Dockerfile, GitHub Actions CI/CD. The first CI run failed on a pandas 3 incompatibility in the feature code; it is fixed and committed locally (42 of 42 tests pass in a CI-style environment on Kaggle, and a stand-in for the image's runtime passes: with only `requirements-api.txt` installed and every other package hidden, the API starts with the image's startup command and serves `/health` and `/predict`) but not pushed.
Rejected on evidence: Isolation Forest, score fusion. The EDA notebook is done (`notebooks/01_eda.ipynb`).

## Kaggle setup

1. kaggle.com -> Settings -> API -> create a token; save it as `~/.kaggle/access_token` (never commit it).
2. `python tools/kaggle_tools.py version features -m "..."` uploads the changed source folder as a Kaggle dataset (keys in `kaggle/datasets.json`); wait until `kaggle datasets files srigowthamm/paywatch-src` shows the update, then `python tools/kaggle_tools.py push pipeline --wait`. `python tools/kaggle_tools.py list` shows every kernel and dataset; `fetch <kernel>` downloads only its small `results.json`.
