# PayWatch

UPI-style transaction fraud scoring: causal feature pipeline, calibrated LightGBM scorer, SHAP explanations, precision-target alert tiers. Built and evaluated honestly on the IEEE-CIS Fraud Detection data (time-ordered split, no look-ahead).

> Status: model, MLOps registry, serving API, drift monitoring, dashboard, adversarial stress test and a retrieval-grounded explanation layer are **built and tested** (on Kaggle). Docker and GitHub CI pass on GitHub (tests, image build and a container `/health` smoke test) and a push that passes deploys automatically. A live demo of the API runs at https://paywatch-api-fpq9.onrender.com (free plan: the first request after idle takes about a minute, and user history resets on restart). The production model is a 3-seed LightGBM, selected by a pre-registered study (see Model improvement study); it replaced the earlier XGBoost model.

## Results (held-out latest 15% of time, 88,581 transactions, 3,083 fraud)

| Model | PR-AUC | ROC-AUC | Recall @1% FPR | Precision, top 1% alerts | Amount-weighted recall |
|---|---|---|---|---|---|
| Baseline (no entity history) | 0.581 | 0.897 | 0.519 | 0.892 | 0.387 |
| Frozen design, train only, 5 seeds | 0.674 +/- 0.006 (ensemble 0.685, 95% CI 0.669-0.700) | 0.941 | 0.632 | 0.947 | 0.526 |
| Production refit (train+val), 452 features | 0.711 | 0.944 | 0.643 | 0.972 | 0.466 |
| Previous production model: XGBoost 3-seed refit, compact (120 features) | 0.706 | 0.948 | 0.637 | 0.968 | 0.465 |
| **Production now: LightGBM 3-seed refit, compact (120 features)** | **0.724** | **0.956** | **0.653** | not recomputed | not recomputed |

Calibrated tiers for the production LightGBM model (isotonic fitted on the first half of the test period, reported on the second half; the previous XGBoost tiers are in brackets):

| Tier | Calibrated threshold | Precision | Recall | Alert rate |
|---|---|---|---|---|
| Soft flag | 0.126 (0.108) | 0.618 (0.601) | 0.694 (0.675) | 4.4% (4.4%) |
| Challenge | 0.200 (0.233) | 0.783 (0.747) | 0.620 (0.598) | 3.1% (3.1%) |
| Hard block | 0.700 (0.732) | 0.924 (0.903) | 0.460 (0.465) | 1.9% (2.0%) |

On the same second half the LightGBM model scores PR-AUC 0.731 and recall at 1% FPR 0.650, against 0.705 and 0.622 for the XGBoost model it replaced (`results/prod_lgbm.json`). Calibration figures (Brier, ECE) were measured on the XGBoost model and have not been recomputed for LightGBM.

## How it is built

- Everything heavy runs on **Kaggle kernels** (no local compute). Source modules in `src/features/` are uploaded as the Kaggle dataset `paywatch-src` and imported by the kernels.
- `src/features/upi_fingerprint.py`: causal per-user behaviour features (amount ratio, merchant entropy, hour deviation, new-merchant flag, ...). User/merchant are **proxies** (card fields + address + account-age anchor; payee email domain).
- `src/features/entity_history.py`: lagged fraud history per entity (uid, card, address, email, device), using only labels older than 7 days. Verified by brute-force recomputation (0 mismatches on 3,000 sampled rows).
- `src/features/engineer.py`: time-ordered 70/15/15 split, train-only frequency encoding and pruning, anchored D columns.
- Model: 3-seed LightGBM (529 rounds, 127 leaves, `scale_pos_weight`, recency weights with tau 120 days) refit on train + validation; `src/models/prod_lgbm.py` builds it. No SMOTE. The earlier model was a 3-seed XGBoost (`train_pipeline.py`).
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

- **API** (`src/api/`): `/predict`, `/feedback`, `/health`, `/model-info`, `/example`; calibrated tiers; SHAP reasons. The streaming user-behaviour state and entity fraud history are parity-tested against the batch pipeline (19 API and parity tests pass in the Kaggle replay harness; the full suite runs in GitHub Actions). A 20,000-request replay on the LightGBM model had 0 errors: server latency about 2.4 ms per request (p99 3.5 ms) without explanations, but about 141 ms (p99 150 ms) with SHAP reasons, because LightGBM's tree-SHAP on 3 models x 529 trees is slower than XGBoost's was (about 24 ms); send `"explain": false` when only the score is needed. Explaining from one seed instead of three would cut that cost by about two thirds, but the top-5 reasons then overlap only 87% with the three-seed average (identical set 44%, same top reason 94%, notebook `04`), so the API keeps the three-seed attributions. State lives in process memory, so run one worker. Models and encoders are in `models/compact/`.
- **Warm start** (`src/api/snapshot.py`): an empty state scores every user as new (replay PR-AUC 0.573); loading the state built from all train+val history gives 0.711 (offline test PR-AUC of the refit model 0.724; with the XGBoost model these were 0.548, 0.690 and 0.706). The 7.4 MB snapshot is derived from competition data and is kept on Kaggle only; rebuild it with `src/models/build_warm_state.py`.
- **Drift monitoring** (`src/monitor/`): PSI + KS on the 84 client-input features (state that grows by design is excluded), score drift, flag-rate shift, and label-delayed performance with bootstrap noise. 30-day simulation with drift injected on day 15 (`src/monitor/run_simulation.py`, results in `data/sim/sim_results.json`), run on the LightGBM model: the window is chosen on the no-drift control only, as the smallest window with zero alerts. The control raised 5, 3 and 1 alerts at 1, 2 and 3 days and none at 5 or 7 days, so the **5-day window** was chosen (with the XGBoost model 3 days was alarm-free; the window set was extended to 5 and 7 days after the model switch, because the 3-day control raised one flag-rate warning at 1.58x against a 1.5x limit). At the 5-day window the control raised 0 alerts and the injected drift raised a warning on day 18 and a critical alert on day 20, with 0 alerts before day 15 (the 3-day window would have flagged it on days 17 and 18). Score-drift and performance alerts never fired: the injected drift did not measurably hurt PR-AUC. Rolling PR-AUC decays from 0.79 to 0.70 in the control and 0.69 in the drift run and is not caught by the 15% rule. Thresholds come from one control stream with overlapping windows, a weak false-alarm estimate.
- **Dashboard** (`streamlit run src/monitor/dashboard.py`): live scoring against the API, drift timeline with alerts, performance and calibration, and a threshold slider with a cost trade-off. Passes 13 automated checks on Kaggle (`tests/kaggle/dashboard_check.py`), including the assistant tab.

## High-confidence alerts: can precision reach 0.99?

Pre-registered test on the previous XGBoost model (`src/models/high_precision_study.py`, results in `results/hp.json`; not repeated for LightGBM): pick the deepest alert threshold on the first half of the test period whose precision has a Clopper-Pearson 95% lower bound of at least 0.98, then require precision >= 0.99 and lower bound >= 0.98 on the held-out second half. **Not met.** The best variant (minimum over the 3 seeds) reached precision 0.970 (95% CI 0.946-0.986) at 18.8% recall on the held-out half. Held-out operating points: top 0.25% of alerts, 111 of 111 correct (CI 0.967-1.000, recall 6.4%); top 0.5%, precision 0.982 (0.955-0.995, recall 12.6%); top 1%, 0.971 (recall 24.8%). About 79% of the highest-confidence alerts are repeat offenders; recall on first-time fraud at this precision is 4.5%, and precision dipped to 0.936 in the fourth week. Do not quote 0.99 precision for this system.

## Retrieval-grounded explanations and analyst assistant (RAG)

`src/rag/` indexes six public RBI documents (FREE-AI report, Digital Payment Security Controls, Limiting Liability circular, Fraud Risk Management Master Directions and FAQs, Payments Vision 2025; 445 passages; list in `src/rag/sources.py`). The index is built on Kaggle (`src/rag/build_index.py`, results in `results/rag.json`) and kept out of the repo (`data/rag/`, gitignored).

- **Retriever: BM25, pure Python.** On 18 hand-labeled questions (a hit means a top-k passage from an expected document contains an accepted key phrase): BM25 finds it in the top 1 / 3 / 5 for 61% / 94% / 100%; a dense MiniLM retriever gets 61% / 78% / 89%; BM25+dense hybrid 67% / 94% / 100%. The dense model is not worth its weight, so serving needs no torch.
- **Assistant answers are extractive by default** (best matching sentences with numbered citations; no API key needed). 14 of the 18 answers contain the key phrase (78%), and all 4 off-topic questions (GST on gold, biryani, ...) are refused: a question is answered only when the retrieved passage covers at least half of its terms. With `ANTHROPIC_API_KEY` set the answer is composed by an LLM constrained to the excerpts; that path is unit-tested with a stub only, never run against the real API.
- **Decision explanations** (`src/rag/explainer.py`): a fixed template over the API response (action, calibrated probability, the top SHAP reasons, history flags) plus up to three on-topic RBI passages (a passage must contain the key term for the action, e.g. "authentication" for a challenge). Nothing is generated freely; the template states that attributions are not proof of intent.
- Dashboard: fifth tab "Analyst assistant" (question box with citations, and an explanation of the last transaction scored in the first tab). Limits: 18 questions is a small, hand-made evaluation; extractive answers can quote the right passage but not the exact sentence (the citation always shows the source); ChromaDB was not used because 445 passages do not need a vector database.

## Exploratory data analysis

`notebooks/01_eda.ipynb` (generated and executed on Kaggle by `src/models/eda_notebook.py`, so every number and plot is computed): 590,540 transactions over 182 days, 3.5% fraud; weekly fraud rate swings between 1.9% and 5.1% (time-ordered splits are required); credit cards (6.7% fraud) and mobile devices (10.2%) are riskier than debit (2.4%) and rows without identity data (2.1% vs 7.9% with it); only 24% of rows carry identity information and 214 of 438 columns are more than half missing; 98% of fraud sits on a card composite that had an earlier transaction and 81% follows an earlier fraud on the same card by at least 7 days, which is what motivated the lagged entity-history features.

**Executed notebooks** (all generated and run on Kaggle by `src/models/eda_notebook.py` and `src/models/build_notebooks.py`; aggregates only, no raw rows): `01_eda`, `02_features` (time-ordered splits, train-only encodings, and an independent recomputation of the 7-day-lag fraud history: 0 of 400 mismatches), `02b_upi_features`, `03_model_xgb` (3 seeds trained on train only, calibration and tiers from validation: tier precision lands within 0.01 to 0.025 of its target on test; train-only ensemble PR-AUC 0.667 versus 0.706 for the shipped model refit on train + validation), `04_shap` (SHAP values sum to the model output to within 9e-6) and `04b_adversarial_test` (on LightGBM: identity rotation evades 31.2% of flagged fraud with one probe, all moves 53.1% with 20; the XGBoost stress test in `results/stress_test.json` gave 31.7% and 58.4%). Notebooks `03`, `04` and `04b` were re-run on the production LightGBM model (`03` trains three XGBoost seeds on train only and compares them with the shipped LightGBM refit; in `04` SHAP values sum to the model output within 4e-10, and entity fraud history carries 31% of the attribution, `V` columns 12%). A finding from `02b` worth stating plainly: in a one-seed CPU ablation the UPI behaviour features add almost nothing (PR-AUC -0.003 when removed, within noise), while the lagged entity fraud history carries the signal (-0.14 when removed). The behaviour features are kept because they are cheap and causal, not because they were shown to help.

## Adversarial stress test

Black-box evasion test (`src/models/stress_test.py`, results in `results/stress_test.json`; the table below is the XGBoost model, and the LightGBM re-run in notebook `04b` is summarised right after it): an attacker who can probe the scorer 1, 5 or 20 times and change only fields they control (amount, payee e-mail domain, device/browser, product and card type, plus the state those drive) tries to push the 1,500 test-period frauds that the model currently flags below the alert tiers. Each move copies a legitimate donor row's values for the affected features.

| Attack (share of flagged frauds pushed below SOFT_FLAG) | 1 probe | 5 probes | 20 probes |
|---|---|---|---|
| Amount mimicry | 4.2% | 7.1% | 8.5% |
| Payee e-mail rotation | 6.1% | 13.7% | 20.7% |
| Device / browser rotation | 1.3% | 4.4% | 9.4% |
| Behaviour mimicry | 1.5% | 3.4% | 5.2% |
| Product / card swap | 3.5% | 5.1% | 5.9% |
| **Fresh identity (no entity history)** | **31.7%** | 36.4% | 40.6% |
| All moves combined | 42.9% | 52.7% | **58.4%** |

On the production LightGBM model (`04b`, same attack code, flagged = calibrated probability at or above the 0.126 SOFT_FLAG threshold, 70.0% of test fraud, 800 targets): amount mimicry 4.0/6.6/7.9%, behaviour mimicry 1.6/2.9/5.0%, payee e-mail rotation 4.9/10.1/13.9%, fresh identity 31.2/33.5/35.9%, all moves 40.9/48.8/53.1% (1/5/20 probes). The weakness is the same and slightly smaller.

XGBoost model: with all moves and 20 probes, 68.7% fall below CHALLENGE and 92.0% below HARD_BLOCK, so the model's overall recall at SOFT_FLAG could fall from about 69% to roughly 29% against a strong adaptive attacker. The dominant weakness is the entity-history dependence: a fresh identity alone evades 32% with a single probe. Probing helps the attacker (31.7% to 40.6% from 1 to 20 queries), so do not expose raw scores to clients.

**The roadmap's "two layers are necessary" claim is not supported.** An Isolation Forest second layer at 1% false-positive rate flagged 0.0% of the model-flagged frauds, 0.0% of the attacked frauds and 0.0% of those that evaded; an OR rule with the model leaves recall unchanged and raises the false-positive rate from 1.9% to 2.9%.

**Adversarial training against the fresh-identity weakness** (`src/models/adversarial_training.py`, results in `results/adv_training.json`). Pre-registered criterion: fresh-identity evasion at 1 probe below 20% with test PR-AUC loss of at most 0.01 and recall@1%FPR loss of at most 0.02. Adding rotated copies of every training fraud met it: evasion fell from 32.9% to 0.0% (all moves combined, 20 probes: 69.5% to 0.4%) while test PR-AUC moved 0.675 to 0.679, recall@1%FPR 0.617 to 0.616 and the precision of flagged alerts rose by 1.1 points. Limits: the defence is specific to the trained transformation (payee e-mail rotation still evades 26% of flagged fraud at 20 probes, baseline 30%) and recall on genuinely first-time fraud did not improve (0.570 vs 0.569); the production models in `models/compact/` do not include this training yet.

Limits of the stress test: random-search attacker in feature space (no gradient attack, no real attacker data); it cannot directly alter the opaque Vesta `V`/`C`/`D` features; the 31% of fraud the model already misses is not included in the targets.

## Limitations (read before quoting the numbers)

The numbers in this section were measured on the previous XGBoost model; the entity-history dependence and the first-time-fraud weakness are properties of the features and are expected to carry over to LightGBM, but they have not been re-measured.

- **Proxy features on non-UPI data.** IEEE-CIS is Western e-commerce. User and merchant IDs are reconstructed, not real UPI fields.
- **Base rate.** 3.5% fraud is far above real UPI. Alert rates and precision-at-alert-rate do not transfer.
- **Label feedback dependence.** The entity-history features assume fraud labels arrive within about 7 days: test PR-AUC is ~0.69 at 7 days, ~0.65 at 14, ~0.61 at 30 (baseline without history: 0.58).
- **First-time fraud is the weak spot.** 73% of test fraud is on entities with no labeled history; recall there is 0.53 (PR-AUC 0.56), vs 0.91 recall on entities with history. Repeat offenders (27% of fraud) are caught ~94%.
- **Large amounts.** Recall at 1% FPR is 0.21 for amounts >= 1000 (43 fraud rows) and 0.46 for 200-1000.
- **Short test window** (~5 weeks): weekly PR-AUC ranges 0.65-0.76.
- **Explanations.** The top SHAP drivers are human-readable entity-history features, but about 12% of SHAP mass sits on opaque Vesta `V` columns (LightGBM; 17% on the earlier XGBoost model).

## Negative results (tested and dropped)

Isolation Forest second layer and score fusion (alone PR-AUC 0.097; validation gave it zero weight), adversarial feature pruning (cost ~0.04 PR-AUC), velocity features, recency-decayed counts, extra entity keys (tied within seed noise), bagging, SMOTE; CatBoost (below XGBoost in the later study) and causal linkage features (no test gain) also did not help (replaced by `scale_pos_weight`).

## Second dataset: Sparkov (simulated), do real user ids change the picture?

On IEEE-CIS the user id is a reconstruction, and removing the UPI behaviour features cost almost nothing (notebook `02b`). To test whether that is a property of the features or of the proxy id, `src/models/sparkov_study.py` (results in `results/sparkov.json`, run on Kaggle) repeats the ablation on the Sparkov credit-card simulation (CC0, 1.85M transactions, a real card id per row, time-ordered: train, validation and the dataset's own later period as test). Criteria were fixed before the run: a set must beat the previous one by at least 0.01 PR-AUC with a paired-bootstrap 95% CI that excludes 0.

| Feature set (3 seeds, test period) | PR-AUC | ROC-AUC | Recall at 1% FPR |
|---|---|---|---|
| A: transaction + profile fields | 0.875 | 0.998 | 0.948 |
| B: A + causal per-user behaviour state | 0.951 | 0.999 | 0.983 |
| C: B + lagged (7-day) fraud history per card and merchant | 0.979 | 1.000 | 0.996 |

B over A: +0.076 (95% CI +0.067 to +0.086). C over B: +0.028 (CI +0.024 to +0.032). Both criteria are met. With a real user id the behaviour features help clearly, which supports the explanation that the IEEE-CIS result came from the proxy id and not from the features.

How much weight this carries: the data is **simulated** (999 active cards, 693 merchants, fraud injected by the generator's rules), so the absolute scores say nothing about real fraud and are not comparable to the IEEE-CIS numbers. The profile fields are constant per user, so with so few users a model can memorise them; this inflates all three sets alike and does not affect the differences. The simulator may also make deviations from a user's habits and bursts of fraud on one card easier to detect than real traffic does. The live API still runs the IEEE-CIS model; nothing here changes it.

## Model improvement study (pre-registered; LightGBM is now the production model)

Fit diagnostics (`results/diagnose.json`): the production configuration memorises its training data (train PR-AUC about 0.97 to 1.00 against about 0.72 on validation, gap 0.248, just under the 0.25 limit fixed beforehand) but validation does not degrade and shallower trees score lower, so it is neither under- nor clearly overfitted. Early stopping with patience 60 is unstable: more rounds scored higher on validation, which pointed at under-training rather than over-fitting.

`results/improve.json` (code in `src/models/common_improve.py` and `improve_*.py`, run on Kaggle): five candidates, each with four rolling-origin folds (selection only), a train-only model, and a refit on train + validation with fixed rounds, the same recipe as the shipped model. Test was read once, for the candidate the folds selected, against the shipped model.

| Candidate | Rolling-fold mean PR-AUC | Test PR-AUC (refit) | Recall at 1% FPR (refit) |
|---|---|---|---|
| Shipped model (patience 60) | 0.6957 | 0.7062 | 0.637 |
| XGBoost, patience 300 | 0.6992 | 0.7196 | 0.649 |
| XGBoost, depth 12 | 0.7128 | 0.7235 | 0.655 |
| LightGBM | 0.7202 | 0.7238 | 0.653 |
| CatBoost | 0.6725 | 0.6793 | 0.616 |
| XGBoost + 20 linkage features | 0.7121 | 0.7176 | 0.643 |
| **Blend of all five (selected by folds)** | **0.7239** | **0.7277** | **0.652** |

The pre-registered criterion (at least +0.01 PR-AUC and +0.01 recall at 1% FPR, paired-bootstrap CI excluding 0) is **met** by the selected blend: PR-AUC +0.0215 (95% CI +0.017 to +0.026), recall +0.0149 (CI +0.007 to +0.026).

Read this with its limits: the gain is modest, and most of it comes from training longer and from LightGBM; the five-model blend adds only about +0.004 over LightGBM alone (0.7238). Fold-stage early stopping is still noisy (about 80 to 5,000 rounds across folds) and the fold means differ by less than their spread, so the fold ranking is not sharp. The causal linkage features (distinct cards per device, email, address) helped on three of four folds (+0.013 mean) but not on test (-0.002), so there is no evidence they help. LightGBM first ran with an early-stopping bug (it stopped on weighted log-loss after 2 to 3 rounds); it was fixed and re-run, and the corrected run is reported. **Decision:** the API now serves LightGBM alone (3 seeds, 529 rounds, refit on train + validation, `results/prod_lgbm.json`), not the five-model blend that met the criterion. That choice was made after the study, for simplicity and size (the blend would ship depth-12 XGBoost with about 2,700 trees per seed), and is therefore post-hoc: LightGBM alone scores +0.018 PR-AUC and +0.016 recall at 1% FPR over the XGBoost model on the full test period, the same direction and size as the blend but without its own pre-registered test. Its model files are 7.4 MB each, smaller than the XGBoost files they replaced.

## Roadmap status

Done: scaffold, data, feature engineering, UPI fingerprint (causal proxy), XGBoost then LightGBM, SHAP, compact serving model, calibration + tiers, MLflow + registry, FastAPI, production simulation, drift detection, Streamlit dashboard (the threshold-tuning tab covers the calibration dashboard).
Also done: adversarial stress test, RAG ingestion and query (BM25), retrieval-grounded explanations. The LLM explanation step is optional and only template mode has run against real data (done differently).
Seen passing on GitHub (2026-10-04, run on the cleaned history): GitHub Actions `test` (all tests), `build` (Docker image) with a container `/health` smoke test. The first CI run (2026-10-03) had failed on a pandas 3 incompatibility in the feature code, which was fixed. Not done: the `deploy` job exits without deploying because no `RENDER_DEPLOY_HOOK_URL` secret or Render service exists.
Rejected on evidence: Isolation Forest, score fusion. All roadmap notebooks are done and executed (`notebooks/`).

## Kaggle setup

1. kaggle.com -> Settings -> API -> create a token; save it as `~/.kaggle/access_token` (never commit it).
2. `python tools/kaggle_tools.py version features -m "..."` uploads the changed source folder as a Kaggle dataset (keys in `kaggle/datasets.json`); wait until `kaggle datasets files srigowthamm/paywatch-src` shows the update, then `python tools/kaggle_tools.py push pipeline --wait`. `python tools/kaggle_tools.py list` shows every kernel and dataset; `fetch <kernel>` downloads only its small `results.json`.
