"""PayWatch dashboard.   Run:  streamlit run src/monitor/dashboard.py

Reads data/sim/sim_results.json (30-day production simulation, built on Kaggle), models/compact/serving_spec.json and
results/prod_lgbm.json. Live scoring talks to the API (PAYWATCH_API_URL, default the public demo API).
"""

import json
import os
import sys
import urllib.request
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from src.monitor import ui  # noqa: E402

CHUNKS_PATH = Path(os.environ.get("PAYWATCH_RAG_CHUNKS", ROOT / "data" / "rag" / "chunks.jsonl"))
SIM_PATH = Path(os.environ.get("PAYWATCH_SIM_RESULTS", ROOT / "data" / "sim" / "sim_results.json"))
MODELS_DIR = Path(os.environ.get("PAYWATCH_MODELS_DIR", ROOT / "models" / "compact"))
RESULTS_PATH = Path(os.environ.get("PAYWATCH_RESULTS", ROOT / "results" / "prod_lgbm.json"))
DEFAULT_API = os.environ.get("PAYWATCH_API_URL", "https://paywatch-api-fpq9.onrender.com")   # public demo API; set the env var for a local one
TIER_ORDER = (("0.5", "SOFT_FLAG"), ("0.7", "CHALLENGE"), ("0.9", "HARD_BLOCK"))
SAMPLE = json.loads((ROOT / "src" / "api" / "example_request.json").read_text(encoding="utf-8"))   # full synthetic transaction, same as the API /example
SUGGESTIONS = ["Within how many working days must a customer report an unauthorised transaction?",
               "What is the limiting liability of a customer in unauthorised electronic banking transactions?",
               "What does the RBI say about explainability of AI models?"]


@st.cache_data
def load_json(path: str):
    p = Path(path)
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


@st.cache_resource
def load_retriever(path: str):
    from src.rag.ingest import load_chunks
    from src.rag.retriever import BM25Retriever

    p = Path(path)
    return BM25Retriever(load_chunks(str(p))) if p.exists() else None


def nearest_point(points, threshold):
    return min(points, key=lambda p: abs(p["threshold"] - threshold))


def cost_table(points, fraud_rate, cost_missed, cost_false_alert, n=1000):
    df = pd.DataFrame(points)
    fraud = fraud_rate * n
    tp = df["recall"] * fraud
    df["missed_fraud_per_1000"] = fraud - tp
    df["false_alerts_per_1000"] = df["alert_rate"] * n - tp
    df["expected_cost_per_1000"] = cost_missed * df["missed_fraud_per_1000"] + cost_false_alert * df["false_alerts_per_1000"]
    return df


def vline(x, color, label=None, dash=(5, 4)):
    rule = alt.Chart(pd.DataFrame({"x": [x], "label": [label or ""]})).mark_rule(color=color, strokeDash=list(dash), strokeWidth=1.5).encode(x="x:Q")
    if not label:
        return rule
    return rule + alt.Chart(pd.DataFrame({"x": [x], "label": [label]})).mark_text(align="left", dx=5, dy=-4, color=color, fontSize=11,
                                                                                    baseline="top").encode(x="x:Q", text="label:N",
                                                                                                           y=alt.value(2))


def call_api(url, payload_text):
    req = urllib.request.Request(url.rstrip("/") + "/predict", data=payload_text.encode("utf-8"),
                                 headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=90) as r:   # the free-plan demo API needs up to a minute to wake after idle
        return json.loads(r.read())


st.set_page_config(page_title="PayWatch", page_icon="🛡️", layout="wide")
ui.inject()
C = ui.colors()

sim = load_json(str(SIM_PATH))
spec = load_json(str(MODELS_DIR / "serving_spec.json"))
prod = load_json(str(RESULTS_PATH))
if sim is None:
    st.warning(f"No simulation results at {SIM_PATH}. Build them with `python tools/kaggle_tools.py push sim --wait`, then `fetch sim --pattern sim_results.json --out data/sim/sim_results.json`.")
    st.stop()

tiers = (spec or {}).get("tiers", {})
thresholds = {k: t["calibrated_threshold"] for k, t in tiers.items() if t}
curve = sim["calibration"]["curve_second_half"]
family = (spec or {}).get("model_family", "model")
ui.hero("PayWatch", "Fraud scoring for UPI-style transactions: a calibrated probability, an action tier and the reasons behind it, "
        "with drift monitoring and an assistant grounded in RBI documents.",
        [f"{family} x {len((spec or {}).get('prod_seeds', [])) or 3} seeds", f"{len((spec or {}).get('features', []))} features",
         "trained on IEEE-CIS (not UPI data)", "time-ordered evaluation"])

tab_over, tab_live, tab_drift, tab_perf, tab_thr, tab_rag = st.tabs(
    ["Overview", "Live scoring", "Drift monitoring", "Performance & calibration", "Threshold tuning", "Analyst assistant"])

# ------------------------------------------------------------------ 0) overview
with tab_over:
    full = (prod or {}).get("test_full")
    k1, k2, k3, k4 = st.columns(4)
    if full:
        k1.markdown(ui.card("PR-AUC (test)", f"{full['pr_auc']:.3f}", "a random scorer gets 0.035"), unsafe_allow_html=True)
        k2.markdown(ui.card("ROC-AUC (test)", f"{full['roc_auc']:.3f}", "ranking quality, 1.0 is perfect"), unsafe_allow_html=True)
        k3.markdown(ui.card("Caught at 1% false alarms", f"{full['recall_at_fpr_1pct']:.0%}", "of frauds in the held-out period"), unsafe_allow_html=True)
    hb = tiers.get("0.9")
    if hb:
        k4.markdown(ui.card("Hard-block precision", f"{hb['second_half_precision']:.0%}", f"at {hb['second_half_recall']:.0%} recall"), unsafe_allow_html=True)
    st.markdown("#### Action tiers")
    st.caption("Thresholds on the calibrated probability, fitted on the first half of the test period and measured on the second half.")
    cols = st.columns(3)
    for col, (key, action) in zip(cols, TIER_ORDER):
        t = tiers.get(key)
        if t:
            col.markdown(ui.tier_card(action, t["calibrated_threshold"], t["second_half_precision"], t["second_half_recall"],
                                      t["second_half_alert_rate"]), unsafe_allow_html=True)
    st.markdown("#### How a transaction is scored")
    st.markdown('<div class="pw-steps">'
                '<div class="pw-step"><b>1 · Transaction arrives</b>raw fields only, no labels needed to score</div>'
                '<div class="pw-step"><b>2 · History features</b>per-user behaviour and entity fraud history from labels at least 7 days old</div>'
                '<div class="pw-step"><b>3 · Model + calibration</b>three LightGBM seeds, scores mapped to real probabilities</div>'
                '<div class="pw-step"><b>4 · Action + reasons</b>allow, soft flag, challenge or hard block, with the top SHAP reasons</div>'
                '</div>', unsafe_allow_html=True)
    with st.expander("Read before quoting these numbers"):
        st.markdown(
            "- The data is a public Western e-commerce fraud dataset (IEEE-CIS), **not UPI traffic**; user ids are reconstructed.\n"
            "- Recall on **first-time fraud is about 57%**; the model relies on fraud labels arriving within about 7 days.\n"
            "- A **fresh identity** evades about 31% of flagged fraud with a single probe (stress test, notebook 04b).\n"
            "- 0.99 precision overall is not reachable; the most confident tier measured 97% precision at about 19% recall.\n"
            "- Drift in the Drift tab is **injected** into a replay of test data, not observed in production.")

# ------------------------------------------------------------------ 1) live scoring
with tab_live:
    left, right = st.columns([5, 6], gap="large")
    with left:
        st.markdown("#### Score a transaction")
        st.caption("A complete synthetic transaction is pre-filled. The demo API runs on a free plan and can take about a minute to wake.")
        with st.expander("API address", expanded=False):
            api_url = st.text_input("API URL", DEFAULT_API)
        payload_text = st.text_area("Transaction (JSON, IEEE-CIS field names)", json.dumps(SAMPLE, indent=2), height=260)
        score_clicked = st.button("Score transaction", type="primary", width="stretch")
    if score_clicked:
        try:
            with right, st.spinner("Scoring (the first call may wake the API)..."):
                st.session_state["last_prediction"] = call_api(api_url, payload_text)
        except Exception as e:  # network / validation errors are shown, not raised
            right.error(f"API call failed: {e}. Start a local API with `uvicorn --factory src.api.main:create_app`.")
    out = st.session_state.get("last_prediction")
    with right:
        st.markdown("#### Result")
        if out is None:
            ui.banner("info", "Press <b>Score transaction</b> to see the decision, the risk gauge and the reasons.")
        else:
            st.markdown(ui.chip(out["action"]), unsafe_allow_html=True)
            st.markdown(ui.gauge(out["calibrated_probability"], thresholds), unsafe_allow_html=True)
            st.caption("Calibrated fraud probability on a square-root scale; coloured bands are the action tiers.")
            c1, c2, c3 = st.columns(3)
            c1.metric("Raw score", f"{out['score']:.4f}")
            c2.metric("Calibrated probability", f"{out['calibrated_probability']:.3f}")
            c3.metric("Action", out["action"])
            if out["reasons"]:
                st.markdown("**Why this score** (effect on the fraud score, log-odds)")
                st.markdown(ui.reasons_list(out["reasons"]), unsafe_allow_html=True)
            flags = {k: v for k, v in out["flags"].items() if v}
            st.caption("Flags: " + (", ".join(k.replace("_", " ") for k in flags) if flags else "none") + f" · latency {out['latency_ms']:.0f} ms · {out['model_version']}")
            with st.expander("Raw API response"):
                st.json(out)

# ------------------------------------------------------------------ 2) drift timeline
with tab_drift:
    cfg = sim["config"]
    c1, c2 = st.columns(2)
    scenario = c1.radio("Scenario", ["drift", "control"], horizontal=True,
                        help="drift = covariate drift injected on day %s; control = no drift (false-alarm check)" % cfg["drift_day"])
    window = c2.selectbox("Monitoring window (days)", cfg["windows_tested"], index=cfg["windows_tested"].index(sim["chosen_window_days"]))
    r = sim[scenario][f"window_{window}"]
    daily = pd.DataFrame(r["daily"])
    fa = r.get("first_alert_day", {})
    warn_d, crit_d = fa.get("feature_drift:WARN"), fa.get("feature_drift:CRITICAL")
    if scenario == "drift" and warn_d is not None:
        ui.banner("warn", f"Input drift was injected on day {int(cfg['drift_day'])}. The monitor flagged it as <b>WARN on day {warn_d:.0f}</b>"
                          + (f" and <b>CRITICAL on day {crit_d:.0f}</b>" if crit_d is not None else "") + f" using a {window}-day window.")
    elif scenario == "control" and not r["alerts"]:
        ui.banner("good", f"No alerts in 30 days of normal traffic at the {window}-day window: no false alarms.")
    elif r["alerts"]:
        ui.banner("warn", f"{len(r['alerts'])} alert(s) in this scenario at the {window}-day window.")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("WARN alerts", r["alert_counts"]["WARN"])
    m2.metric("CRITICAL alerts", r["alert_counts"]["CRITICAL"])
    m3.metric("Alerts on/before drift day", r["alerts_before_drift_day"] if r["alerts_before_drift_day"] is not None else "n/a")
    lag = r.get("detection_lag_days", {}).get("feature_drift:WARN") or r.get("detection_lag_days", {}).get("feature_drift:CRITICAL")
    m4.metric("Input-drift detection lag (days)", f"{lag:+.0f}" if lag is not None else "-")
    st.caption(f"{cfg['monitored_features']} of {cfg['total_features']} features monitored (client inputs + fixed lookups); "
               "state that grows by design is excluded. Dashed line = injected drift day.")

    base = alt.Chart(daily).encode(x=alt.X("day:Q", title="day (window end)"))
    left, right = st.columns(2)
    drift_layer = base.mark_bar(color=C["blue"], cornerRadiusEnd=3, size=9).encode(
        y=alt.Y("n_joint_significant:Q", title="features with PSI>0.2 and KS>0.2"), tooltip=["day", "n_joint_significant"])
    left.altair_chart((drift_layer + vline(cfg["drift_day"], C["orange"], "drift injected")) if scenario == "drift" else drift_layer, width="stretch")
    score_layer = base.mark_line(color=C["blue"], strokeWidth=2, point=alt.OverlayMarkDef(size=36, filled=True)).encode(
        y=alt.Y("score_psi:Q", title="score PSI"), tooltip=["day", alt.Tooltip("score_psi:Q", format=".3f")])
    right.altair_chart((score_layer + vline(cfg["drift_day"], C["orange"])) if scenario == "drift" else score_layer, width="stretch")
    st.markdown("#### Alerts")
    st.dataframe(pd.DataFrame(r["alerts"])[["day", "severity", "signal", "message"]] if r["alerts"] else
                 pd.DataFrame({"result": ["no alerts"]}), hide_index=True, width="stretch")
    if r.get("final_window_distributions"):
        st.markdown("#### Most-drifted inputs (final window vs reference)")
        dist = {d["feature"]: d for d in r["final_window_distributions"]}
        feat = st.selectbox("Feature", list(dist))
        d = dist[feat]
        long = pd.DataFrame({"bin": d["bins"] * 2, "share": d["reference"] + d["current"],
                             "period": ["reference"] * len(d["bins"]) + ["current"] * len(d["bins"])})
        st.altair_chart(alt.Chart(long).mark_bar(cornerRadiusEnd=3).encode(
            x=alt.X("bin:N", sort=None, title=None), y=alt.Y("share:Q", title="share of transactions"),
            color=alt.Color("period:N", scale=alt.Scale(domain=["reference", "current"], range=[C["blue"], C["orange"]]), legend=alt.Legend(orient="top", title=None)),
            xOffset="period:N", tooltip=["bin", "period", alt.Tooltip("share:Q", format=".3f")]), width="stretch")

# ------------------------------------------------------------------ 3) performance and calibration
with tab_perf:
    r = sim["drift"][f"window_{sim['chosen_window_days']}"]
    daily = pd.DataFrame(r["daily"])
    base_pr = r["baseline"]["pr_auc"]
    perf = daily.dropna(subset=["pr_auc_mature_window"])
    st.markdown("#### Label-delayed PR-AUC (rows whose labels have arrived)")
    line = alt.Chart(perf).mark_line(color=C["blue"], strokeWidth=2, point=alt.OverlayMarkDef(size=36, filled=True)).encode(
        x=alt.X("day:Q", title="day"), y=alt.Y("pr_auc_mature_window:Q", title="PR-AUC", scale=alt.Scale(zero=False)),
        tooltip=["day", alt.Tooltip("pr_auc_mature_window:Q", format=".3f")])
    base_rule = alt.Chart(pd.DataFrame({"y": [base_pr]})).mark_rule(color=C["muted"], strokeDash=[6, 4], strokeWidth=1.5).encode(y="y:Q")
    st.altair_chart(line + base_rule + vline(sim["config"]["drift_day"], C["orange"], "drift injected"), width="stretch")
    st.caption(f"Grey dashed = baseline PR-AUC {base_pr:.3f} (first {r['baseline']['ref_days']} days). "
               "Labels arrive 7 days late, so performance alerts always lag input-drift alerts.")
    st.markdown("#### Reliability (second half of the stream, calibrator out-of-sample)")
    rel = pd.DataFrame(sim["calibration"]["reliability_second_half"])
    diag = alt.Chart(pd.DataFrame({"x": [0, 1], "y": [0, 1]})).mark_line(color=C["muted"], strokeDash=[5, 4]).encode(x="x:Q", y="y:Q")
    pts = alt.Chart(rel).mark_circle(size=110, color=C["blue"], opacity=0.9, stroke="white", strokeWidth=1.5).encode(
        x=alt.X("mean_predicted:Q", title="predicted probability", scale=alt.Scale(type="sqrt")),
        y=alt.Y("observed_fraud_rate:Q", title="observed fraud rate", scale=alt.Scale(type="sqrt")),
        tooltip=["n", alt.Tooltip("mean_predicted:Q", format=".3f"), alt.Tooltip("observed_fraud_rate:Q", format=".3f")])
    st.altair_chart(diag + pts, width="stretch")
    st.caption("Points on the dashed diagonal mean a stated probability matches how often fraud really occurs.")

# ------------------------------------------------------------------ 4) threshold slider
with tab_thr:
    st.write("Pick the calibrated-probability cut-off and see the business trade-off on the replayed stream.")
    thr = st.slider("Calibrated threshold", 0.02, 0.94, 0.20, 0.02)
    p = nearest_point(curve["points"], thr)
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Precision", f"{p['precision']:.1%}")
    k2.metric("Recall", f"{p['recall']:.1%}")
    k3.metric("Alert rate", f"{p['alert_rate']:.1%}")
    k4.metric("Amount-weighted recall", f"{p['amount_weighted_recall']:.1%}")
    pts = pd.DataFrame(curve["points"]).melt("threshold", ["precision", "recall", "alert_rate"], var_name="metric")
    scale = alt.Scale(domain=["precision", "recall", "alert_rate"], range=[C["blue"], C["orange"], C["aqua"]])
    lines = alt.Chart(pts).mark_line(strokeWidth=2).encode(
        x=alt.X("threshold:Q", title="calibrated threshold"), y=alt.Y("value:Q", title=None, axis=alt.Axis(format="%")),
        color=alt.Color("metric:N", scale=scale, legend=alt.Legend(orient="top", title=None)))
    last = pts.loc[pts.groupby("metric")["threshold"].idxmax()]
    ends = alt.Chart(last).mark_text(align="left", dx=6, fontSize=11).encode(
        x="threshold:Q", y="value:Q", text="metric:N", color=alt.Color("metric:N", scale=scale, legend=None))
    st.altair_chart(lines + ends + vline(thr, "#d03b3b", f"{thr:.2f}"), width="stretch")
    st.markdown("#### Cost trade-off")
    cc1, cc2 = st.columns(2)
    cost_missed = cc1.number_input("Cost of a missed fraud", 1.0, 1_000_000.0, 1000.0, 50.0)
    cost_fp = cc2.number_input("Cost of a false alert", 0.0, 100_000.0, 10.0, 1.0)
    ct = cost_table(curve["points"], curve["fraud_rate"], cost_missed, cost_fp)
    best = ct.loc[ct["expected_cost_per_1000"].idxmin()]
    chosen = ct.loc[(ct["threshold"] - thr).abs().idxmin()]
    cost_line = alt.Chart(ct).mark_line(color=C["blue"], strokeWidth=2).encode(
        x=alt.X("threshold:Q", title="calibrated threshold"), y=alt.Y("expected_cost_per_1000:Q", title="expected cost per 1,000 transactions"),
        tooltip=["threshold", alt.Tooltip("expected_cost_per_1000:Q", format=",.0f")])
    st.altair_chart(cost_line + vline(thr, "#d03b3b", "your threshold") + vline(float(best["threshold"]), C["aqua"], "cheapest", dash=(2, 3)), width="stretch")
    st.write(f"At {thr:.2f}: cost {chosen['expected_cost_per_1000']:,.0f} per 1,000 "
             f"({chosen['missed_fraud_per_1000']:.1f} missed fraud, {chosen['false_alerts_per_1000']:.1f} false alerts). "
             f"Cheapest threshold: **{best['threshold']:.2f}** (cost {best['expected_cost_per_1000']:,.0f}).")
    st.caption(f"Base rate in this dataset is {curve['fraud_rate']:.1%}, far above real UPI; re-derive thresholds on real traffic.")

# ------------------------------------------------------------------ 5) analyst assistant (RAG over RBI documents)
with tab_rag:
    from src.rag.explainer import explain_decision
    import src.rag.qa_chain as qa_chain   # module access, so a stale module after a hot redeploy degrades instead of crashing
    answer = qa_chain.answer
    default_llm = getattr(qa_chain, "default_llm", lambda: None)
    provider_name = getattr(qa_chain, "provider_name", lambda: "LLM")

    retriever = load_retriever(str(CHUNKS_PATH))
    if retriever is None:
        st.info(f"No knowledge base at {CHUNKS_PATH}. Build it with `python tools/kaggle_tools.py push rag --wait`, then `fetch rag --pattern chunks.jsonl --out data/rag/chunks.jsonl`.")
    else:
        llm = default_llm()
        sources = sorted({c["title"] for c in retriever.chunks})
        st.caption(f"{len(retriever.chunks)} passages from {len(sources)} RBI documents. Mode: "
                   + (f"LLM ({provider_name()} key set), constrained to the retrieved excerpts; falls back to extractive on errors" if llm else
                      "extractive (no API key): returns the best matching sentences with citations and refuses when the "
                      "documents do not cover the question"))
        st.session_state.setdefault("rag_q", SUGGESTIONS[0])
        st.markdown("**Try a question**")
        for sc, sug in zip(st.columns(len(SUGGESTIONS)), SUGGESTIONS):
            sc.button(sug, key=f"sug_{sug[:20]}", on_click=lambda s=sug: st.session_state.update(rag_q=s), width="stretch")
        question = st.text_input("Ask about RBI fraud, liability or AI rules", key="rag_q")
        if question.strip():
            res = answer(question, retriever, llm=llm)
            if "not covered" in res["answer"].lower() and not res["citations"]:
                ui.banner("info", res["answer"])
            else:
                st.markdown(res["answer"])
            for c in res["citations"]:
                with st.expander(f"[{c['n']}] {c['title']} (page {c['page']})"):
                    st.write(c["excerpt"])
                    st.markdown(f"[source]({c['url']})")
        st.markdown("#### Explain the last scored transaction")
        last = st.session_state.get("last_prediction")
        if last is None:
            st.write("Score a transaction in the Live scoring tab, then come back here.")
        else:
            ex = explain_decision(last, retriever, llm=llm)
            st.text(ex["explanation"])
            if ex["regulatory_context"]:
                st.write("Relevant RBI passages:")
                for c in ex["regulatory_context"]:
                    st.markdown(f"- **{c['title']}**, p.{c['page']}: {c['excerpt'][:200]}... [source]({c['url']})")

st.markdown("---")
st.caption("PayWatch is a research demo trained on a public e-commerce fraud dataset; it is not for production decisions. "
           "[Source on GitHub](https://github.com/Sri-Gowtham/paywatch) · [Live API docs](https://paywatch-api-fpq9.onrender.com/docs)")
