"""PayWatch monitoring dashboard.   Run:  streamlit run src/monitor/dashboard.py

Reads data/sim/sim_results.json (30-day production simulation, built on Kaggle) and
models/compact/serving_spec.json. Live scoring talks to the API (PAYWATCH_API_URL, default localhost:8000).
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
CHUNKS_PATH = Path(os.environ.get("PAYWATCH_RAG_CHUNKS", ROOT / "data" / "rag" / "chunks.jsonl"))
SIM_PATH = Path(os.environ.get("PAYWATCH_SIM_RESULTS", ROOT / "data" / "sim" / "sim_results.json"))
MODELS_DIR = Path(os.environ.get("PAYWATCH_MODELS_DIR", ROOT / "models" / "compact"))
DEFAULT_API = os.environ.get("PAYWATCH_API_URL", "http://localhost:8000")
TIER_NAMES = {"0.5": "SOFT_FLAG (precision target 50%)", "0.7": "CHALLENGE (target 70%)", "0.9": "HARD_BLOCK (target 90%)"}

SAMPLE = {
    "transaction_id": "demo-1",
    "fields": {"TransactionDT": 15000000, "TransactionAmt": 120.0, "ProductCD": "W", "card1": 13926, "card2": 111,
               "card3": 150, "card5": 226, "addr1": 315, "D1": 20, "P_emaildomain": "gmail.com"},
    "explain": True, "commit": False,
}


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


def vline(x, color="#999"):
    return alt.Chart(pd.DataFrame({"x": [x]})).mark_rule(color=color, strokeDash=[4, 4]).encode(x="x:Q")


def call_api(url, payload_text):
    req = urllib.request.Request(url.rstrip("/") + "/predict", data=payload_text.encode("utf-8"),
                                 headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=10) as r:
        return json.loads(r.read())


st.set_page_config(page_title="PayWatch", layout="wide")
st.title("PayWatch - fraud scoring and monitoring")

sim = load_json(str(SIM_PATH))
spec = load_json(str(MODELS_DIR / "serving_spec.json"))
if sim is None:
    st.warning(f"No simulation results at {SIM_PATH}. Build them with `python tools/kaggle_tools.py push sim --wait`, then `fetch sim --pattern sim_results.json --out data/sim/sim_results.json`.")
    st.stop()

curve = sim["calibration"]["curve_second_half"]
tab_live, tab_drift, tab_perf, tab_thr, tab_rag = st.tabs(
    ["Live scoring", "Drift monitoring", "Performance & calibration", "Threshold tuning", "Analyst assistant"])

# ------------------------------------------------------------------ 1) live scoring + tier policy
with tab_live:
    left, right = st.columns([1, 1])
    with left:
        api_url = st.text_input("API URL", DEFAULT_API)
        payload_text = st.text_area("Transaction (JSON, IEEE-CIS field names)", json.dumps(SAMPLE, indent=2), height=300)
        score_clicked = st.button("Score transaction")
    with right:
        st.subheader("Tier policy")
        tiers = (spec or {}).get("tiers", {})
        rows = []
        for key, name in TIER_NAMES.items():
            t = tiers.get(key)
            if not t:
                continue
            p = nearest_point(curve["points"], t["calibrated_threshold"])
            rows.append({"tier": name, "calibrated threshold": round(t["calibrated_threshold"], 3),
                         "precision (replay)": round(p["precision"], 3), "recall (replay)": round(p["recall"], 3),
                         "alert rate": f"{p['alert_rate']:.1%}"})
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
        st.caption("Replay numbers are from the 30-day control simulation (second half, calibrator out-of-sample).")
    if score_clicked:
        try:
            out = call_api(api_url, payload_text)
        except Exception as e:  # network / validation errors are shown, not raised
            st.error(f"API call failed: {e}. Start it with `uvicorn --factory src.api.main:create_app`.")
        else:
            st.session_state["last_prediction"] = out
            c1, c2, c3 = st.columns(3)
            c1.metric("Raw score", f"{out['score']:.4f}")
            c2.metric("Calibrated probability", f"{out['calibrated_probability']:.3f}")
            c3.metric("Action", out["action"])
            st.dataframe(pd.DataFrame(out["reasons"])[["text", "direction", "shap", "family"]], hide_index=True,
                         use_container_width=True)
            st.json(out["flags"])

# ------------------------------------------------------------------ 2) drift timeline
with tab_drift:
    cfg = sim["config"]
    c1, c2 = st.columns(2)
    scenario = c1.radio("Scenario", ["drift", "control"], horizontal=True,
                        help="drift = covariate drift injected on day %s; control = no drift (false-alarm check)" % cfg["drift_day"])
    window = c2.selectbox("Monitoring window (days)", cfg["windows_tested"], index=cfg["windows_tested"].index(sim["chosen_window_days"]))
    r = sim[scenario][f"window_{window}"]
    daily = pd.DataFrame(r["daily"])
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
    drift_layer = base.mark_bar().encode(y=alt.Y("n_joint_significant:Q", title="features with PSI>0.2 and KS>0.2"))
    left.altair_chart((drift_layer + vline(cfg["drift_day"])) if scenario == "drift" else drift_layer, use_container_width=True)
    score_layer = base.mark_line(point=True).encode(y=alt.Y("score_psi:Q", title="score PSI"))
    right.altair_chart((score_layer + vline(cfg["drift_day"])) if scenario == "drift" else score_layer, use_container_width=True)
    st.subheader("Alerts")
    st.dataframe(pd.DataFrame(r["alerts"])[["day", "severity", "signal", "message"]] if r["alerts"] else
                 pd.DataFrame({"result": ["no alerts"]}), hide_index=True, use_container_width=True)
    if r.get("final_window_distributions"):
        st.subheader("Most-drifted inputs (final window vs reference)")
        dist = {d["feature"]: d for d in r["final_window_distributions"]}
        feat = st.selectbox("Feature", list(dist))
        d = dist[feat]
        long = pd.DataFrame({"bin": d["bins"] * 2, "share": d["reference"] + d["current"],
                             "period": ["reference"] * len(d["bins"]) + ["current"] * len(d["bins"])})
        st.altair_chart(alt.Chart(long).mark_bar().encode(x=alt.X("bin:N", sort=None), y="share:Q", color="period:N",
                                                         xOffset="period:N"), use_container_width=True)

# ------------------------------------------------------------------ 3) performance and calibration
with tab_perf:
    r = sim["drift"][f"window_{sim['chosen_window_days']}"]
    daily = pd.DataFrame(r["daily"])
    base_pr = r["baseline"]["pr_auc"]
    perf = daily.dropna(subset=["pr_auc_mature_window"])
    st.subheader("Label-delayed PR-AUC (rows whose labels have arrived)")
    line = alt.Chart(perf).mark_line(point=True).encode(x="day:Q", y=alt.Y("pr_auc_mature_window:Q", scale=alt.Scale(zero=False)))
    st.altair_chart(line + vline(sim["config"]["drift_day"]) + alt.Chart(pd.DataFrame({"y": [base_pr]})).mark_rule(color="green").encode(y="y:Q"),
                    use_container_width=True)
    st.caption(f"Green = baseline PR-AUC {base_pr:.3f} (first {r['baseline']['ref_days']} days). "
               "Labels arrive 7 days late, so performance alerts always lag input-drift alerts.")
    st.subheader("Reliability (second half of the stream, calibrator out-of-sample)")
    rel = pd.DataFrame(sim["calibration"]["reliability_second_half"])
    diag = alt.Chart(pd.DataFrame({"x": [0, 1], "y": [0, 1]})).mark_line(color="#999").encode(x="x:Q", y="y:Q")
    pts = alt.Chart(rel).mark_circle(size=90).encode(x=alt.X("mean_predicted:Q", scale=alt.Scale(type="sqrt")),
                                                     y=alt.Y("observed_fraud_rate:Q", scale=alt.Scale(type="sqrt")),
                                                     tooltip=["n", "mean_predicted", "observed_fraud_rate"])
    st.altair_chart(diag + pts, use_container_width=True)

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
    st.altair_chart(alt.Chart(pts).mark_line().encode(x="threshold:Q", y=alt.Y("value:Q", title=None), color="metric:N")
                    + vline(thr, "#d62728"), use_container_width=True)
    st.subheader("Cost trade-off")
    cc1, cc2 = st.columns(2)
    cost_missed = cc1.number_input("Cost of a missed fraud", 1.0, 1_000_000.0, 1000.0, 50.0)
    cost_fp = cc2.number_input("Cost of a false alert", 0.0, 100_000.0, 10.0, 1.0)
    ct = cost_table(curve["points"], curve["fraud_rate"], cost_missed, cost_fp)
    best = ct.loc[ct["expected_cost_per_1000"].idxmin()]
    chosen = ct.loc[(ct["threshold"] - thr).abs().idxmin()]
    st.altair_chart(alt.Chart(ct).mark_line().encode(x="threshold:Q", y=alt.Y("expected_cost_per_1000:Q", title="expected cost per 1,000 transactions"))
                    + vline(thr, "#d62728") + vline(float(best["threshold"]), "#2ca02c"), use_container_width=True)
    st.write(f"At {thr:.2f}: cost {chosen['expected_cost_per_1000']:,.0f} per 1,000 "
             f"({chosen['missed_fraud_per_1000']:.1f} missed fraud, {chosen['false_alerts_per_1000']:.1f} false alerts). "
             f"Cheapest threshold: **{best['threshold']:.2f}** (cost {best['expected_cost_per_1000']:,.0f}).")
    st.caption(f"Base rate in this dataset is {curve['fraud_rate']:.1%}, far above real UPI; re-derive thresholds on real traffic.")

# ------------------------------------------------------------------ 5) analyst assistant (RAG over RBI documents)
with tab_rag:
    from src.rag.explainer import explain_decision
    from src.rag.qa_chain import anthropic_llm, answer

    retriever = load_retriever(str(CHUNKS_PATH))
    if retriever is None:
        st.info(f"No knowledge base at {CHUNKS_PATH}. Build it with `python tools/kaggle_tools.py push rag --wait`, then `fetch rag --pattern chunks.jsonl --out data/rag/chunks.jsonl` "
                "(public RBI circulars; the index is not committed).")
    else:
        llm = anthropic_llm()
        sources = sorted({c["title"] for c in retriever.chunks})
        st.caption(f"{len(retriever.chunks)} passages from {len(sources)} RBI documents. Mode: "
                   + ("LLM (ANTHROPIC_API_KEY set), constrained to the retrieved excerpts" if llm else
                      "extractive (no API key): returns the best matching sentences with citations and refuses when the "
                      "documents do not cover the question"))
        question = st.text_input("Ask about RBI fraud, liability or AI rules",
                                 "Within how many working days must a customer report an unauthorised transaction?")
        if question.strip():
            res = answer(question, retriever, llm=llm)
            st.write(res["answer"])
            for c in res["citations"]:
                with st.expander(f"[{c['n']}] {c['title']} (page {c['page']})"):
                    st.write(c["excerpt"])
                    st.markdown(f"[source]({c['url']})")
        st.subheader("Explain the last scored transaction")
        last = st.session_state.get("last_prediction")
        if last is None:
            st.write("Score a transaction in the first tab, then come back here.")
        else:
            ex = explain_decision(last, retriever, llm=llm)
            st.text(ex["explanation"])
            if ex["regulatory_context"]:
                st.write("Relevant RBI passages:")
                for c in ex["regulatory_context"]:
                    st.markdown(f"- **{c['title']}**, p.{c['page']}: {c['excerpt'][:200]}... [source]({c['url']})")
