"""PayWatch — build and evaluate the RAG knowledge base (Kaggle, CPU, internet ON).

Fetches public RBI documents, chunks them, builds the BM25 index (chunks.jsonl), and measures retrieval quality
on a labeled question set: a hit means a top-k chunk from an expected source contains an accepted key phrase.
Compares BM25 vs dense (all-MiniLM-L6-v2) vs hybrid (reciprocal-rank fusion).
"""

import glob
import json
import os
import subprocess
import sys
import time

subprocess.run([sys.executable, "-m", "pip", "install", "-q", "pypdf", "beautifulsoup4", "requests", "sentence-transformers"],
               check=False)

T0 = time.time()
find = lambda p: sorted(glob.glob(p, recursive=True))  # noqa: E731
OUT = "/kaggle/working"


def log(msg):
    print(f"[{(time.time() - T0) / 60:5.1f} min] {msg}", flush=True)


rag_dir = os.path.dirname(find("/kaggle/input/**/qa_chain.py")[0])
os.makedirs("/tmp/repo/src", exist_ok=True)
os.system(f"cp -r {rag_dir} /tmp/repo/src/rag")
open("/tmp/repo/src/__init__.py", "w").close()
sys.path.insert(0, "/tmp/repo")

from src.rag.explainer import explain_decision  # noqa: E402
from src.rag.ingest import build, save_chunks  # noqa: E402
from src.rag.qa_chain import answer  # noqa: E402
from src.rag.retriever import BM25Retriever, HybridRetriever  # noqa: E402
from src.rag.sources import SOURCES  # noqa: E402

chunks, report = build(SOURCES)
for r in report:
    log(f"source {r['id']}: " + (f"ok pages={r['pages']} chunks={r['chunks']}" if r["ok"] else f"FAILED {r['error']}"))
save_chunks(chunks, f"{OUT}/chunks.jsonl")
log(f"total chunks={len(chunks)}")

QUESTIONS = [
    ("Within how many working days must a customer report an unauthorised transaction to have zero liability?",
     ["limiting_liability_2017"], ["three working days", "3 working days"]),
    ("Within how many working days must the bank credit the disputed amount back to the customer?",
     ["limiting_liability_2017"], ["10 working days", "ten working days"]),
    ("Who has the burden of proving customer liability in unauthorised electronic banking transactions?",
     ["limiting_liability_2017"], ["burden of proving", "burden of proof"]),
    ("What is the customer's liability when the loss is due to the customer's own negligence?",
     ["limiting_liability_2017"], ["negligence"]),
    ("What does the RBI AI framework say about explainability of AI models?", ["free_ai_2025"], ["explainab"]),
    ("Does the RBI AI framework recommend a board approved AI policy?", ["free_ai_2025"], ["board-approved", "board approved"]),
    ("What are the sutras or guiding principles for responsible AI in the financial sector?", ["free_ai_2025"], ["sutra"]),
    ("How should regulated entities handle bias and fairness in AI models?", ["free_ai_2025"], ["bias", "fairness"]),
    ("What are the requirements for AI incident reporting?", ["free_ai_2025"], ["incident"]),
    ("What authentication is required for digital payment transactions?", ["digital_payment_security_2021"], ["authentication"]),
    ("What fraud risk management controls must regulated entities have for digital payments?",
     ["digital_payment_security_2021"], ["fraud risk"]),
    ("What are the security controls for mobile payment applications?", ["digital_payment_security_2021"], ["mobile"]),
    ("What is an early warning signal framework for detecting fraud?",
     ["fraud_risk_mgmt_md_2024", "fraud_risk_mgmt_faq_2025"], ["early warning signal"]),
    ("What natural justice principle must banks follow before classifying an account as fraud?",
     ["fraud_risk_mgmt_md_2024", "fraud_risk_mgmt_faq_2025"], ["natural justice", "show cause"]),
    ("To whom must banks report incidents of fraud?",
     ["fraud_risk_mgmt_md_2024", "fraud_risk_mgmt_faq_2025"], ["law enforcement"]),
    ("Which committee of the board oversees fraud risk management?",
     ["fraud_risk_mgmt_md_2024", "fraud_risk_mgmt_faq_2025"], ["committee of the board", "special committee"]),
    ("What does Payments Vision 2025 say about fraud mitigation in digital payments?", ["payments_vision_2025"], ["fraud"]),
    ("How can digital payment fraud be reduced using technology?", ["payments_vision_2025", "digital_payment_security_2021"], ["fraud"]),
]
usable = [q for q in QUESTIONS if any(c["source_id"] in q[1] for c in chunks)]
log(f"evaluation questions usable (their source was fetched): {len(usable)} of {len(QUESTIONS)}")

if not chunks:                                   # every fetch failed: keep the failure report instead of crashing
    json.dump({"sources": report, "n_chunks": 0, "error": "no source could be fetched"},
              open(f"{OUT}/results.json", "w"), indent=1)
    raise SystemExit("no source could be fetched; see results.json for the per-source report")
bm25 = BM25Retriever(chunks)
retrievers = {"bm25": bm25}
try:
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cpu")
    vecs = model.encode([c["text"] for c in chunks], batch_size=64, normalize_embeddings=True, show_progress_bar=False)
    embed = lambda q: model.encode([q], normalize_embeddings=True)[0]  # noqa: E731

    class Dense:
        def search(self, q, k=5):
            import numpy as np

            top = np.argsort(-(vecs @ embed(q)))[:k]
            return [{**chunks[int(i)], "score": float(vecs[int(i)] @ embed(q))} for i in top]

    retrievers["dense"] = Dense()
    retrievers["hybrid"] = HybridRetriever(bm25, vecs, embed)
    log("dense retriever ready")
except Exception as e:  # keep the BM25 result even if the model download fails
    log(f"dense retriever unavailable: {type(e).__name__}: {str(e)[:150]}")

results = {"sources": report, "n_chunks": len(chunks), "questions": len(usable), "retrieval": {}}
for name, ret in retrievers.items():
    stats = {f"phrase_hit@{k}": 0 for k in (1, 3, 5)} | {f"source_hit@{k}": 0 for k in (1, 3, 5)}
    misses = []
    for question, sources, phrases in usable:
        hits = ret.search(question, 5)
        for k in (1, 3, 5):
            top = hits[:k]
            stats[f"source_hit@{k}"] += any(h["source_id"] in sources for h in top)
            stats[f"phrase_hit@{k}"] += any(h["source_id"] in sources and any(p in h["text"].lower() for p in phrases) for h in top)
        if not any(h["source_id"] in sources and any(p in h["text"].lower() for p in phrases) for h in hits):
            misses.append(question)
    results["retrieval"][name] = {k: v / max(len(usable), 1) for k, v in stats.items()} | {"misses_at_5": misses}
    if name == "bm25":
        answered = abstained = answer_hits = 0
        for question, sources, phrases in usable:
            out = answer(question, ret)
            if out["mode"] == "none":
                abstained += 1
            else:
                answered += 1
                answer_hits += any(p in out["answer"].lower() for p in phrases)
        results["extractive_answers"] = {"answerable_questions": len(usable), "abstained": abstained,
                                         "answer_contains_key_phrase": answer_hits,
                                         "answer_hit_rate_of_answered": answer_hits / max(answered, 1)}
        off_topic = ["What is the GST rate on gold jewellery?", "How do I cook biryani?", "Who won the cricket world cup in 2011?",
                     "What is the tax treatment of cryptocurrency gains?"]
        refused = sum(answer(q, ret)["mode"] == "none" for q in off_topic)
        results["off_topic_refusals"] = {"questions": len(off_topic), "refused": refused}
        log(f"extractive answers: {results['extractive_answers']}; off-topic refused {refused}/{len(off_topic)}")
    log(f"{name}: " + " ".join(f"{k}={v / max(len(usable), 1):.2f}" for k, v in stats.items()))

demo_q = ["Within how many working days must a customer report an unauthorised transaction?",
          "What does the framework say about explainability of AI decisions?"]
results["qa_demo"] = [answer(q, bm25) for q in demo_q]
sample = {"action": "CHALLENGE", "calibrated_probability": 0.31,
          "reasons": [{"text": "this user had a 69% fraud rate in earlier labeled transactions", "direction": "increases_risk"},
                      {"text": "seconds since this user's previous transaction = 41", "direction": "increases_risk"}],
          "flags": {"new_user": False, "uid_has_labeled_history": True, "uid_has_prior_fraud": True, "weak_identity": False}}
results["explain_demo"] = explain_decision(sample, bm25)
json.dump(results, open(f"{OUT}/results.json", "w"), indent=1, ensure_ascii=False)

print("\n=== SOURCES ===")
for r in report:
    print(r)
print("\n=== RETRIEVAL (labeled phrase hits) ===")
for name, r in results["retrieval"].items():
    print(name, {k: round(v, 2) for k, v in r.items() if k != "misses_at_5"})
    for m in r["misses_at_5"]:
        print("   miss:", m)
print("\n=== QA DEMO ===")
for d in results["qa_demo"]:
    print(d["answer"][:500], "| cites:", [(c["title"][:40], c["page"]) for c in d["citations"][:2]])
print("\n=== EXPLAIN DEMO ===")
print(results["explain_demo"]["explanation"])
for c in results["explain_demo"]["regulatory_context"]:
    print("  context:", c["title"][:60], "p.", c["page"], "|", c["excerpt"][:140])
log("done")
