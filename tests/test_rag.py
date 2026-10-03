import pytest

from src.rag.explainer import describe_flags, explain_decision, template_explanation
from src.rag.ingest import build, chunk_pages, clean, split_text
from src.rag.qa_chain import answer, best_sentences, excerpt
from src.rag.retriever import BM25Retriever, tokenize

CHUNKS = [
    {"id": "liab:p1:0", "source_id": "liab", "title": "Limiting Liability", "url": "u1", "page": 1,
     "text": "A customer has zero liability where the unauthorised transaction occurs due to third party breach and the "
             "customer reports it to the bank within three working days of receiving the communication."},
    {"id": "auth:p2:0", "source_id": "auth", "title": "Digital Payment Security", "url": "u2", "page": 2,
     "text": "Regulated entities shall implement an additional factor of authentication for card not present transactions "
             "and for high risk digital payments."},
    {"id": "ai:p3:0", "source_id": "ai", "title": "FREE-AI", "url": "u3", "page": 3,
     "text": "Financial institutions should ensure explainability and transparency of artificial intelligence models and "
             "maintain a board approved policy on the use of AI."},
    {"id": "fraud:p4:0", "source_id": "fraud", "title": "Fraud Risk Management", "url": "u4", "page": 4,
     "text": "Banks shall report frauds to law enforcement agencies and put in place early warning signals for the "
             "detection of fraud in accounts."},
]
PREDICTION = {"action": "CHALLENGE", "calibrated_probability": 0.31,
              "reasons": [{"text": "this user had a 69% fraud rate in earlier labeled transactions", "direction": "increases_risk"},
                          {"text": "transaction amount = 25", "direction": "decreases_risk"}],
              "flags": {"new_user": False, "uid_has_labeled_history": True, "uid_has_prior_fraud": True, "weak_identity": False}}


def test_tokenize_drops_stopwords_and_stems():
    assert tokenize("The customers are reporting frauds") == ["customer", "report", "fraud"]


def test_bm25_ranks_relevant_chunk_first():
    r = BM25Retriever(CHUNKS)
    assert r.search("zero liability if reported within three working days", 2)[0]["id"] == "liab:p1:0"
    assert r.search("explainability of AI models", 1)[0]["id"] == "ai:p3:0"
    assert r.search("early warning signals for fraud", 1)[0]["id"] == "fraud:p4:0"


def test_bm25_unknown_terms_return_nothing():
    assert BM25Retriever(CHUNKS).search("zzzz qqqq", 3) == []


def test_split_text_respects_size_and_carries_overlap():
    text = "\n\n".join(f"Paragraph number {i} " + "word " * 40 for i in range(12))
    chunks = split_text(text, size=500, overlap=60)
    assert len(chunks) > 3 and max(len(c) for c in chunks) < 700
    assert chunks[1][:30] in chunks[0] or chunks[0][-60:][:20] in chunks[1]


def test_chunk_pages_keeps_page_numbers_and_skips_tiny_chunks():
    src = {"id": "s", "title": "T", "url": "u"}
    out = chunk_pages([(7, "x " * 200), (8, "tiny")], src)
    assert out and all(c["page"] == 7 for c in out) and out[0]["id"].startswith("s:p7:")


def test_clean_joins_hyphenated_breaks():
    assert clean("regu-\nlated   entities") == "regulated entities"


def test_build_reports_failures_without_stopping():
    pytest.importorskip("bs4")
    html = b"<html><body><p>" + b"Banks must report frauds promptly. " * 20 + b"</p></body></html>"

    def fetcher(url):
        if "bad" in url:
            raise RuntimeError("boom")
        return html

    sources = [{"id": "ok", "kind": "html", "title": "OK", "url": "http://ok"},
               {"id": "bad", "kind": "html", "title": "BAD", "url": "http://bad"}]
    chunks, report = build(sources, fetcher)
    assert chunks and report[0]["ok"] and not report[1]["ok"] and "boom" in report[1]["error"]


def test_answer_extractive_has_numbered_citations():
    out = answer("When does a customer have zero liability?", BM25Retriever(CHUNKS), k=2)
    assert out["mode"] == "extractive" and "[1]" in out["answer"] and out["citations"][0]["title"] == "Limiting Liability"


def test_answer_with_no_hits_says_so():
    assert answer("zzzz", BM25Retriever(CHUNKS))["mode"] == "none"


def test_answer_abstains_when_query_terms_are_not_covered():
    out = answer("what does the framework say about quantum blockchain liability taxation", BM25Retriever(CHUNKS))
    assert out["mode"] == "none" and out["citations"] == [] and "not covered" in out["answer"]


def test_search_reports_term_coverage():
    hit = BM25Retriever(CHUNKS).search("zero liability three working days", 1)[0]
    assert hit["id"] == "liab:p1:0" and hit["coverage"] == 1.0


def test_answer_llm_path_receives_excerpts():
    seen = {}

    def fake_llm(prompt):
        seen["prompt"] = prompt
        return " Three working days [1]. "

    out = answer("zero liability reporting window", BM25Retriever(CHUNKS), llm=fake_llm)
    assert out["mode"] == "llm" and out["answer"] == "Three working days [1]." and "Limiting Liability" in seen["prompt"]


def test_best_sentences_prefers_overlap():
    hits = [{"text": "Unrelated filler sentence about something else entirely here. "
                     "Customers have zero liability when they report within three working days."}]
    assert "zero liability" in best_sentences("zero liability report", hits, n=1)[0]["sentence"]


def test_template_explanation_states_facts_only():
    text = template_explanation(PREDICTION)
    assert "held for additional customer authentication" in text and "31%" in text
    assert "69% fraud rate" in text and "raises the score" in text and "lowers the score" in text
    assert "previously confirmed fraud" in text


def test_describe_flags():
    assert describe_flags({"new_user": True, "weak_identity": True}) == [
        "no earlier transactions have been seen for this user, so there is no behavioural baseline",
        "the identity could not be anchored well (account-age or address fields are missing)"]


def test_explain_decision_adds_regulatory_context_and_optional_llm():
    out = explain_decision(PREDICTION, BM25Retriever(CHUNKS))
    assert out["mode"] == "template" and out["regulatory_context"] and out["regulatory_context"][0]["n"] == 1
    out2 = explain_decision(PREDICTION, BM25Retriever(CHUNKS), llm=lambda p: "Rephrased.")
    assert out2["mode"] == "llm" and out2["explanation"] == "Rephrased." and out2["template_explanation"] == out["template_explanation"]


def test_best_sentences_anchor_on_the_rarest_query_term():
    hits = [{"text": "The FREE-AI report sets out the framework and principles of the FREE-AI committee report here. "
                     "Explainability of model decisions must be ensured for affected customers."}]
    idf = {"free": 1.0, "ai": 0.4, "report": 0.5, "framework": 0.6, "explainability": 6.0}
    top = best_sentences("What does the FREE-AI report framework say about explainability?", hits, n=1, idf=idf)
    assert "Explainability of model decisions" in top[0]["sentence"]


def test_excerpt_windows_around_the_focus_term():
    text = "x " * 400 + "additional authentication is required here. " + "y " * 400
    out = excerpt(text, ["authenticat"], width=200)
    assert "authentication" in out and out.startswith("...") and len(out) <= 205
    assert excerpt("short text", None).startswith("short")
