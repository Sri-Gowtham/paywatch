"""Plain-English explanation of a scoring decision, grounded in the model's SHAP reasons and (optionally)
retrieved RBI passages. The factual core is a TEMPLATE over the API response, so nothing is invented;
an LLM, if configured, may only rephrase those facts."""

from typing import Callable, Dict, List, Optional

from .qa_chain import citation

ACTION_TEXT = {
    "ALLOW": "allowed without friction",
    "SOFT_FLAG": "allowed but flagged for monitoring",
    "CHALLENGE": "held for additional customer authentication",
    "HARD_BLOCK": "blocked as highly likely fraud",
}
# (query, substrings of which at least one must appear in the passage): keeps context on-topic
ACTION_QUERIES = {
    "HARD_BLOCK": [("declining a suspected fraudulent transaction and informing the customer", ["fraud"]),
                   ("early warning signals for fraud detection", ["early warning"])],
    "CHALLENGE": [("additional authentication factor for a risky digital payment transaction", ["authenticat"])],
    "SOFT_FLAG": [("transaction monitoring and fraud detection alerts", ["monitor"])],
    "ALLOW": [],
}
EXPLAINABILITY_QUERY = ("explainability of AI model decisions", ["explainab"])


def describe_flags(flags: Dict) -> List[str]:
    out = []
    if flags.get("new_user"):
        out.append("no earlier transactions have been seen for this user, so there is no behavioural baseline")
    if flags.get("uid_has_prior_fraud"):
        out.append("a previously confirmed fraud is linked to this user")
    elif flags.get("uid_has_labeled_history"):
        out.append("this user has earlier labeled transactions and none were confirmed fraud")
    if flags.get("weak_identity"):
        out.append("the identity could not be anchored well (account-age or address fields are missing)")
    return out


def template_explanation(prediction: Dict, max_reasons: int = 4) -> str:
    action = prediction.get("action", "ALLOW")
    p = prediction.get("calibrated_probability", 0.0)
    lines = [f"This transaction was {ACTION_TEXT.get(action, action)} (calibrated fraud probability {p:.0%})."]
    reasons = prediction.get("reasons", [])[:max_reasons]
    if reasons:
        lines.append("Strongest factors in the model's score (attributions, not proof of intent):")
        for r in reasons:
            arrow = "raises" if r.get("direction") == "increases_risk" else "lowers"
            lines.append(f"- {r['text']} ({arrow} the score)")
    ctx = describe_flags(prediction.get("flags", {}))
    if ctx:
        lines.append("Context: " + "; ".join(ctx) + ".")
    return "\n".join(lines)


def regulatory_context(prediction: Dict, retriever, max_items: int = 3, min_coverage: float = 0.5) -> List[Dict]:
    queries = list(ACTION_QUERIES.get(prediction.get("action", "ALLOW"), [])) + [EXPLAINABILITY_QUERY]
    out, seen = [], set()
    for query, required in queries:
        for hit in retriever.search(query, 5):
            text = hit["text"].lower()
            if hit["id"] in seen or hit.get("coverage", 1.0) < min_coverage or not any(r in text for r in required):
                continue
            seen.add(hit["id"])
            out.append(citation(hit, len(out) + 1, focus=required))
            break
        if len(out) >= max_items:
            break
    return out


def explain_decision(prediction: Dict, retriever=None, llm: Optional[Callable[[str], str]] = None) -> Dict:
    base = template_explanation(prediction)
    context = regulatory_context(prediction, retriever) if retriever is not None else []
    text, mode = base, "template"
    if llm is not None:
        prompt = ("Rewrite the following facts as a short explanation (3 sentences max) for a fraud analyst. "
                  "Use only these facts; do not add or infer anything.\n\n" + base)
        text, mode = llm(prompt).strip(), "llm"
    return {"explanation": text, "template_explanation": base, "regulatory_context": context, "mode": mode}
