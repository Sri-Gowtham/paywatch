"""Question answering over the retrieved RBI passages.

Default mode is EXTRACTIVE (best matching sentences with numbered citations): it needs no API key and cannot
invent text. If ANTHROPIC_API_KEY is set (and the `anthropic` package is installed) an LLM can compose the
answer, constrained to the retrieved excerpts.
"""

import os
import re
from typing import Callable, Dict, List, Optional

from .retriever import tokenize

SYSTEM_PROMPT = ("You answer questions for a fraud-operations analyst using ONLY the numbered excerpts from Reserve Bank "
                 "of India documents. Cite excerpts like [1]. If the excerpts do not contain the answer, say so plainly. "
                 "Never add facts that are not in the excerpts.")
DEFAULT_MODEL = "claude-sonnet-5-5"


def anthropic_llm(model: str = DEFAULT_MODEL, max_tokens: int = 500) -> Optional[Callable[[str], str]]:
    """Returns a prompt->text callable, or None when no key / package is available (caller falls back)."""
    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        return None
    try:
        import anthropic
    except ImportError:
        return None
    client = anthropic.Anthropic(api_key=key)

    def call(prompt: str) -> str:
        resp = client.messages.create(model=model, max_tokens=max_tokens, system=SYSTEM_PROMPT,
                                      messages=[{"role": "user", "content": prompt}])
        return "".join(b.text for b in resp.content if getattr(b, "type", "") == "text")

    return call


def excerpt(text: str, focus: Optional[List[str]] = None, width: int = 300) -> str:
    """First `width` chars, or a window around the first occurrence of a focus substring."""
    flat = text.replace("\n", " ")
    if focus:
        low = flat.lower()
        positions = [low.find(f) for f in focus if low.find(f) >= 0]
        if positions:
            start = max(0, min(positions) - width // 3)
            return ("..." if start else "") + flat[start:start + width]
    return flat[:width]


def citation(hit: Dict, n: int, focus: Optional[List[str]] = None) -> Dict:
    return {"n": n, "title": hit["title"], "url": hit["url"], "page": hit["page"], "score": round(hit.get("score", 0.0), 3),
            "excerpt": excerpt(hit["text"], focus)}


def best_sentences(question: str, hits: List[Dict], n: int = 3, idf: Optional[Dict[str, float]] = None,
                   use_anchor: bool = True) -> List[Dict]:
    """Pick the sentences that best answer the question. Common query words (e.g. 'report', 'framework') must
    not outvote the distinctive one, so sentences must contain the rarest query term found in the hits."""
    q = set(tokenize(question))
    anchor = None
    if use_anchor and hits:
        present = [t for t in q if any(t in set(tokenize(h["text"])) for h in hits)]
        if present:
            anchor = max(present, key=lambda t: (idf or {}).get(t, 1.0))
    scored = []
    for rank, hit in enumerate(hits):
        for sent in re.split(r"(?<=[.!?;])\s+|\n+", hit["text"]):
            sent = sent.strip()
            if len(sent) < 40:
                continue
            toks = set(tokenize(sent))
            if anchor and anchor not in toks:
                continue
            matched = q & toks
            overlap = sum((idf or {}).get(t, 1.0) for t in matched) if idf else len(matched)
            if overlap:
                scored.append((overlap - 0.15 * rank, rank, sent))
    if not scored and anchor:
        return best_sentences(question, hits, n, idf, use_anchor=False)
    scored.sort(key=lambda t: -t[0])
    out, seen = [], set()
    for _, rank, sent in scored:
        key = sent[:60]
        if key in seen:
            continue
        seen.add(key)
        out.append({"hit": rank, "sentence": sent})
        if len(out) == n:
            break
    return out


NO_ANSWER = "This is not covered by the indexed RBI documents, so I cannot answer it from them."


def answer(question: str, retriever, k: int = 4, llm: Optional[Callable[[str], str]] = None,
           min_coverage: float = 0.5) -> Dict:
    """Retrieve, drop passages that cover too few of the question's terms, then answer or abstain."""
    hits = [h for h in retriever.search(question, k) if h.get("coverage", 1.0) >= min_coverage]
    if not hits:
        return {"answer": NO_ANSWER, "citations": [], "mode": "none"}
    cites = [citation(h, i + 1) for i, h in enumerate(hits)]
    if llm is not None:
        excerpts = "\n\n".join(f"[{i + 1}] ({h['title']}, p.{h['page']})\n{h['text']}" for i, h in enumerate(hits))
        prompt = f"Excerpts:\n{excerpts}\n\nQuestion: {question}"
        return {"answer": llm(prompt).strip(), "citations": cites, "mode": "llm"}
    picks = best_sentences(question, hits, idf=getattr(retriever, 'idf', None))
    if not picks:
        return {"answer": "The retrieved passages do not contain a direct statement on this; see the citations.",
                "citations": cites, "mode": "extractive"}
    text = " ".join(f"{p['sentence']} [{p['hit'] + 1}]" for p in picks)
    return {"answer": text, "citations": cites, "mode": "extractive"}
