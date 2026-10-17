"""Question answering over the retrieved RBI passages.

Default mode is EXTRACTIVE (best matching sentences with numbered citations): it needs no API key and cannot
invent text. If ANTHROPIC_API_KEY is set (and the `anthropic` package is installed), or GEMINI_API_KEY (a free
Google AI Studio key), an LLM can compose the answer, constrained to the retrieved excerpts. If the LLM call fails (rate limit,
network) the answer falls back to the extractive one.
"""

import json
import os
import re
import urllib.error
import urllib.request
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


DEFAULT_GEMINI_MODEL = "gemini-2.5-flash"   # override with the GEMINI_MODEL env var if Google renames or retires it


def gemini_llm(model: Optional[str] = None, max_tokens: int = 500) -> Optional[Callable[[str], str]]:
    """Free-tier alternative: Google AI Studio key in GEMINI_API_KEY (or GOOGLE_API_KEY). Plain HTTPS, no extra package.
    Returns a prompt->text callable, or None when no key is set."""
    key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not key:
        return None
    name = model or os.environ.get("GEMINI_MODEL", DEFAULT_GEMINI_MODEL)
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{name}:generateContent"

    def call(prompt: str) -> str:
        body = json.dumps({"system_instruction": {"parts": [{"text": SYSTEM_PROMPT}]},
                           "contents": [{"role": "user", "parts": [{"text": prompt}]}],
                           "generationConfig": {"maxOutputTokens": max_tokens, "temperature": 0.2}}).encode("utf-8")
        req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json", "x-goog-api-key": key}, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                data = json.loads(r.read())
        except urllib.error.HTTPError as e:   # surface Google's own message (bad key, unknown model, rate limit); it never echoes the key
            detail = " ".join(e.read().decode("utf-8", "replace")[:240].split())
            raise RuntimeError(f"Gemini HTTP {e.code} for model {name}: {detail}") from None
        parts = data["candidates"][0]["content"]["parts"]
        return "".join(p.get("text", "") for p in parts)

    return call


def default_llm() -> Optional[Callable[[str], str]]:
    """The first configured provider: Anthropic (ANTHROPIC_API_KEY), else Gemini (GEMINI_API_KEY); None means extractive mode."""
    return anthropic_llm() or gemini_llm()


def provider_name() -> str:
    if os.environ.get("ANTHROPIC_API_KEY") and anthropic_llm() is not None:
        return "Anthropic"
    return "Gemini" if gemini_llm() is not None else "none"


def _tidy(piece: str, cut_end: bool) -> str:
    """Passages are cut at fixed character offsets, so a slice can start or stop inside a word. Drop a leading fragment such as
    'ion. (b) Limited...' (a lowercase stub ending at a sentence boundary) and, when the slice stops mid-text, a trailing partial word."""
    m = re.match(r"[a-z][^.]{0,38}\.\s+(?=[A-Z(])", piece)
    if m:
        piece = piece[m.end():]
    if cut_end and not re.search(r"[.!?;:)\]]$", piece):
        cut = piece.rfind(" ")
        if cut > len(piece) * 0.6:
            piece = piece[:cut] + "…"
    return piece


def excerpt(text: str, focus: Optional[List[str]] = None, width: int = 300) -> str:
    """First `width` chars, or a window around the first occurrence of a focus substring (tidied to whole words)."""
    flat = text.replace("\n", " ")
    if focus:
        low = flat.lower()
        positions = [low.find(f) for f in focus if low.find(f) >= 0]
        if positions:
            start = max(0, min(positions) - width // 3)
            return ("..." if start else "") + _tidy(flat[start:start + width], start + width < len(flat))
    return _tidy(flat[:width], len(flat) > width)


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
            if len(sent) < 40 or re.match(r"(?i)dear\s+sir", sent):   # skip letter salutations
                continue
            toks = set(tokenize(sent))
            if anchor and anchor not in toks:
                continue
            matched = q & toks
            overlap = sum((idf or {}).get(t, 1.0) for t in matched) if idf else len(matched)
            if overlap:
                truncated = not re.search(r"[.!?;:)\]\"]$", sent)        # a passage's last sentence may be cut off
                scored.append((overlap - 0.15 * rank - (1.5 if truncated else 0.0), rank, sent))
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
        try:
            return {"answer": llm(prompt).strip(), "citations": cites, "mode": "llm"}
        except Exception as e:  # rate limit, network or provider error: fall back to the extractive answer below
            llm_error = f"{type(e).__name__}: {str(e)[:120]}"
    picks = best_sentences(question, hits, idf=getattr(retriever, 'idf', None))
    extra = {"llm_error": llm_error} if llm is not None else {}
    if not picks:
        return {"answer": "The retrieved passages do not contain a direct statement on this; see the citations.",
                "citations": cites, "mode": "extractive", **extra}
    text = " ".join(f"{p['sentence']} [{p['hit'] + 1}]" for p in picks)
    return {"answer": text, "citations": cites, "mode": "extractive", **extra}
