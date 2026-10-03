"""Fetch public documents, extract text, split into overlapping chunks, save as JSONL.

Heavy/optional dependencies (requests, pypdf, bs4) are imported lazily so the rest of the RAG package
(retriever, qa_chain, explainer) stays dependency-free. Run on Kaggle (internet on); the output index is
kept out of the repo.
"""

import json
import re
from typing import Callable, Dict, Iterable, List, Optional, Tuple

USER_AGENT = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) "
              "Chrome/124.0 Safari/537.36 PayWatch-research")


def fetch(url: str, timeout: int = 120, attempts: int = 3) -> bytes:
    """Download with browser-like headers; PDFs are validated (magic header + trailing %%EOF) and retried when
    the server returns a truncated body or an HTML block page. Failures carry diagnostics."""
    import requests

    headers = {"User-Agent": USER_AGENT, "Accept": "application/pdf,text/html;q=0.9,*/*;q=0.8",
               "Accept-Encoding": "identity", "Referer": "https://www.rbi.org.in/"}
    expect_pdf = url.lower().endswith(".pdf")
    last = "no attempt"
    for _ in range(attempts):
        try:
            r = requests.get(url, headers=headers, timeout=timeout)
        except Exception as e:
            last = f"{type(e).__name__}: {str(e)[:120]}"
            continue
        body = r.content
        diag = f"HTTP {r.status_code} content-type={r.headers.get('Content-Type', '')} bytes={len(body)} head={body[:40]!r}"
        if not r.ok:
            last = diag
            continue
        if expect_pdf and not (body[:5] == b"%PDF-" and b"%%EOF" in body[-2048:]):
            last = "invalid or truncated PDF: " + diag
            continue
        return body
    raise RuntimeError(last)


def clean(text: str) -> str:
    text = text.replace(" ", " ").replace("\x00", "")
    text = re.sub(r"-\n(?=[a-z])", "", text)            # hyphenated line breaks
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def parse_pdf(content: bytes) -> List[Tuple[int, str]]:
    import io

    from pypdf import PdfReader

    reader = PdfReader(io.BytesIO(content))
    pages = []
    for i, page in enumerate(reader.pages, start=1):
        try:
            pages.append((i, clean(page.extract_text() or "")))
        except Exception:
            pages.append((i, ""))
    return [(p, t) for p, t in pages if t]


def parse_html(content: bytes) -> List[Tuple[int, str]]:
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(content, "html.parser")
    for tag in soup(["script", "style", "nav", "header", "footer", "noscript"]):
        tag.decompose()
    text = clean(soup.get_text("\n"))
    return [(1, text)] if text else []


def split_text(text: str, size: int = 1100, overlap: int = 150) -> List[str]:
    """Paragraph-aware splitter: pack paragraphs up to `size` chars, carry `overlap` chars into the next chunk."""
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    pieces: List[str] = []
    for p in paragraphs:
        if len(p) <= size:
            pieces.append(p)
        else:                                            # very long paragraph: split on sentence boundaries
            sentences = re.split(r"(?<=[.;:])\s+", p)
            buf = ""
            for s in sentences:
                if buf and len(buf) + len(s) + 1 > size:
                    pieces.append(buf)
                    buf = s
                else:
                    buf = f"{buf} {s}".strip()
            if buf:
                pieces.append(buf)
    chunks, cur = [], ""
    for piece in pieces:
        if cur and len(cur) + len(piece) + 1 > size:
            chunks.append(cur)
            cur = (cur[-overlap:] + " " + piece).strip() if overlap else piece
        else:
            cur = f"{cur}\n{piece}".strip()
    if cur:
        chunks.append(cur)
    return chunks


def chunk_pages(pages: Iterable[Tuple[int, str]], source: Dict, size: int = 1100, overlap: int = 150,
                min_chars: int = 120) -> List[Dict]:
    out = []
    for page_no, text in pages:
        for j, piece in enumerate(split_text(text, size, overlap)):
            if len(piece) < min_chars:
                continue
            out.append({"id": f"{source['id']}:p{page_no}:{j}", "source_id": source["id"], "title": source["title"],
                        "url": source["url"], "page": page_no, "text": piece})
    return out


def build(sources: List[Dict], fetcher: Callable[[str], bytes] = fetch) -> Tuple[List[Dict], List[Dict]]:
    """Returns (chunks, per-source report). A failing source is reported, not fatal."""
    chunks, report = [], []
    for s in sources:
        try:
            content = fetcher(s["url"])
            pages = parse_pdf(content) if s["kind"] == "pdf" else parse_html(content)
            cs = chunk_pages(pages, s)
            chunks += cs
            report.append({"id": s["id"], "ok": True, "bytes": len(content), "pages": len(pages), "chunks": len(cs),
                           "chars": sum(len(c["text"]) for c in cs)})
        except Exception as e:  # network/parse failures are recorded and the build continues
            report.append({"id": s["id"], "ok": False, "error": f"{type(e).__name__}: {str(e)[:200]}"})
    return chunks, report


def save_chunks(chunks: List[Dict], path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")


def load_chunks(path: str) -> List[Dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]
