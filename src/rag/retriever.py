"""BM25 retriever (pure Python, no heavy dependencies) with an optional dense/hybrid mode for evaluation."""

import math
import re
from collections import Counter, defaultdict
from typing import Callable, Dict, List, Optional, Sequence

STOP = frozenset("""a an and are as at be been but by can could did do does for from had has have how i if in into is it its
may of on or our should so such than that the their then there these they this those to under was we were what when
where which who whom why will with would you your shall also any all each other per than not no yes""".split())
_TOKEN = re.compile(r"[a-z0-9]+")


def stem(tok: str) -> str:
    for suffix in ("ations", "ation", "ings", "ing", "edly", "ed", "ies", "es", "s"):
        if tok.endswith(suffix) and len(tok) - len(suffix) >= 3:
            return tok[: -len(suffix)] + ("y" if suffix == "ies" else "")
    return tok


def tokenize(text: str) -> List[str]:
    return [stem(t) for t in _TOKEN.findall(text.lower()) if t not in STOP and len(t) > 1]


class BM25Retriever:
    def __init__(self, chunks: Sequence[Dict], k1: float = 1.5, b: float = 0.75):
        self.chunks = list(chunks)
        self.k1, self.b = k1, b
        self.doc_tokens = [tokenize(c["text"]) for c in self.chunks]
        self.doc_len = [len(t) for t in self.doc_tokens]
        self.avgdl = (sum(self.doc_len) / len(self.doc_len)) if self.doc_len else 0.0
        self.index = defaultdict(list)
        for i, toks in enumerate(self.doc_tokens):
            for term, tf in Counter(toks).items():
                self.index[term].append((i, tf))
        n = len(self.chunks)
        self.idf = {t: math.log(1 + (n - len(p) + 0.5) / (len(p) + 0.5)) for t, p in self.index.items()}

    def scores(self, query: str) -> Dict[int, float]:
        sc: Dict[int, float] = defaultdict(float)
        for term in set(tokenize(query)):
            for i, tf in self.index.get(term, ()):
                norm = tf + self.k1 * (1 - self.b + self.b * self.doc_len[i] / (self.avgdl or 1.0))
                sc[i] += self.idf[term] * tf * (self.k1 + 1) / norm
        return sc

    def coverage(self, query: str, doc_index: int) -> float:
        """Share of the query's distinct content terms that occur in the chunk (0..1)."""
        terms = set(tokenize(query))
        return len(terms & set(self.doc_tokens[doc_index])) / len(terms) if terms else 0.0

    def search(self, query: str, k: int = 5) -> List[Dict]:
        ranked = sorted(self.scores(query).items(), key=lambda kv: -kv[1])[:k]
        return [{**self.chunks[i], "score": float(s), "coverage": self.coverage(query, i)} for i, s in ranked]


class HybridRetriever:
    """Reciprocal-rank fusion of BM25 and a dense scorer (embed: str -> vector; doc_vectors aligned with chunks).
    Used for evaluation on Kaggle; the serving path uses BM25 only (no torch on small machines)."""

    def __init__(self, bm25: BM25Retriever, doc_vectors, embed: Callable, rrf_k: int = 60):
        import numpy as np

        self.bm25, self.embed, self.rrf_k = bm25, embed, rrf_k
        self.vectors = np.asarray(doc_vectors, dtype="float32")

    def search(self, query: str, k: int = 5) -> List[Dict]:
        import numpy as np

        sparse = sorted(self.bm25.scores(query).items(), key=lambda kv: -kv[1])[:50]
        q = np.asarray(self.embed(query), dtype="float32")
        dense = np.argsort(-(self.vectors @ q))[:50]
        fused: Dict[int, float] = defaultdict(float)
        for rank, (i, _) in enumerate(sparse):
            fused[i] += 1.0 / (self.rrf_k + rank + 1)
        for rank, i in enumerate(dense):
            fused[int(i)] += 1.0 / (self.rrf_k + rank + 1)
        top = sorted(fused.items(), key=lambda kv: -kv[1])[:k]
        return [{**self.bm25.chunks[i], "score": float(s)} for i, s in top]
