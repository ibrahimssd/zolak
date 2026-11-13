"""Hybrid retriever shared by the index builder (offline) and the Hugging Face Space (runtime).

dense : BAAI/bge-m3 embeddings, FAISS inner-product (cosine; vectors are normalized)
sparse: BM25 over Arabic-normalized tokens (catches German terms: Sperrkonto, Anmeldung, ...)
fusion: reciprocal-rank fusion (RRF)

Index directory layout (built by src/rag/build_index.py):
    kb.faiss         dense index (row i == line i of kb.jsonl)
    kb.jsonl         the knowledge-base records
    config.json      {"embed_model": ..., "dim": ...}
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

AR_DIACRITICS = re.compile("[ؐ-ًؚ-ٰٟۖ-ۭـ]")
AR_NORMALIZE = str.maketrans({"أ": "ا", "إ": "ا", "آ": "ا", "ٱ": "ا", "ى": "ي", "ة": "ه", "ؤ": "و", "ئ": "ي"})
DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹", "0123456789" * 2)
# very frequent function words (MSA + Sudanese) that only add noise to BM25
STOP = set("""في من على الى إلى عن مع ده دا دي ديل دة هو هي انا انت انتو نحن هم ما لا لو او أو و يا ال الي اللي البي
بتاع بتاعت تبع كدا كده برضو بس عشان علشان ممكن زول شنو كيف وين متين هل كان بكون يكون عندي عندو عندها ليك ليه ليها
the a an of to in on for and or is are be with from at by der die das und ist ein eine""".split())


def normalize_ar(text: str) -> str:
    t = AR_DIACRITICS.sub("", text.translate(DIGITS)).translate(AR_NORMALIZE).lower()
    return re.sub(r"(.)\1{2,}", r"\1\1", t)


def tokenize(text: str) -> list[str]:
    toks = re.findall(r"[^\W_]+", normalize_ar(text))
    out = []
    for t in toks:
        if t in STOP or len(t) < 2:
            continue
        # light Arabic prefix stripping: ال / وال / بال / فال / لل  (keeps the stem searchable)
        t2 = re.sub(r"^(وال|بال|فال|كال|ال|لل)(?=..)", "", t)
        out.append(t2)
    return out


def doc_text(rec: dict) -> str:
    """Text that represents a KB record for both dense and sparse search."""
    return "\n".join([rec["question"], rec["answer"], *rec.get("facts", []), " ".join(rec.get("german_terms", []))])


@dataclass
class Hit:
    rec: dict
    score: float                 # fused RRF score (ranking only)
    dense_rank: int | None
    bm25_rank: int | None
    dense_sim: float = 0.0       # cosine similarity question<->record (use for relevance thresholds)


class HybridRetriever:
    def __init__(self, index_dir: str | Path, embed_model=None, device: str = "cpu"):
        import faiss
        from rank_bm25 import BM25Okapi

        d = Path(index_dir)
        self.cfg = json.loads((d / "config.json").read_text())
        self.records = [json.loads(l) for l in (d / "kb.jsonl").open(encoding="utf-8")]
        self.index = faiss.read_index(str(d / "kb.faiss"))
        self.bm25 = BM25Okapi([tokenize(doc_text(r)) for r in self.records])
        if embed_model is None:
            from sentence_transformers import SentenceTransformer
            embed_model = SentenceTransformer(self.cfg["embed_model"], device=device)
        self.embedder = embed_model

    def embed(self, texts: list[str]):
        return self.embedder.encode(texts, normalize_embeddings=True, convert_to_numpy=True).astype("float32")

    def search(self, query: str, k: int = 3, pool: int = 30, rrf_k: int = 60, qvec=None) -> list[Hit]:
        import numpy as np

        qv = self.embed([query]) if qvec is None else qvec.reshape(1, -1)
        dscores, didx = self.index.search(qv, min(pool, len(self.records)))
        dense = [int(i) for i in didx[0] if i >= 0]
        dsim = {int(i): float(s) for i, s in zip(didx[0], dscores[0]) if i >= 0}
        bscores = self.bm25.get_scores(tokenize(query))
        sparse = [int(i) for i in np.argsort(-bscores)[:pool] if bscores[i] > 0]
        fused: dict[int, float] = {}
        for rank, i in enumerate(dense):
            fused[i] = fused.get(i, 0) + 1 / (rrf_k + rank + 1)
        for rank, i in enumerate(sparse):
            fused[i] = fused.get(i, 0) + 1 / (rrf_k + rank + 1)
        top = sorted(fused, key=fused.get, reverse=True)[:k]
        if any(i not in dsim for i in top):  # BM25-only hits: compute their dense similarity directly
            vecs = self.index.reconstruct_batch(np.array(top, dtype="int64"))
            dsim.update({i: float(v @ qv[0]) for i, v in zip(top, vecs)})
        return [Hit(self.records[i], fused[i], dense.index(i) if i in dense else None,
                    sparse.index(i) if i in sparse else None, dsim[i]) for i in top]
