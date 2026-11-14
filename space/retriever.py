"""Hybrid retriever shared by the index builder (offline) and the Hugging Face Space (runtime).

dense : BAAI/bge-m3 embeddings, FAISS inner-product (cosine; vectors are normalized)
sparse: BM25 over Arabic-normalized tokens (catches German terms: Sperrkonto, Anmeldung, ...)
fusion: reciprocal-rank fusion (RRF) of question similarity (weight 2), record similarity and BM25

Index directory layout (built by src/rag/build_index.py):
    kb.faiss         dense index of whole records (row i == line i of kb.jsonl)
    kb_q.faiss       dense index of the records' questions only (optional)
    kb.jsonl         the knowledge-base records
    config.json      {"embed_model": ..., "dim": ...}
"""
from __future__ import annotations

import json
import os
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
        # question-only index: a student's question is closest to the record's question (FAQ-style matching)
        qpath = d / "kb_q.faiss"
        self.qindex = faiss.read_index(str(qpath)) if qpath.exists() else None
        self.bm25 = BM25Okapi([tokenize(doc_text(r)) for r in self.records])
        # group record id -> summary of its question cluster (a summary consolidates several discussions)
        self.summary_of = {m: r for r in self.records if r.get("source") == "summary" for m in r.get("members", [])}
        if embed_model is None:
            from sentence_transformers import SentenceTransformer
            embed_model = SentenceTransformer(self.embed_path(), device=device)
        self.embedder = embed_model

    def embed_path(self) -> str:
        """Hub id of the embedder, or a local copy of a fine-tuned one kept in a private model repo
        (config "embed_repo"; LOCAL_EMBED overrides for offline runs)."""
        name = self.cfg["embed_model"]
        if not self.cfg.get("embed_repo"):
            return name
        if os.environ.get("LOCAL_EMBED"):
            return os.environ["LOCAL_EMBED"]
        from huggingface_hub import snapshot_download
        root = snapshot_download(self.cfg["embed_repo"], allow_patterns=[f"{name}/*"], token=os.environ.get("HF_TOKEN"))
        return str(Path(root) / name)

    def embed(self, texts: list[str]):
        return self.embedder.encode(texts, normalize_embeddings=True, convert_to_numpy=True).astype("float32")

    def search(self, query: str, k: int = 3, pool: int = 30, rrf_k: int = 60, qvec=None) -> list[Hit]:
        """RRF over question-similarity (weight 2), whole-record similarity and BM25.
        Hit.dense_sim = max(question sim, record sim), used for the relevance threshold."""
        import numpy as np

        qv = self.embed([query]) if qvec is None else qvec.reshape(1, -1)
        n = min(pool, len(self.records))
        dscores, didx = self.index.search(qv, n)
        dense = [int(i) for i in didx[0] if i >= 0]
        sim = {int(i): float(s) for i, s in zip(didx[0], dscores[0]) if i >= 0}
        qlist: list[int] = []
        if self.qindex is not None:
            qscores, qidx = self.qindex.search(qv, n)
            qlist = [int(i) for i in qidx[0] if i >= 0]
            for i, s in zip(qidx[0], qscores[0]):
                if i >= 0:
                    sim[int(i)] = max(sim.get(int(i), -1.0), float(s))
        bscores = self.bm25.get_scores(tokenize(query))
        sparse = [int(i) for i in np.argsort(-bscores)[:pool] if bscores[i] > 0]
        fused: dict[int, float] = {}
        for weight, ranking in ((2.0, qlist), (1.0, dense), (1.0, sparse)):
            for rank, i in enumerate(ranking):
                fused[i] = fused.get(i, 0) + weight / (rrf_k + rank + 1)
        top = sorted(fused, key=fused.get, reverse=True)[:k]
        missing = [i for i in top if i not in sim]
        if missing:  # BM25-only hits: compute their similarity directly
            vecs = self.index.reconstruct_batch(np.array(missing, dtype="int64"))
            sim.update({i: float(v @ qv[0]) for i, v in zip(missing, vecs)})
        return [Hit(self.records[i], fused[i], dense.index(i) if i in dense else None,
                    sparse.index(i) if i in sparse else None, sim[i]) for i in top]

    def top_entries(self, query: str, k: int = 2, exclude_ids: set | frozenset = frozenset()) -> list[dict]:
        """The k entries the chatbot answers from, with no similarity cut-off. A hit that belongs to a
        summarized cluster is replaced by that cluster's summary, and duplicates collapse, so the k entries
        cover different discussions. `exclude_ids` hides records (and summaries) from the ranking, e.g. the
        held-out eval records while building training data."""
        out, seen = [], set()
        for h in self.search(query, k=4 * k + len(exclude_ids)):
            if h.rec["id"] in exclude_ids:
                continue
            rec = self.summary_of.get(h.rec["id"], h.rec)
            if rec["id"] in exclude_ids:
                rec = h.rec
            if rec["id"] not in seen:
                seen.add(rec["id"])
                out.append(rec)
                if len(out) == k:
                    break
        return out
