"""Step 5 — build the hybrid retrieval index from kb.jsonl and measure retrieval quality.

Output dir (default data/index/) is what gets uploaded to the private HF dataset and loaded by the Space.

Usage:
    source scripts/env.sh
    python -m src.rag.build_index                    # build
    python -m src.rag.build_index --eval-paraphrases data/processed/eval_paraphrases.jsonl
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "space"))
from retriever import HybridRetriever, doc_text  # noqa: E402

EMBED_MODEL = "BAAI/bge-m3"


def build(kb_path: Path, out_dir: Path, device: str):
    import faiss
    from sentence_transformers import SentenceTransformer

    records = [json.loads(l) for l in kb_path.open(encoding="utf-8")]
    model = SentenceTransformer(EMBED_MODEL, device=device)
    emb = model.encode([doc_text(r) for r in records], normalize_embeddings=True, batch_size=32,
                       convert_to_numpy=True, show_progress_bar=True).astype("float32")
    index = faiss.IndexFlatIP(emb.shape[1])
    index.add(emb)
    out_dir.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(out_dir / "kb.faiss"))
    qemb = model.encode([r["question"] for r in records], normalize_embeddings=True, batch_size=64,
                        convert_to_numpy=True, show_progress_bar=False).astype("float32")
    qindex = faiss.IndexFlatIP(qemb.shape[1])
    qindex.add(qemb)
    faiss.write_index(qindex, str(out_dir / "kb_q.faiss"))
    shutil.copy(kb_path, out_dir / "kb.jsonl")
    (out_dir / "config.json").write_text(json.dumps({"embed_model": EMBED_MODEL, "dim": int(emb.shape[1]),
                                                     "n_records": len(records)}, indent=1))
    print(f"indexed {len(records)} records (dim {emb.shape[1]}) -> {out_dir}")
    return model


def evaluate(index_dir: Path, pairs_path: Path, model, ks=(1, 3, 5)):
    """pairs: {"id": kb id, "query": paraphrased user question}; reports recall@k."""
    r = HybridRetriever(index_dir, embed_model=model)
    pairs = [json.loads(l) for l in pairs_path.open(encoding="utf-8")]
    hits = {k: 0 for k in ks}
    for p in pairs:
        ids = [h.rec["id"] for h in r.search(p["query"], k=max(ks))]
        for k in ks:
            hits[k] += p["id"] in ids[:k]
    print("retrieval:", {f"recall@{k}": round(hits[k] / len(pairs), 3) for k in ks}, f"(n={len(pairs)})")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--kb", default="data/processed/kb.jsonl")
    ap.add_argument("--out", default="data/index")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--eval-paraphrases", default="")
    args = ap.parse_args()
    model = build(Path(args.kb), Path(args.out), args.device)
    if args.eval_paraphrases:
        evaluate(Path(args.out), Path(args.eval_paraphrases), model)


if __name__ == "__main__":
    main()
