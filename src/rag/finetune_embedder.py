"""Teach the retrieval embedder (bge-m3) Sudanese dialect, and measure retrieval before/after.

bge-m3 was trained mostly on MSA: Sudanese words like «داير» or «هسي» are split into pieces it never saw used,
so a Sudanese question can score low against the KB entry that answers it. Fine-tuning pairs student-style
queries with the KB record that answers them:
  * real dialect: the asker's own WhatsApp wording of each thread (anonymized threads.jsonl)
  * teacher paraphrases (optional file, data/processed/paraphrases.jsonl: {"id", "query", "style"})
Hard negatives come from the current index (top-ranked wrong records outside the gold's summary cluster).
Held-out eval records (eval.jsonl), and summaries containing them, are never trained on.

  python -m src.rag.finetune_embedder train --out $ZOLAK_MODELS/zolak-embed-v1
  python -m src.rag.finetune_embedder eval  --embed BAAI/bge-m3 --embed $ZOLAK_MODELS/zolak-embed-v1

eval builds a temporary index per embedder (same KB, same hybrid retrieval as the chatbot: top_entries) and
reports hit@1 / hit@2 / MRR@10 on: the eval questions as distilled (MSA-like), the askers' real wording,
teacher paraphrases (if any), plus how far off-topic questions stay from the KB (top similarity).
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import statistics as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "space"))
from retriever import HybridRetriever, doc_text  # noqa: E402

PROC = Path("data/processed")
GREETING = re.compile(r"^(?:\W*(?:شباب|يا شباب|ي شباب|الاخوة|الإخوة|الأخوة|اخوانا|السلام عليكم(?: ورحمة الله(?: وبركاته)?)?|"
                      r"سلام(?: عليكم)?|مساء الخير|صباح الخير|معليش(?: على الإزعاج)?|عندي استفسار(?: بسيط)?|"
                      r"عندي سؤال|استفسار|لو سمحتو|بعد إذنكم)\W*)+")
QUESTION_CUE = re.compile(r"[؟?]|(?<!\w)(?:شنو|كيف|كيفن|وين|متين|منو|ياتو|كم|هل|ممكن|بسأل|بسال|في زول|في حد|"
                          r"عندو فكرة|عندو معلومة|محتاج|داير|دايرة|عايز|عايزة)(?!\w)")
# questions the group never discussed (to check how far they stay from the KB)
OFFTOPIC = [
    "كيف أطبخ كسرة؟", "شنو أحسن طريقة أعمل بيها العصيدة؟", "منو كسب كاس العالم الفات؟", "الهلال والمريخ لعبو متين؟",
    "كيف أتعلم البرمجة بلغة بايثون؟", "شنو أحسن موبايل أشتريهو هسي؟", "الطقس في الخرطوم بكرة كيف؟",
    "كيف أعمل قهوة بالجبنة؟", "أحسن فيلم سوداني شفتو شنو؟", "كيف أنقص وزني بسرعة؟", "شنو عاصمة أستراليا؟",
    "كيف أصلح الغسالة لو ما بتلف؟", "اكتب لي قصيدة عن النيل", "كم سعر الدهب في السودان؟", "كيف أربي قطة؟",
    "شنو أحسن تمرين للضهر؟", "كيف أعمل فول بالزيت؟", "ترجم لي الجملة دي للإنجليزي: أنا جعان",
    "متين رمضان السنة الجاية؟", "شنو الفرق بين الآيفون والسامسونج؟", "كيف أزرع طماطم في البلكونة؟",
    "منو أحسن مغني سوداني؟", "كيف أحل معادلة من الدرجة التانية؟", "وين ألقى تذاكر حفلة في الخرطوم؟",
    "How do I bake bread at home?", "Wer hat die Fußball-Bundesliga gewonnen?", "شنو أحسن لعبة بلايستيشن؟",
    "كيف أعمل حساب في تيك توك؟", "الشاي باللبن بيتعمل كيف؟", "كيف أتعلم السواقة بسرعة؟",
]


def asker_wording(t: dict, max_chars: int = 400) -> str:
    """The asker's own question messages, greetings stripped."""
    parts = [m["text"] for m in t["messages"] if m["role"] == "السائل" and m["part"] != "reply"]
    text = GREETING.sub("", " ".join(parts)).strip()
    return text[:max_chars]


def load_json(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.open(encoding="utf-8")] if path.exists() else []


def query_sets(kb: list[dict], eval_ids: set[str]) -> dict[str, list[tuple[str, str]]]:
    """name -> [(query, gold record id)] for train / dev / eval sets."""
    threads = {t["thread_id"]: t for t in load_json(PROC / "threads.jsonl")}
    by_id = {r["id"]: r for r in kb}
    real = []
    for rid, r in by_id.items():
        if r.get("source") == "thread" and rid in threads:
            q = asker_wording(threads[rid])
            if len(q.split()) >= 4 and QUESTION_CUE.search(q):
                real.append((q, rid))
    para = [(p["query"], p["id"]) for p in load_json(PROC / "paraphrases.jsonl") if p["id"] in by_id]
    rnd = random.Random(0)
    train_ids = sorted({rid for _, rid in real if rid not in eval_ids})
    dev_ids = set(rnd.sample(train_ids, max(50, len(train_ids) // 10)))
    sets = {
        "train": [x for x in real + para if x[1] not in eval_ids and x[1] not in dev_ids],
        "dev_real": [x for x in real if x[1] in dev_ids],
        "eval_distilled": [(by_id[i]["question"], i) for i in sorted(eval_ids) if i in by_id],
        "eval_real": [x for x in real if x[1] in eval_ids],
        "eval_paraphrase": [x for x in para if x[1] in eval_ids],
    }
    return sets


def hits_gold(rec: dict, gold: str) -> bool:
    return rec["id"] == gold or gold in rec.get("members", [])


def evaluate(retr: HybridRetriever, queries: list[tuple[str, str]]) -> dict:
    h1 = h2 = rr = 0.0
    top_sims = []
    for q, gold in queries:
        top = retr.top_entries(q, k=2)
        h1 += bool(top) and hits_gold(top[0], gold)
        h2 += any(hits_gold(r, gold) for r in top)
        ranked, seen = [], set()
        hits = retr.search(q, k=20)
        top_sims.append(max(h.dense_sim for h in hits))
        for h in hits:
            rec = retr.summary_of.get(h.rec["id"], h.rec)
            if rec["id"] not in seen:
                seen.add(rec["id"])
                ranked.append(rec)
        rr += next((1 / (i + 1) for i, r in enumerate(ranked[:10]) if hits_gold(r, gold)), 0.0)
    n = max(len(queries), 1)
    return {"n": len(queries), "hit@1": round(h1 / n, 3), "hit@2": round(h2 / n, 3), "mrr@10": round(rr / n, 3),
            "top_sim_median": round(st.median(top_sims), 3) if top_sims else None}


def offtopic_sims(retr: HybridRetriever) -> list[float]:
    return [max(h.dense_sim for h in retr.search(q, k=5)) for q in OFFTOPIC]


def cmd_eval(args) -> None:
    from sentence_transformers import SentenceTransformer

    from src.rag.build_index import build
    kb_path = Path(args.kb)
    kb = load_json(kb_path)
    eval_ids = {r["id"] for r in load_json(PROC / args.eval)}
    sets = query_sets(kb, eval_ids)
    report = {}
    for emb in args.embed:
        out = Path(os.environ.get("ZOLAK_STORAGE", "/tmp")) / "tmp_index" / Path(emb).name
        model = build(kb_path, out, args.device, emb)
        retr = HybridRetriever(out, embed_model=model)
        res = {name: evaluate(retr, qs) for name, qs in sets.items() if name != "train" and qs}
        off = offtopic_sims(retr)
        res["offtopic_top_sim_median"] = round(st.median(off), 3)
        report[Path(emb).name] = res
        del model
    print(json.dumps(report, ensure_ascii=False, indent=1))
    if args.report:
        Path(args.report).write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")


def cmd_train(args) -> None:
    from datasets import Dataset
    from sentence_transformers import (SentenceTransformer, SentenceTransformerTrainer,
                                       SentenceTransformerTrainingArguments, losses)
    from sentence_transformers.training_args import BatchSamplers

    kb = load_json(Path(args.kb))
    by_id = {r["id"]: r for r in kb}
    eval_ids = {r["id"] for r in load_json(PROC / args.eval)}
    sets = query_sets(kb, eval_ids)
    retr = HybridRetriever(args.index, device="cuda")     # current index: hard-negative mining
    pos = {r["id"]: i for i, r in enumerate(retr.records)}
    qvecs = retr.qindex.reconstruct_n(0, retr.qindex.ntotal)   # stored question embeddings
    cluster_of = {m: s["id"] for s in kb if s.get("source") == "summary" for m in s.get("members", [])}
    hidden = eval_ids | {s["id"] for s in kb if s.get("source") == "summary" and eval_ids & set(s.get("members", []))}

    rows = []
    for q, gold in sets["train"]:
        g = by_id[gold]
        gq = qvecs[pos[gold]]
        neg = None
        for h in retr.search(q, k=12):
            r = h.rec
            if r["id"] in hidden or hits_gold(r, gold) or r["id"] == cluster_of.get(gold):
                continue
            if cluster_of.get(r["id"]) and cluster_of.get(r["id"]) == cluster_of.get(gold):
                continue
            if float(qvecs[pos[r["id"]]] @ gq) >= 0.85:   # near-duplicate of the gold: not a negative
                continue
            neg = r
            break
        if neg is None:
            continue
        # two positive views: the record's question (question index) and the whole record (record index)
        rows.append({"anchor": q, "positive": g["question"], "negative": neg["question"]})
        rows.append({"anchor": q, "positive": doc_text(g), "negative": doc_text(neg)})
    random.Random(0).shuffle(rows)
    print(f"{len(rows)} training rows from {len(sets['train'])} queries "
          f"({len(set(x[1] for x in sets['train']))} records); dev {len(sets['dev_real'])}")
    del retr

    model = SentenceTransformer(args.base, device="cuda")
    model.max_seq_length = 512
    loss = losses.CachedMultipleNegativesRankingLoss(model, mini_batch_size=32)
    targs = SentenceTransformerTrainingArguments(
        output_dir=str(Path(args.out) / "_run"), num_train_epochs=args.epochs, per_device_train_batch_size=args.batch,
        learning_rate=args.lr, warmup_ratio=0.1, bf16=True, batch_sampler=BatchSamplers.NO_DUPLICATES,
        logging_steps=5, save_strategy="no", report_to="none", seed=0)
    SentenceTransformerTrainer(model=model, args=targs, train_dataset=Dataset.from_list(rows), loss=loss).train()
    model.save(args.out)
    print(f"saved -> {args.out}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("train", "eval"):
        p = sub.add_parser(name)
        p.add_argument("--kb", default="data/processed/kb_group.jsonl")
        p.add_argument("--eval", default="eval.jsonl", help="file in data/processed with the held-out records")
        p.add_argument("--device", default="cuda")
    t = sub.choices["train"]
    t.add_argument("--index", default="data/index", help="current index, used to mine hard negatives")
    t.add_argument("--base", default="BAAI/bge-m3")
    t.add_argument("--out", required=True)
    t.add_argument("--epochs", type=float, default=2)
    t.add_argument("--batch", type=int, default=128)
    t.add_argument("--lr", type=float, default=1e-5)
    e = sub.choices["eval"]
    e.add_argument("--embed", action="append", required=True, help="embedder(s) to compare")
    e.add_argument("--report", default="")
    args = ap.parse_args()
    cmd_train(args) if args.cmd == "train" else cmd_eval(args)


if __name__ == "__main__":
    main()
