"""Step 3 — reconstruct question threads and collect long informative posts.

Thread = a question anchor + its context:
  * prelude: the asker's own messages in the 10 min before the question (questions split over lines)
  * follow-ups: subsequent kept messages up to `--window-h` hours / `--max-msgs` messages,
    stopping early after `--gap-h` hours of silence
  * consecutive questions by the same asker within 10 min are merged into one thread
Participants are relabelled per thread (السائل, عضو1, عضو2, ...) — no speaker ids reach the LLM.
Threads where nobody else replied are dropped (no knowledge to extract).

Info posts = kept non-question messages with >= `--info-min-words` content words
(experience write-ups, embassy notices, guides). They are distilled separately.

Input : data/processed/messages.jsonl  (keep == True)
Output: data/processed/threads.jsonl, data/processed/info_posts.jsonl

Usage: python -m src.ingest.threads
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timedelta
from pathlib import Path
from statistics import mean, median

from .filter import content_words

ASKER = "السائل"


def _ts(m: dict) -> datetime:
    return datetime.fromisoformat(m["ts"])


def build_threads(msgs: list[dict], window_h: float, gap_h: float, max_msgs: int) -> list[dict]:
    kept = [m for m in msgs if m["keep"]]
    threads: list[dict] = []
    consumed_anchor: set[int] = set()
    for i, q in enumerate(kept):
        if not q["is_question"] or q["id"] in consumed_anchor:
            continue
        asker, t0 = q["speaker"], _ts(q)

        # prelude: asker's lines in the 10 minutes before
        prelude = []
        j = i - 1
        while j >= 0 and kept[j]["speaker"] == asker and t0 - _ts(kept[j]) <= timedelta(minutes=10):
            prelude.insert(0, kept[j])
            j -= 1

        # merge immediate follow-up questions by the same asker
        anchors = [q]
        k = i + 1
        while k < len(kept) and kept[k]["speaker"] == asker and _ts(kept[k]) - _ts(anchors[-1]) <= timedelta(minutes=10):
            anchors.append(kept[k])
            if kept[k]["is_question"]:
                consumed_anchor.add(kept[k]["id"])
            k += 1

        # follow-up window
        follow, last = [], _ts(anchors[-1])
        while k < len(kept) and len(follow) < max_msgs:
            m = kept[k]
            t = _ts(m)
            if t - t0 > timedelta(hours=window_h) or t - last > timedelta(hours=gap_h):
                break
            follow.append(m)
            last = t
            k += 1

        others = [m for m in follow if m["speaker"] != asker]
        if not others:
            continue

        roles: dict[str, str] = {asker: ASKER}

        def role(spk: str) -> str:
            if spk not in roles:
                roles[spk] = f"عضو{len(roles)}"
            return roles[spk]

        def item(m: dict, part: str) -> dict:
            return {
                "id": m["id"], "role": role(m["speaker"]), "part": part, "ts": m["ts"], "text": m["text"],
                "mentions_asker": asker in m.get("mentions", []),
            }

        context = [item(m, "prelude") for m in prelude] + [item(m, "question") for m in anchors] \
            + [item(m, "reply") for m in follow]
        threads.append({
            "thread_id": f"t{q['id']}",
            "anchor_id": q["id"],
            "date": q["ts"][:10],
            "question": "\n".join(m["text"] for m in prelude + anchors),
            "messages": context,
            "n_replies": len(others),
            "n_repliers": len({m["speaker"] for m in others}),
            "n_mentions_asker": sum(asker in m.get("mentions", []) for m in others),
        })
    return threads


def info_posts(msgs: list[dict], min_words: int) -> list[dict]:
    return [
        {"post_id": f"p{m['id']}", "id": m["id"], "date": m["ts"][:10], "text": m["text"],
         "n_words": len(content_words(m["text"]))}
        for m in msgs
        if m["keep"] and not m["is_question"] and len(content_words(m["text"])) >= min_words
    ]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", default="data/processed/messages.jsonl")
    ap.add_argument("--out-dir", default="data/processed")
    ap.add_argument("--window-h", type=float, default=48)
    ap.add_argument("--gap-h", type=float, default=12)
    ap.add_argument("--max-msgs", type=int, default=40)
    ap.add_argument("--info-min-words", type=int, default=60)
    args = ap.parse_args()

    msgs = [json.loads(l) for l in open(args.input, encoding="utf-8")]
    threads = build_threads(msgs, args.window_h, args.gap_h, args.max_msgs)
    posts = info_posts(msgs, args.info_min_words)

    out = Path(args.out_dir)
    for name, rows in (("threads.jsonl", threads), ("info_posts.jsonl", posts)):
        with (out / name).open("w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    n_q = sum(m["keep"] and m["is_question"] for m in msgs)
    lens = [len(t["messages"]) for t in threads]
    chars = [sum(len(x["text"]) for x in t["messages"]) for t in threads]
    print(f"questions: {n_q} -> threads with replies: {len(threads)}")
    print(f"messages/thread mean {mean(lens):.1f} median {median(lens)}; "
          f"chars/thread median {median(chars)} p95 {sorted(chars)[int(len(chars) * .95)]}")
    print(f"threads where someone @mentioned the asker: {sum(t['n_mentions_asker'] > 0 for t in threads)}")
    print(f"info posts (>= {args.info_min_words} words): {len(posts)}")


if __name__ == "__main__":
    main()
