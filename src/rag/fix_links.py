"""Repair and validate the links stored in the knowledge base.

1. Repair: an old normalization step squeezed 3+ repeated letters, so "www." became "ww." and "www2." became
   "ww2.". Every host that starts with "ww" + "." or a digit is restored (the squeeze always left exactly 2).
   The repair is applied to the record files and to the text of threads/messages/posts (used by the KB audit).
2. Validate: GET every unique link (redirects followed, 2 tries). On a certificate host mismatch the bare host
   (without "www.") is tried. Links that answer 404/410 or whose host does not exist are dropped; 403 and TLS
   refusals (bot blocking) are kept.

Report: $ZOLAK_STORAGE/logs/links.json
Usage:  python3 -m src.rag.fix_links [--no-check]
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import os
import re
import socket
import ssl
import time
import urllib.error
import urllib.request
from pathlib import Path

RECORD_FILES = ["kb.jsonl", "qa.jsonl", "eval.jsonl", "kb_summaries.jsonl", "kb_group.jsonl", "kb_raw.jsonl"]
TEXT_FILES = ["threads.jsonl", "messages.jsonl", "info_posts.jsonl"]
SQUEEZED = re.compile(r"(?<![\w.])(https?://)?ww(?=\.|\d)")   # ww.x / ww2.x  ->  www.x / www2.x


def repair(text: str) -> str:
    return SQUEEZED.sub(lambda m: (m.group(1) or "") + "www", text)


def fetch(url: str) -> int | str:
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (link check)"})
    try:
        with urllib.request.urlopen(req, timeout=20, context=ssl.create_default_context()) as r:
            return r.status
    except urllib.error.HTTPError as e:
        return e.code
    except urllib.error.URLError as e:
        r = e.reason
        if isinstance(r, socket.gaierror):
            return "dns"
        if isinstance(r, ssl.SSLCertVerificationError) and "mismatch" in str(r).lower():
            return "cert-mismatch"
        return type(r).__name__
    except Exception as e:  # timeouts, resets
        return type(e).__name__


def check(url: str) -> tuple[str, str | None, int | str]:
    """-> (url, replacement or None to drop, status)"""
    status = fetch(url)
    if status not in (200, 403) and not isinstance(status, int) and status != "dns":
        time.sleep(2)
        status = fetch(url)
    if status == "cert-mismatch" and "://www." in url:
        alt = url.replace("://www.", "://", 1)
        s2 = fetch(alt)
        return (url, alt, s2) if s2 == 200 else (url, url, status)
    if status in (404, 410, "dns"):
        return url, None, status
    return url, url, status


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--proc", default="data/processed")
    ap.add_argument("--no-check", action="store_true", help="only repair, no network check")
    args = ap.parse_args()
    proc = Path(args.proc)

    # 1. textual repair of every file (record files are rewritten again after validation)
    for name in RECORD_FILES + TEXT_FILES:
        p = proc / name
        if p.exists():
            raw = p.read_text(encoding="utf-8")
            fixed = repair(raw)
            if fixed != raw:
                p.write_text(fixed, encoding="utf-8")
                print(f"repaired {name}: {len(SQUEEZED.findall(raw))} links")

    files = {n: [json.loads(l) for l in (proc / n).open(encoding="utf-8")] for n in RECORD_FILES if (proc / n).exists()}
    urls = sorted({u for rows in files.values() for r in rows for u in r.get("links", [])})
    if args.no_check:
        return
    with cf.ThreadPoolExecutor(16) as ex:
        results = list(ex.map(check, urls))
    mapping = {u: rep for u, rep, _ in results}
    report = [{"url": u, "status": s, "action": "drop" if rep is None else ("keep" if rep == u else f"-> {rep}")}
              for u, rep, s in results]
    for name, rows in files.items():
        for r in rows:
            if r.get("links"):
                r["links"] = list(dict.fromkeys(mapping[u] for u in r["links"] if mapping.get(u)))
        with (proc / name).open("w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    logs = Path(os.environ.get("ZOLAK_STORAGE", ".")) / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    (logs / "links.json").write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    dropped = [x for x in report if x["action"] == "drop"]
    moved = [x for x in report if x["action"].startswith("->")]
    print(f"{len(urls)} unique links: {len(urls) - len(dropped)} kept ({len(moved)} rewritten), {len(dropped)} dropped")
    for x in dropped + moved:
        print(f"  {x['status']}  {x['action']:>6.6}  {x['url'][:110]}")


if __name__ == "__main__":
    main()
