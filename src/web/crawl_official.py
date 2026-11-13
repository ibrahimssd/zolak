"""Crawl a small set of official German sites for topics Sudanese students ask about, once (refresh monthly).

Polite: honours robots.txt, 1 request/second, topic-filtered links only, a page cap per site.
Output (local, gitignored):
  data/official/pages.jsonl   {"url", "title", "site", "fetched", "text"}
  data/official/chunks.jsonl  {"chunk_id", "url", "title", "site", "fetched", "text"}  (~1,800-char sections)

Usage: python3 -m src.web.crawl_official [--per-site 35]
"""
from __future__ import annotations

import argparse
import json
import re
import time
from collections import deque
from datetime import date
from pathlib import Path
from urllib import robotparser
from urllib.parse import urldefrag, urljoin, urlparse

import requests
import trafilatura
from bs4 import BeautifulSoup

UA = "ZolakBot/1.0 (community helper for Sudanese students; contact via github.com/ibrahimssd/zolak)"

SITES = {  # site -> (seed URLs, allowed path prefix)
    "study-in-germany.de": (["https://www.study-in-germany.de/en/"], "/en/"),
    "make-it-in-germany.com": (["https://www.make-it-in-germany.com/en/"], "/en/"),
    "auswaertiges-amt.de": (["https://www.auswaertiges-amt.de/en/visa-service"], "/en/"),
    "daad.de": (["https://www.daad.de/en/study-and-research-in-germany/"], "/en/"),
    "anerkennung-in-deutschland.de": (["https://www.anerkennung-in-deutschland.de/html/en/index.php"], "/html/en/"),
    "bamf.de": (["https://www.bamf.de/EN/Startseite/startseite_node.html"], "/EN/"),
}

TOPIC = re.compile(
    r"blocked|sperrkonto|financ|finanz|cost|visa|visum|entry|residence|aufenthalt|permit|regist|anmeld|"
    r"health|insurance|kranken|work|job|employ|student|studies|study|admission|apply|application|bewerb|"
    r"recogni|anerkenn|degree|language|german-course|studienkolleg|preparatory|uni-assist|aps|accommodation|"
    r"housing|family|reunif|opportunity|chancenkarte|blue-card|graduat|after-graduation|living", re.I)
SKIP = re.compile(r"\.(pdf|jpg|png|zip|docx?|xlsx?)$|/(press|presse|newsroom|events?|jobs-at|career|login|search)\b", re.I)


def main_text(html: str) -> tuple[str, str]:
    text = trafilatura.extract(html, include_links=False, include_tables=True, favor_precision=True) or ""
    title = (BeautifulSoup(html, "lxml").title or "").get_text(strip=True) if "<title" in html else ""
    return title, text.strip()


def chunks(text: str, size: int = 1800) -> list[str]:
    paras = [p.strip() for p in re.split(r"\n{1,}", text) if len(p.strip()) > 40]
    out, cur = [], ""
    for p in paras:
        if len(cur) + len(p) > size and cur:
            out.append(cur.strip()); cur = ""
        cur += p + "\n"
    if len(cur.strip()) > 200:
        out.append(cur.strip())
    return out


def crawl_site(site: str, seeds: list[str], prefix: str, cap: int, session: requests.Session) -> list[dict]:
    rp = robotparser.RobotFileParser()
    try:
        rp.set_url(f"https://www.{site}/robots.txt"); rp.read()
    except Exception:
        rp = None
    seen, pages, queue = set(), [], deque((s, 0) for s in seeds)
    while queue and len(pages) < cap:
        url, depth = queue.popleft()
        url = urldefrag(url)[0]
        if url in seen or (rp and not rp.can_fetch(UA, url)):
            continue
        seen.add(url)
        try:
            r = session.get(url, timeout=20)
            time.sleep(1.0)
        except requests.RequestException as e:
            print(f"  ! {url}: {type(e).__name__}")
            continue
        if r.status_code != 200 or "text/html" not in r.headers.get("content-type", ""):
            continue
        title, text = main_text(r.text)
        if len(text) >= 500 and (depth == 0 or TOPIC.search(url + " " + title)):
            pages.append({"url": url, "title": title, "site": site, "fetched": date.today().isoformat(), "text": text})
        if depth < 3:
            soup = BeautifulSoup(r.text, "lxml")
            for a in soup.find_all("a", href=True):
                nxt = urldefrag(urljoin(url, a["href"]))[0]
                u = urlparse(nxt)
                if u.netloc.endswith(site) and u.path.startswith(prefix) and not SKIP.search(u.path) \
                        and nxt not in seen and TOPIC.search(nxt + " " + a.get_text(" ", strip=True)):
                    queue.append((nxt, depth + 1))
    return pages


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="data/official")
    ap.add_argument("--per-site", type=int, default=35)
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    s = requests.Session(); s.headers["User-Agent"] = UA
    all_pages = []
    for site, (seeds, prefix) in SITES.items():
        pages = crawl_site(site, seeds, prefix, args.per_site, s)
        print(f"{site}: {len(pages)} pages")
        all_pages += pages
    n = 0
    with (out / "pages.jsonl").open("w", encoding="utf-8") as fp, (out / "chunks.jsonl").open("w", encoding="utf-8") as fc:
        for i, p in enumerate(all_pages):
            fp.write(json.dumps(p, ensure_ascii=False) + "\n")
            for j, c in enumerate(chunks(p["text"])):
                fc.write(json.dumps({"chunk_id": f"o{i}-{j}", **{k: p[k] for k in ("url", "title", "site", "fetched")},
                                     "text": c}, ensure_ascii=False) + "\n")
                n += 1
    print(f"{len(all_pages)} pages -> {n} chunks in {out}")


if __name__ == "__main__":
    main()
