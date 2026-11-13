"""Deploy to Hugging Face (run on this server after `hf auth login` with a WRITE token).

Creates (if missing) and uploads:
  <user>/sudachat-models     PRIVATE model repo   <- GGUF model file
  <user>/sudachat-data       PRIVATE dataset      <- index/ (kb.faiss, kb.jsonl, config.json)
  <user>/sudachat-feedback   PRIVATE dataset      <- written by the Space (👍/👎)
  <user>/sudachat            PUBLIC Gradio Space  <- space/ (code only), with secret HF_TOKEN + variables

Usage:
  source scripts/env.sh
  python3 scripts/deploy_hf.py --step model   # upload GGUF (4-5 GB, once per model version)
  python3 scripts/deploy_hf.py --step data    # upload index/ (after every KB rebuild; leak-tested first)
  python3 scripts/deploy_hf.py --step space   # create/update the Space code + settings
  python3 scripts/deploy_hf.py --step all
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

from huggingface_hub import HfApi, get_token

ROOT = Path(__file__).resolve().parents[1]
STORAGE = Path(os.environ.get("SUDACHAT_STORAGE", f"/temp/{os.environ.get('USER', 'user')}/sudaChat"))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--step", choices=["model", "data", "space", "all"], required=True)
    ap.add_argument("--model-file", default=str(STORAGE / "models/gguf/ALLaM-7B-Instruct-preview-Q4_K_M.gguf"))
    ap.add_argument("--index-dir", default=str(ROOT / "data/index"))
    ap.add_argument("--space-name", default="sudachat")
    ap.add_argument("--wheel-url", default="", help="public URL of the llama-cpp-python wheel (GitHub release)")
    args = ap.parse_args()

    token = get_token()
    if not token:
        sys.exit("Not logged in: run `hf auth login` with a WRITE token first.")
    api = HfApi(token=token)
    user = api.whoami()["name"]
    model_repo, data_repo, fb_repo = f"{user}/sudachat-models", f"{user}/sudachat-data", f"{user}/sudachat-feedback"
    space_repo = f"{user}/{args.space_name}"
    steps = {"model", "data", "space"} if args.step == "all" else {args.step}

    if "model" in steps:
        api.create_repo(model_repo, repo_type="model", private=True, exist_ok=True)
        mf = Path(args.model_file)
        print(f"uploading {mf.name} ({mf.stat().st_size / 1e9:.1f} GB) -> {model_repo} (private)")
        api.upload_file(path_or_fileobj=str(mf), path_in_repo=mf.name, repo_id=model_repo, repo_type="model")

    if "data" in steps:
        # privacy gate: never upload a KB that fails the leak test
        r = subprocess.run([sys.executable, "-m", "pytest", "-q", "tests/test_pii_leak.py"], cwd=ROOT)
        if r.returncode != 0:
            sys.exit("leak test failed — not uploading the KB")
        api.create_repo(data_repo, repo_type="dataset", private=True, exist_ok=True)
        print(f"uploading {args.index_dir} -> {data_repo}/index (private)")
        api.upload_folder(folder_path=args.index_dir, path_in_repo="index", repo_id=data_repo, repo_type="dataset",
                          allow_patterns=["kb.faiss", "kb.jsonl", "config.json"])

    if "space" in steps:
        req = (ROOT / "space/requirements.txt").read_text()
        if "__WHEEL_URL__" in req:
            if not args.wheel_url:
                sys.exit("space/requirements.txt still has __WHEEL_URL__: pass --wheel-url (GitHub release asset)")
            req = req.replace("__WHEEL_URL__", args.wheel_url)
        api.create_repo(fb_repo, repo_type="dataset", private=True, exist_ok=True)
        api.create_repo(space_repo, repo_type="space", space_sdk="gradio", private=False, exist_ok=True)
        api.add_space_secret(space_repo, "HF_TOKEN", token)
        for k, v in {"MODEL_REPO": model_repo, "MODEL_FILE": Path(args.model_file).name,
                     "DATA_REPO": data_repo, "FEEDBACK_REPO": fb_repo}.items():
            api.add_space_variable(space_repo, k, v)
        api.upload_folder(folder_path=str(ROOT / "space"), repo_id=space_repo, repo_type="space",
                          ignore_patterns=["wheels/*", "__pycache__/*", "feedback/*", "requirements.txt"])
        api.upload_file(path_or_fileobj=req.encode(), path_in_repo="requirements.txt", repo_id=space_repo,
                        repo_type="space")
        print(f"Space: https://huggingface.co/spaces/{space_repo}  (API: {space_repo})")


if __name__ == "__main__":
    main()
