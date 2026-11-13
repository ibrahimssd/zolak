"""Step 7b — LoRA fine-tune of the student (ALLaM-7B-Instruct) on the Zolna SFT set, then merge.

Loss is computed on the assistant answer only (prompt-completion format). bf16, gradient checkpointing,
one GPU is enough for a 7B model with LoRA.

Usage (training venv):
    source scripts/env.sh
    CUDA_VISIBLE_DEVICES=6 $ZOLNA_VENVS/train/bin/python -m src.train.sft_lora --run v1
    CUDA_VISIBLE_DEVICES=6 $ZOLNA_VENVS/train/bin/python -m src.train.sft_lora --run v1 --merge-only
Outputs: $ZOLNA_CHECKPOINTS/zolna-allam-lora-<run>/ (adapter) and $ZOLNA_MODELS/zolna-allam-<run>/ (merged, bf16)
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

BASE = "humain-ai/ALLaM-7B-Instruct-preview"


def load_rows(path: Path) -> list[dict]:
    rows = []
    for line in path.open(encoding="utf-8"):
        r = json.loads(line)
        rows.append({"prompt": r["prompt"], "completion": r["completion"]})
    return rows


def train(args) -> Path:
    import torch
    from datasets import Dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import SFTConfig, SFTTrainer

    tok = AutoTokenizer.from_pretrained(BASE)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(BASE, torch_dtype=torch.bfloat16, attn_implementation="sdpa")

    proc = Path(args.proc)
    train_ds = Dataset.from_list(load_rows(proc / "sft_train.jsonl"))
    val_ds = Dataset.from_list(load_rows(proc / "sft_val.jsonl"))

    out = Path(os.environ["ZOLNA_CHECKPOINTS"]) / f"zolna-allam-lora-{args.run}"
    cfg = SFTConfig(
        output_dir=str(out),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch,
        per_device_eval_batch_size=args.batch,
        gradient_accumulation_steps=args.accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        weight_decay=0.0,
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        max_length=args.max_len,
        completion_only_loss=True,       # learn the answer, not the prompt
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        report_to="none",
        seed=0,
    )
    lora = LoraConfig(r=args.rank, lora_alpha=2 * args.rank, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
                      target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
    trainer = SFTTrainer(model=model, args=cfg, train_dataset=train_ds, eval_dataset=val_ds,
                         processing_class=tok, peft_config=lora)
    trainer.model.print_trainable_parameters()
    trainer.train()
    final = out / "final"
    trainer.save_model(str(final))
    tok.save_pretrained(str(final))
    metrics = trainer.evaluate()
    (out / "eval_metrics.json").write_text(json.dumps(metrics, indent=1))
    print("eval:", metrics)
    return final


def merge(adapter: Path, run: str) -> Path:
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    base = AutoModelForCausalLM.from_pretrained(BASE, torch_dtype=torch.bfloat16)
    merged = PeftModel.from_pretrained(base, str(adapter)).merge_and_unload()
    out = Path(os.environ["ZOLNA_MODELS"]) / f"zolna-allam-{run}"
    merged.save_pretrained(str(out), safe_serialization=True)
    AutoTokenizer.from_pretrained(BASE).save_pretrained(str(out))
    print("merged model ->", out)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--proc", default="data/processed")
    ap.add_argument("--run", default="v1")
    ap.add_argument("--epochs", type=float, default=2)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--accum", type=int, default=4)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--max-len", type=int, default=2048)
    ap.add_argument("--merge-only", action="store_true")
    args = ap.parse_args()

    adapter = Path(os.environ["ZOLNA_CHECKPOINTS"]) / f"zolna-allam-lora-{args.run}" / "final"
    if not args.merge_only:
        adapter = train(args)
    merge(adapter, args.run)


if __name__ == "__main__":
    main()
