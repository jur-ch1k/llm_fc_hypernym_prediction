"""Инференс базовой модели (без LoRA) с agent loop и TSV-тулзами.

Запуск без аргументов:
    python run_inference_base.py

С указанием начальной строки (0-based, включительно):
    python run_inference_base.py --from 50

Всегда сохраняет полный .md-лог (--verbose) и поле final_answer (--add-final).
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
from datetime import datetime
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPTS.parent.parent
REPO_ROOT = PROJECT_ROOT.parent
HF_CACHE_ROOT = REPO_ROOT / ".cache" / "huggingface"
HF_HUB_CACHE = HF_CACHE_ROOT / "hub"

# Только локальный HF-кэш в репозитории (не ~/.cache на диске C:).
# os.environ["HF_HOME"] = str(HF_CACHE_ROOT)
# os.environ["HF_HUB_CACHE"] = str(HF_HUB_CACHE)

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(SCRIPTS))

from sft_pipeline.config import CORPUS_DIR, WORDNET_PATH  # noqa: E402
from sft_pipeline.data_loaders import iter_dataset_rows, load_context  # noqa: E402
from taxoenrich.core import RuWordNet  # noqa: E402
from run_inference import (  # noqa: E402
    extract_final_answer,
    run_agent_loop,
    save_result,
    save_verbose_dialog,
)

BASE_MODEL_ID = "Qwen/Qwen3.5-4B"
SEED_DIR = "datasets/seed_42"
MAX_ITERS = 15
TEMPERATURE = 0.0
OUTPUT_ROOT = PROJECT_ROOT / "output" / "inference"


def load_base_model_and_tokenizer(model_id: str):
    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        trust_remote_code=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    return model, tokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Инференс базовой модели (без LoRA)")
    parser.add_argument(
        "--from",
        dest="from_index",
        type=int,
        default=0,
        metavar="N",
        help="Начать с строки N датасета (0-based, включительно) и обработать до конца",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.from_index < 0:
        raise ValueError(f"--from должен быть >= 0, получено {args.from_index}")

    dataset_path = PROJECT_ROOT / SEED_DIR / "test_dataset.tsv"
    if not dataset_path.exists():
        raise FileNotFoundError(f"Test dataset не найден: {dataset_path}")

    run_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = OUTPUT_ROOT / f"run_base_{run_ts}"
    run_dir.mkdir(parents=True, exist_ok=True)

    print(f"HF cache: {HF_HUB_CACHE}")
    print(f"Loading base model (no adapter): {BASE_MODEL_ID}")
    model, tokenizer = load_base_model_and_tokenizer(BASE_MODEL_ID)

    wordnet_path = str(PROJECT_ROOT / WORDNET_PATH)
    wn = RuWordNet(wordnet_path)
    all_rows = iter_dataset_rows(None, str(dataset_path))
    rows = [row for row in all_rows if row.row_index >= args.from_index]
    if not rows:
        raise ValueError(
            f"Нет строк для инференса: --from {args.from_index}, "
            f"в датасете {len(all_rows)} строк (индексы 0..{len(all_rows) - 1})"
        )
    print(f"Dataset rows: {len(rows)} (from index {args.from_index}, total {len(all_rows)})")

    run_meta = {
        "base_only": True,
        "base_model": BASE_MODEL_ID,
        "from_index": args.from_index,
        "limit": None,
        "max_iters": MAX_ITERS,
        "temperature": TEMPERATURE,
        "seed_dir": str(dataset_path),
        "verbose": True,
        "add_final": True,
        "timestamp": datetime.now().isoformat(),
        "run_dir": str(run_dir),
    }
    with open(run_dir / "run_meta.json", "w", encoding="utf-8") as f:
        json.dump(run_meta, f, ensure_ascii=False, indent=2)

    ok_count = 0
    failed_count = 0

    for row in rows:
        context_file = row.context_files[0]
        context_text = load_context(context_file, str(PROJECT_ROOT / CORPUS_DIR))
        word = row.word
        print(f"[{row.row_index}] {word}")

        out_stem = f"{row.row_index}_{word}"

        if not context_text:
            payload = {
                "target_word": word,
                "target_ids": row.target_ids,
                "selected_synsets": [],
                "final_result": None,
                "status": "failed",
                "iterations": 0,
                "error": f"Контекст не загружен: {context_file}",
                "final_answer": None,
            }
            failed_count += 1
            save_verbose_dialog(
                run_dir / f"{out_stem}.md",
                word,
                row.row_index,
                [
                    {
                        "role": "system",
                        "content": f"Ошибка: контекст не загружен ({context_file})",
                    }
                ],
                status="failed",
                final_result=None,
            )
        else:
            loop_result = run_agent_loop(
                model,
                tokenizer,
                wn,
                context_text,
                max_iters=MAX_ITERS,
                temperature=TEMPERATURE,
            )
            payload = {
                "target_word": word,
                "target_ids": row.target_ids,
                "selected_synsets": loop_result["selected_synsets"],
                "final_result": loop_result["final_result"],
                "status": loop_result["status"],
                "iterations": loop_result["iterations"],
                "total_selections": loop_result["total_selections"],
                "final_answer": extract_final_answer(loop_result["messages"]),
            }
            if loop_result["status"] == "ok":
                ok_count += 1
            else:
                failed_count += 1
            save_verbose_dialog(
                run_dir / f"{out_stem}.md",
                word,
                row.row_index,
                loop_result["messages"],
                status=loop_result["status"],
                final_result=loop_result["final_result"],
            )

        save_result(run_dir / f"{out_stem}.json", payload)

    run_meta["ok"] = ok_count
    run_meta["failed"] = failed_count
    with open(run_dir / "run_meta.json", "w", encoding="utf-8") as f:
        json.dump(run_meta, f, ensure_ascii=False, indent=2)

    print(f"Done: {run_dir} (ok={ok_count}, failed={failed_count})")

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
