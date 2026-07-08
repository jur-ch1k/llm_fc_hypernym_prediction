"""
Гистограмма суммарной длины примеров SFT-датасета (Qwen3.5-2B chat template).

Для каждой записи считает число токенов после apply_chat_template
(с tools=get_hyponyms, как при обучении), затем выводит накопительную
гистограмму с заданным шагом: сколько примеров имеют длину <= порога.

Запуск:
    python scripts/dataset_token_length_histogram.py output/dataset/train_tsv_nohdr_full.jsonl
    python scripts/dataset_token_length_histogram.py output/dataset/train_tsv_nohdr_full.jsonl --step 500 -o lengths.txt
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import sys
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent
ROOT = SCRIPTS.parent
REPO_ROOT = ROOT.parent

# Только локальный HF-кэш в репозитории (не ~/.cache).
os.environ["HF_HOME"] = str(REPO_ROOT / ".cache" / "huggingface")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(SCRIPTS))

from analyze_dataset_tokens import load_records, resolve_output_path  # noqa: E402
from utils.tools import hyponym_only  # noqa: E402

DEFAULT_MODEL = "Qwen/Qwen3.5-4B"
DEFAULT_STEP = 500

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass


def normalize_messages_for_template(messages: list[dict]) -> list[dict]:
    """Qwen3.5 chat template ожидает arguments как dict, не JSON-строку."""
    msgs = copy.deepcopy(messages)
    for msg in msgs:
        if msg.get("role") != "assistant":
            continue
        for tc in msg.get("tool_calls") or []:
            fn = tc.get("function", tc)
            args = fn.get("arguments")
            if isinstance(args, str):
                fn["arguments"] = json.loads(args)
    return msgs


def load_qwen_tokenizer(model_id: str = DEFAULT_MODEL):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        trust_remote_code=True,
        local_files_only=True,
    )
    cache_dir = Path(os.environ["HF_HOME"])
    print(f"Tokenizer: {model_id} (HF_HOME={cache_dir})")
    return tokenizer


def count_record_tokens(tokenizer, messages: list[dict]) -> int:
    normalized = normalize_messages_for_template(messages)
    result = tokenizer.apply_chat_template(
        normalized,
        tools=hyponym_only,
        tokenize=True,
        return_dict=True,
        enable_thinking=False,
    )
    return len(result["input_ids"])


def cumulative_histogram(lengths: list[int], step: int) -> list[tuple[int, int]]:
    if not lengths:
        return [(0, 0)]

    max_len = max(lengths)
    max_bucket = max(step, math.ceil(max_len / step) * step)
    thresholds = list(range(0, max_bucket + step, step))
    return [(threshold, sum(1 for length in lengths if length <= threshold)) for threshold in thresholds]


def format_histogram_table(rows: list[tuple[int, int]], total: int) -> str:
    lines = ["число_токенов\tчисло_примеров\t% от общей доли"]
    for threshold, count in rows:
        pct = 100.0 * count / total if total else 0.0
        lines.append(f"<={threshold}\t{count}\t{pct:.1f}%")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Накопительная гистограмма длин примеров SFT-датасета (Qwen3.5-2B)",
    )
    parser.add_argument(
        "dataset",
        help="Путь к JSONL или JSON (одна запись / массив)",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"HF model id (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--step",
        type=int,
        default=DEFAULT_STEP,
        help=f"Шаг порогов в токенах (default: {DEFAULT_STEP})",
    )
    parser.add_argument(
        "--output",
        "-o",
        default=None,
        help="Записать отчёт в файл вместо stdout",
    )
    args = parser.parse_args()

    if args.step <= 0:
        raise SystemExit("--step должен быть > 0")

    dataset_path = Path(args.dataset)
    if not dataset_path.is_absolute():
        dataset_path = ROOT / dataset_path

    records = load_records(dataset_path)
    if not records:
        print("Датасет пуст.")
        sys.exit(1)

    tokenizer = load_qwen_tokenizer(args.model)

    lengths: list[int] = []
    for i, record in enumerate(records):
        messages = record.get("messages")
        if not messages:
            lengths.append(0)
            continue
        lengths.append(count_record_tokens(tokenizer, messages))
        if (i + 1) % 100 == 0 or i + 1 == len(records):
            print(f"Обработано {i + 1}/{len(records)}", file=sys.stderr)

    histogram = cumulative_histogram(lengths, args.step)
    table = format_histogram_table(histogram, len(lengths))

    summary_lines = [
        f"Записей: {len(lengths)}",
        f"min={min(lengths)} max={max(lengths)} mean={sum(lengths) / len(lengths):.1f}",
        f"Шаг: {args.step}",
        "",
        table,
    ]
    report = "\n".join(summary_lines)

    if args.output:
        out_path = resolve_output_path(args.output)
        out_path.write_text(report + "\n", encoding="utf-8")
        print(f"Отчёт записан в {out_path}")
    else:
        print(report)


if __name__ == "__main__":
    main()
