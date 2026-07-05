"""
Построение trajectory JSON для строк TSV датасета.

Запуск:
    python scripts/build_trajectories.py --row-index 0
    python scripts/build_trajectories.py --word АБСЕНТЕИЗМ
    python scripts/build_trajectories.py --limit 10
"""

import argparse
import logging
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from taxoenrich.core import RuWordNet

from sft_pipeline.config import MAX_HYPONYMS, MAX_WORDS, OUTPUT_TRAJECTORIES, WORDNET_PATH
from sft_pipeline.data_loaders import (
    iter_dataset_rows,
    iter_rows_by_word,
    load_dataset_row,
)
from sft_pipeline.trajectory import build_trajectories_for_row

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)


def _resolve_rows(args: argparse.Namespace) -> list:
    if args.row_index is not None:
        return [load_dataset_row(args.row_index)]
    if args.word:
        return iter_rows_by_word(args.word)
    if args.limit is not None:
        return iter_dataset_rows(args.limit)
    raise SystemExit("Укажите --row-index, --word или --limit")


def main():
    parser = argparse.ArgumentParser(description="Построить trajectory JSON")
    parser.add_argument("--row-index", type=int, help="0-based номер строки TSV")
    parser.add_argument("--word", help="Все строки TSV с этим словом")
    parser.add_argument("--limit", type=int, help="Первые N строк TSV")
    parser.add_argument("--out", default=OUTPUT_TRAJECTORIES, help="Директория для JSON")
    parser.add_argument(
        "--max-words",
        type=int,
        default=MAX_WORDS,
        help="Макс. число слов синсета в ответе тулзы (дефолт: %(default)s)",
    )
    parser.add_argument(
        "--max-hyponyms",
        type=int,
        default=MAX_HYPONYMS,
        help="Макс. число гипонимов/гиперонимов в ответе тулзы (дефолт: %(default)s)",
    )
    args = parser.parse_args()

    rows = _resolve_rows(args)
    if not rows:
        log.error("Строки не найдены")
        sys.exit(1)

    log.info("Загрузка RuWordNet...")
    wn = RuWordNet(WORDNET_PATH)
    log.info("Synsets: %d", len(wn.synsets))

    total_ok = 0
    total_no_path = 0
    total_invalid = 0

    for row in rows:
        log.info(
            "row=%d word=%s targets=%d context=%s",
            row.row_index,
            row.word,
            len(row.target_ids),
            row.context_files[0],
        )
        ok, no_path, invalid = build_trajectories_for_row(
            wn,
            row,
            args.out,
            max_words=args.max_words,
            max_hyponyms=args.max_hyponyms,
        )
        total_ok += ok
        total_no_path += no_path
        total_invalid += invalid

    log.info(
        "Готово: ok=%d, no_path=%d, invalid_target=%d",
        total_ok,
        total_no_path,
        total_invalid,
    )


if __name__ == "__main__":
    main()
