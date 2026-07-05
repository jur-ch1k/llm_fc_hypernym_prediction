"""
Сборка JSONL-датасета: trajectories + SFT messages.

Запуск:
    python scripts/build_dataset.py --row-index 0
    python scripts/build_dataset.py --word АБСЕНТЕИЗМ
    python scripts/build_dataset.py --limit 10
    python scripts/build_dataset.py --word АБСЕНТЕИЗМ --from-cache
    python scripts/build_dataset.py --dataset datasets/seed_42/train_dataset.tsv
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path

from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from taxoenrich.core import RuWordNet

from sft_pipeline.config import (
    DATASET_PATH,
    MAX_HYPONYMS,
    MAX_WORDS,
    OUTPUT_DATASET,
    OUTPUT_TRAJECTORIES,
    WORDNET_PATH,
)
from sft_pipeline.data_loaders import (
    DatasetRow,
    iter_dataset_rows,
    iter_rows_by_word,
    load_context,
    load_dataset_row,
)
from sft_pipeline.messages import messages_to_jsonl_record, trajectory_to_messages
from sft_pipeline.trajectory import (
    build_trajectories_for_row,
    load_trajectory,
    trajectory_filename,
)

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)

_WORD_FIELD_WIDTH = 32
_PROGRESS_BAR_FORMAT = (
    "{desc}: {percentage:3.0f}%|{bar:32}| {n_fmt}/{total_fmt} "
    "[{elapsed}<{remaining}, {rate_fmt}] {postfix}"
)


def _format_current_word(word: str) -> str:
    if len(word) > _WORD_FIELD_WIDTH:
        return word[: _WORD_FIELD_WIDTH - 1] + "…"
    return word.ljust(_WORD_FIELD_WIDTH)


def _resolve_rows(args: argparse.Namespace) -> list[DatasetRow]:
    dataset_path = args.dataset
    if args.row_index is not None:
        return [load_dataset_row(args.row_index, dataset_path)]
    if args.word:
        return iter_rows_by_word(args.word, dataset_path)
    if args.limit is not None:
        return iter_dataset_rows(args.limit, dataset_path)
    return iter_dataset_rows(None, dataset_path)


def _trajectory_path(traj_dir: Path, row: DatasetRow, target_id: str) -> Path:
    return traj_dir / trajectory_filename(row.word, row.row_index, target_id)


def _process_row(
    wn: RuWordNet,
    row: DatasetRow,
    traj_dir: Path,
    from_cache: bool,
    *,
    max_words: int = MAX_WORDS,
    max_hyponyms: int = MAX_HYPONYMS,
) -> list[dict]:
    if not from_cache:
        build_trajectories_for_row(
            wn,
            row,
            traj_dir,
            max_words=max_words,
            max_hyponyms=max_hyponyms,
        )

    records: list[dict] = []
    for target_id in row.target_ids:
        traj_path = _trajectory_path(traj_dir, row, target_id)
        if not traj_path.exists():
            tqdm.write(f"Пропуск: нет trajectory {traj_path.name}")
            continue

        traj = load_trajectory(traj_path)
        if traj.status != "ok":
            continue

        context_text = load_context(traj.context_file)
        if not context_text:
            tqdm.write(f"Пропуск {traj_path.name}: нет контекста {traj.context_file}")
            continue

        messages = trajectory_to_messages(wn, traj, context_text)
        records.append(messages_to_jsonl_record(messages))

    return records


def main():
    parser = argparse.ArgumentParser(description="Собрать JSONL-датасет")
    parser.add_argument(
        "--dataset",
        default=DATASET_PATH,
        help="Путь к TSV-датасету относительно корня проекта (дефолт: %(default)s)",
    )
    parser.add_argument("--row-index", type=int, help="0-based номер строки TSV")
    parser.add_argument("--word", help="Все строки TSV с этим словом")
    parser.add_argument(
        "--limit",
        type=int,
        help="Первые N строк TSV (если не указан ни --limit, ни --row-index, ни --word — весь файл)",
    )
    parser.add_argument(
        "--out",
        default=str(Path(OUTPUT_DATASET) / "train.jsonl"),
        help="Выходной JSONL",
    )
    parser.add_argument(
        "--from-cache",
        action="store_true",
        help="Не пересчитывать trajectories, читать из cache",
    )
    parser.add_argument(
        "--traj-dir",
        default=OUTPUT_TRAJECTORIES,
        help="Директория с trajectory JSON",
    )
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
    traj_dir = Path(args.traj_dir)
    traj_dir.mkdir(parents=True, exist_ok=True)

    log.info("Загрузка RuWordNet...")
    wn = RuWordNet(WORDNET_PATH)

    all_records: list[dict] = []

    with tqdm(
        rows,
        desc="Сборка",
        unit="строка",
        bar_format=_PROGRESS_BAR_FORMAT,
        dynamic_ncols=False,
    ) as pbar:
        for row in pbar:
            pbar.set_postfix_str(_format_current_word(row.word), refresh=False)
            if not args.from_cache:
                build_trajectories_for_row(
                    wn,
                    row,
                    traj_dir,
                    max_words=args.max_words,
                    max_hyponyms=args.max_hyponyms,
                )

            records = _process_row(
                wn,
                row,
                traj_dir,
                from_cache=True,
                max_words=args.max_words,
                max_hyponyms=args.max_hyponyms,
            )
            all_records.extend(records)
            pbar.refresh()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for record in all_records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    log.info(
        "Готово: %d записей → %s (датасет: %s, trajectories: %s)",
        len(all_records),
        out_path,
        args.dataset,
        traj_dir,
    )


if __name__ == "__main__":
    main()
