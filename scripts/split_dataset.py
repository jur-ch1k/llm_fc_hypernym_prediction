"""
Случайное разбиение context_analyser_dataset.tsv на train / eval / test.

Разбивает по уникальным словам (первый столбец) для избежания data leakage:
одно слово не попадёт в разные сплиты. Перемешивает слова с фиксированным
seed и сохраняет три TSV в подпапку seed_{seed} рядом с исходным файлом.

Запуск:
    python scripts/split_dataset.py
    python scripts/split_dataset.py --seed 42
    python scripts/split_dataset.py --train 80 --eval 10 --test 10 --seed 0
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT = ROOT / "datasets/context_analyser_dataset.tsv"

DEFAULT_TRAIN_PCT = 70
DEFAULT_EVAL_PCT = 10
DEFAULT_TEST_PCT = 20
DEFAULT_SEED = 42


def _resolve_splits(
    train: int | None,
    eval_pct: int | None,
    test: int | None,
) -> tuple[int, int, int]:
    values = (train, eval_pct, test)
    if all(v is None for v in values):
        return DEFAULT_TRAIN_PCT, DEFAULT_EVAL_PCT, DEFAULT_TEST_PCT

    if not all(v is not None for v in values):
        raise SystemExit(
            "Укажите все три доли (--train, --eval, --test) или не передавайте ни одной "
            "(тогда используется 70/10/20)."
        )

    total = train + eval_pct + test
    if total != 100:
        raise SystemExit(
            f"Сумма долей должна быть 100, получено {total} "
            f"(train={train}, eval={eval_pct}, test={test})."
        )

    return train, eval_pct, test


def _read_lines(path: Path) -> list[str]:
    with open(path, encoding="utf-8") as f:
        lines = [line for line in f if line.strip()]
    if not lines:
        raise SystemExit(f"Файл пуст: {path}")
    return lines


def _group_lines_by_word(lines: list[str]) -> dict[str, list[str]]:
    """Группирует строки по первому столбцу (слову)."""
    groups: dict[str, list[str]] = {}
    for line in lines:
        word = line.split("\t", 1)[0].strip()
        groups.setdefault(word, []).append(line)
    return groups


def _write_lines(path: Path, lines: list[str]) -> None:
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for line in lines:
            f.write(line if line.endswith("\n") else line + "\n")


def split_lines(
    lines: list[str],
    train_pct: int,
    eval_pct: int,
    test_pct: int,
    seed: int,
) -> tuple[list[str], list[str], list[str]]:
    """Разбивает строки по уникальным словам (первый столбец) для избежания data leakage."""
    word_groups = _group_lines_by_word(lines)
    words = list(word_groups.keys())

    rng = random.Random(seed)
    rng.shuffle(words)

    n = len(words)
    test_n = n * test_pct // 100
    eval_n = n * eval_pct // 100
    # Остаток от деления уходит в train.
    train_n = n - test_n - eval_n

    if train_n < 0:
        raise RuntimeError("Некорректное разбиение: отрицательный размер train.")

    test_words = words[:test_n]
    eval_words = words[test_n : test_n + eval_n]
    train_words = words[test_n + eval_n :]

    train_lines = [line for w in train_words for line in word_groups[w]]
    eval_lines = [line for w in eval_words for line in word_groups[w]]
    test_lines = [line for w in test_words for line in word_groups[w]]

    return train_lines, eval_lines, test_lines


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Случайный split TSV-датасета на train / eval / test."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help=f"Путь к исходному TSV (по умолчанию: {DEFAULT_INPUT})",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help=f"Seed для перемешивания (по умолчанию: {DEFAULT_SEED})",
    )
    parser.add_argument(
        "--train",
        type=int,
        default=None,
        metavar="PCT",
        help="Доля train в процентах (вместе с --eval и --test, сумма = 100)",
    )
    parser.add_argument(
        "--eval",
        type=int,
        default=None,
        metavar="PCT",
        help="Доля eval в процентах",
    )
    parser.add_argument(
        "--test",
        type=int,
        default=None,
        metavar="PCT",
        help="Доля test в процентах",
    )
    args = parser.parse_args()

    input_path: Path = args.input.resolve()
    if not input_path.is_file():
        raise SystemExit(f"Файл не найден: {input_path}")

    train_pct, eval_pct, test_pct = _resolve_splits(args.train, args.eval, args.test)

    lines = _read_lines(input_path)
    train_lines, eval_lines, test_lines = split_lines(
        lines, train_pct, eval_pct, test_pct, args.seed
    )

    out_dir = input_path.parent / f"seed_{args.seed}"
    out_dir.mkdir(parents=True, exist_ok=True)

    train_path = out_dir / "train_dataset.tsv"
    eval_path = out_dir / "eval_dataset.tsv"
    test_path = out_dir / "test_dataset.tsv"

    _write_lines(train_path, train_lines)
    _write_lines(eval_path, eval_lines)
    _write_lines(test_path, test_lines)

    print(f"Исходный файл: {input_path}")
    print(f"Строк всего: {len(lines)}")
    print(f"Уникальных слов: {len(_group_lines_by_word(lines))}")
    print(f"Seed: {args.seed}")
    print(f"Доли: train={train_pct}% eval={eval_pct}% test={test_pct}%")
    print(f"Размеры (строки): train={len(train_lines)} eval={len(eval_lines)} test={len(test_lines)}")
    print(f"Размеры (слова): train={len(_group_lines_by_word(train_lines))} eval={len(_group_lines_by_word(eval_lines))} test={len(_group_lines_by_word(test_lines))}")
    print(f"Выходная папка: {out_dir}")
    print(f"  {train_path.name}")
    print(f"  {eval_path.name}")
    print(f"  {test_path.name}")


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass
    main()
