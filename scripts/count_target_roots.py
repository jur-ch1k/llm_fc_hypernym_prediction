"""
Для каждого target из context_analyser_dataset.tsv считает,
сколько корневых synsets (pos=N, без hypernym) достижимо вверх
по цепочке hypernym-рёбер.

Запуск:
    python scripts/count_target_roots.py
    python scripts/count_target_roots.py --show-multi
    python scripts/count_target_roots.py --output output/target_roots_stats.json
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from taxoenrich.core import RuWordNet

DEFAULT_WORDNET = "wordnets/RuWordNet"
FALLBACK_WORDNET = ROOT.parent / "data" / "RuWordNet"
DEFAULT_DATASET = Path("datasets/context_analyser_dataset.tsv")
FALLBACK_DATASET = ROOT.parent / "data" / "datasets" / "datasets" / "context_analyser_dataset.tsv"


def resolve_wordnet_path(explicit: str | None) -> str:
    if explicit:
        return explicit
    if Path(DEFAULT_WORDNET).exists():
        return DEFAULT_WORDNET
    if FALLBACK_WORDNET.exists():
        return str(FALLBACK_WORDNET)
    raise FileNotFoundError(
        f"RuWordNet не найден: ни {DEFAULT_WORDNET!r}, ни {FALLBACK_WORDNET!r}"
    )


def resolve_dataset_path(explicit: str | None) -> Path:
    if explicit:
        return Path(explicit)
    if DEFAULT_DATASET.exists():
        return DEFAULT_DATASET
    if FALLBACK_DATASET.exists():
        return FALLBACK_DATASET
    raise FileNotFoundError(
        f"TSV не найден: ни {DEFAULT_DATASET!r}, ни {FALLBACK_DATASET!r}"
    )


def collect_roots(wn: RuWordNet, pos: str) -> set[str]:
    roots: set[str] = set()
    for sid, synset in wn.synsets.items():
        if synset.synset_type != pos:
            continue
        if not synset.rels.get("hypernym"):
            roots.add(sid)
    return roots


def find_ancestor_roots(
    wn: RuWordNet,
    target_id: str,
    roots: set[str],
) -> set[str]:
    """
    BFS вверх по hypernym от target.
    Возвращает множество корней, которые являются предками target.
    """
    if target_id not in wn.synsets:
        return set()

    found: set[str] = set()
    queue = [target_id]
    visited: set[str] = set()

    while queue:
        node_id = queue.pop(0)
        if node_id in visited:
            continue
        visited.add(node_id)

        if node_id in roots:
            found.add(node_id)

        synset = wn.synsets.get(node_id)
        if synset is None:
            continue

        for parent_id in synset.rels.get("hypernym", []):
            if parent_id not in visited:
                queue.append(parent_id)

    return found


def iter_dataset_targets(dataset_path: Path) -> list[tuple[str, str, str]]:
    """(word, target_id, target_name) для каждой строки TSV."""
    rows: list[tuple[str, str, str]] = []
    with open(dataset_path, encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 3:
                continue
            word = parts[0]
            target_ids = ast.literal_eval(parts[1])
            target_names = ast.literal_eval(parts[2])
            for target_id, target_name in zip(target_ids, target_names):
                rows.append((word, target_id, target_name))
    return rows


def describe_synset(wn: RuWordNet, synset_id: str) -> str:
    s = wn.synsets[synset_id]
    return f"{s.synset_name} `{synset_id}`"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Сколько корней достижимо вверх от каждого target"
    )
    parser.add_argument("--wordnet", default=None)
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--pos", choices=["N", "A", "V"], default="N")
    parser.add_argument(
        "--show-multi",
        action="store_true",
        help="Вывести target с 0 или >1 корнем",
    )
    parser.add_argument(
        "--show-all",
        action="store_true",
        help="Вывести все target (может быть длинно)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="JSON с деталями по каждому target",
    )
    args = parser.parse_args()

    wordnet_path = resolve_wordnet_path(args.wordnet)
    dataset_path = resolve_dataset_path(args.dataset)

    print(f"Загрузка RuWordNet: {wordnet_path}")
    wn = RuWordNet(wordnet_path)

    roots = collect_roots(wn, args.pos)
    print(f"Корней (pos={args.pos}): {len(roots)}")
    print(f"Датасет: {dataset_path}")

    rows = iter_dataset_targets(dataset_path)
    print(f"Записей (word x target): {len(rows)}")

    # Уникальные target_id (один target может встречаться у разных слов)
    unique_targets: dict[str, list[tuple[str, str]]] = {}
    for word, target_id, target_name in rows:
        unique_targets.setdefault(target_id, []).append((word, target_name))

    print(f"Уникальных target_id: {len(unique_targets)}")
    print()

    per_target: dict[str, dict] = {}
    count_histogram: Counter[int] = Counter()
    missing_in_wordnet: list[str] = []

    for target_id, occurrences in sorted(unique_targets.items()):
        ancestor_roots = find_ancestor_roots(wn, target_id, roots)
        n_roots = len(ancestor_roots)

        if target_id not in wn.synsets:
            missing_in_wordnet.append(target_id)
            count_histogram[-1] += 1
        else:
            count_histogram[n_roots] += 1

        per_target[target_id] = {
            "target_name": occurrences[0][1],
            "words": sorted({w for w, _ in occurrences}),
            "n_ancestor_roots": n_roots,
            "ancestor_root_ids": sorted(ancestor_roots),
            "in_wordnet": target_id in wn.synsets,
        }

    print("=" * 70)
    print("Гистограмма: сколько корней достижимо вверх от target")
    print("=" * 70)
    for key in sorted(count_histogram.keys()):
        if key == -1:
            label = "нет в RuWordNet"
        elif key == 0:
            label = "0 корней (не поднимается до top-level)"
        elif key == 1:
            label = "1 корень (однозначно)"
        else:
            label = f"{key} корней (неоднозначность)"
        print(f"  {label:40s} {count_histogram[key]:5d}")

    total = len(unique_targets)
    one_root = count_histogram.get(1, 0)
    zero = count_histogram.get(0, 0)
    multi = sum(v for k, v in count_histogram.items() if k > 1)
    missing = count_histogram.get(-1, 0)

    print()
    print(f"Итого уникальных target: {total}")
    print(f"  ровно 1 корень:  {one_root} ({100 * one_root / total:.1f}%)")
    print(f"  0 корней:        {zero} ({100 * zero / total:.1f}%)")
    print(f"  2+ корней:       {multi} ({100 * multi / total:.1f}%)")
    print(f"  нет в wordnet:   {missing} ({100 * missing / total:.1f}%)")

    if args.show_multi or args.show_all:
        print()
        print("=" * 70)
        print("Детали по target")
        print("=" * 70)
        for target_id, info in sorted(per_target.items()):
            n = info["n_ancestor_roots"]
            if not args.show_all and n == 1 and info["in_wordnet"]:
                continue
            name = info["target_name"]
            words = ", ".join(info["words"][:3])
            if len(info["words"]) > 3:
                words += f" (+{len(info['words']) - 3})"
            root_ids = ", ".join(info["ancestor_root_ids"]) or "-"
            print(f"\n{target_id} - {name}")
            print(f"  слова в датасете: {words}")
            print(f"  ancestor-корней: {n}")
            print(f"  root ids: {root_ids}")
            if info["ancestor_root_ids"]:
                for rid in info["ancestor_root_ids"]:
                    if rid in wn.synsets:
                        print(f"    - {describe_synset(wn, rid)}")

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "pos": args.pos,
            "n_roots_total": len(roots),
            "root_ids": sorted(roots),
            "histogram": {str(k): v for k, v in sorted(count_histogram.items())},
            "targets": per_target,
        }
        args.output.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"\nJSON сохранён: {args.output}")


if __name__ == "__main__":
    main()
