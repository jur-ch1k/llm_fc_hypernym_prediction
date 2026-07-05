"""
Для каждой пары (слово, target_id) из context_analyser_dataset.tsv проверяет:
  1) есть ли слово где-либо в RuWordNet (pos=N);
  2) входит ли слово в лексемы gold-синсета target_id.

На основе этого предлагается финальный ответ SFT:
  - include in  -> слово уже среди synset_words target (правило 1)
  - hyponym of  -> слово не в target-синсете (правило 2)

Запуск:
    python scripts/classify_sft_final_decision.py
    python scripts/classify_sft_final_decision.py --show-samples
    python scripts/classify_sft_final_decision.py --output output/sft_final_decision_stats.json
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import sys
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from taxoenrich.core import RuWordNet
from utils.data_processing import normalize_word

DEFAULT_WORDNET = "wordnets/RuWordNet"
FALLBACK_WORDNET = ROOT.parent / "data" / "RuWordNet"
DEFAULT_DATASET = Path("datasets/context_analyser_dataset.tsv")
FALLBACK_DATASET = ROOT.parent / "data" / "datasets" / "datasets" / "context_analyser_dataset.tsv"


@dataclass
class TargetRecord:
    row_index: int
    word: str
    target_id: str
    target_name: str
    in_graph: bool
    graph_synset_ids: list[str]
    in_target_synset: bool
    matched_forms: list[str]
    suggested_decision: str  # include_in | hyponym_of
    target_in_wordnet: bool


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


def word_forms(word: str) -> set[str]:
    """Нормализованные формы слова в стиле RuWordNet (lower, underscore)."""
    forms: set[str] = set()
    lower = word.strip().lower()

    if not lower:
        return forms

    forms.add(lower.replace(" ", "_"))

    parts = lower.replace("_", " ").split()
    if len(parts) == 1:
        try:
            forms.add(normalize_word(lower).replace(" ", "_"))
        except Exception:
            pass
    else:
        lemmatized = "_".join(normalize_word(part) for part in parts)
        forms.add(lemmatized)

    return {f for f in forms if f}


def synset_forms(synset_words: set[str]) -> set[str]:
    return {w.lower() for w in synset_words}


def find_graph_synsets(wn: RuWordNet, forms: set[str], pos: str) -> list[str]:
    found: set[str] = set()
    for form in forms:
        for synset_id in wn.sense2synid.get(form, []):
            synset = wn.synsets.get(synset_id)
            if synset is None:
                continue
            if synset.synset_type != pos:
                continue
            found.add(synset_id)
    return sorted(found)


def classify_record(
    wn: RuWordNet,
    row_index: int,
    word: str,
    target_id: str,
    target_name: str,
    pos: str,
) -> TargetRecord:
    forms = word_forms(word)
    graph_synset_ids = find_graph_synsets(wn, forms, pos)

    target_in_wordnet = target_id in wn.synsets
    in_target_synset = False
    matched_forms: list[str] = []

    if target_in_wordnet:
        target_words = synset_forms(wn.synsets[target_id].synset_words)
        matched_forms = sorted(forms & target_words)
        in_target_synset = bool(matched_forms)

    suggested = "include_in" if in_target_synset else "hyponym_of"

    return TargetRecord(
        row_index=row_index,
        word=word,
        target_id=target_id,
        target_name=target_name,
        in_graph=bool(graph_synset_ids),
        graph_synset_ids=graph_synset_ids,
        in_target_synset=in_target_synset,
        matched_forms=matched_forms,
        suggested_decision=suggested,
        target_in_wordnet=target_in_wordnet,
    )


def iter_dataset_records(dataset_path: Path) -> list[tuple[int, str, str, str]]:
    """(row_index, word, target_id, target_name) для каждой пары word x target."""
    rows: list[tuple[int, str, str, str]] = []
    with open(dataset_path, encoding="utf-8") as f:
        for row_index, line in enumerate(f):
            parts = line.strip().split("\t")
            if len(parts) < 3:
                continue
            word = parts[0]
            target_ids = ast.literal_eval(parts[1])
            target_names = ast.literal_eval(parts[2])
            for target_id, target_name in zip(target_ids, target_names):
                rows.append((row_index, word, target_id, target_name))
    return rows


def describe_synset(wn: RuWordNet, synset_id: str) -> str:
    synset = wn.synsets[synset_id]
    return f"{synset.synset_name} `{synset_id}`"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Классификация финального ответа SFT: include_in vs hyponym_of"
    )
    parser.add_argument("--wordnet", default=None)
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--pos", choices=["N", "A", "V"], default="N")
    parser.add_argument(
        "--show-samples",
        action="store_true",
        help="Показать по несколько примеров для каждого класса",
    )
    parser.add_argument(
        "--show-all-include",
        action="store_true",
        help="Показать все случаи include_in",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="JSON с деталями по каждой записи",
    )
    args = parser.parse_args()

    wordnet_path = resolve_wordnet_path(args.wordnet)
    dataset_path = resolve_dataset_path(args.dataset)

    print(f"Загрузка RuWordNet: {wordnet_path}")
    wn = RuWordNet(wordnet_path)
    print(f"Synsets: {len(wn.synsets)}, senses: {len(wn.senses)}")
    print(f"Датасет: {dataset_path}")

    raw_rows = iter_dataset_records(dataset_path)
    print(f"Записей (word x target): {len(raw_rows)}")

    records: list[TargetRecord] = []
    missing_target = 0
    for row_index, word, target_id, target_name in raw_rows:
        rec = classify_record(wn, row_index, word, target_id, target_name, args.pos)
        records.append(rec)
        if not rec.target_in_wordnet:
            missing_target += 1

    decision_hist = Counter(r.suggested_decision for r in records)
    in_graph_hist = Counter("in_graph" if r.in_graph else "not_in_graph" for r in records)
    cross_hist = Counter(
        (
            "in_target_synset -> include_in"
            if r.in_target_synset
            else ("in_graph_other -> hyponym_of" if r.in_graph else "not_in_graph -> hyponym_of")
        )
        for r in records
    )

    unique_words = {r.word for r in records}
    words_in_graph = {r.word for r in records if r.in_graph}
    words_not_in_graph = unique_words - words_in_graph

    print()
    print("=" * 70)
    print("Предлагаемый финальный ответ SFT (по паре word x target_id)")
    print("=" * 70)
    for key in sorted(decision_hist.keys()):
        n = decision_hist[key]
        label = "include in (правило 1)" if key == "include_in" else "hyponym of (правило 2)"
        print(f"  {label:35s} {n:5d} ({100 * n / len(records):.1f}%)")

    print()
    print("=" * 70)
    print("Слово в графе RuWordNet (pos=N), по записям word x target")
    print("=" * 70)
    for key in sorted(in_graph_hist.keys()):
        n = in_graph_hist[key]
        label = "есть в графе" if key == "in_graph" else "нет в графе"
        print(f"  {label:35s} {n:5d} ({100 * n / len(records):.1f}%)")

    print()
    print("=" * 70)
    print("Детализация")
    print("=" * 70)
    for label, n in sorted(cross_hist.items(), key=lambda x: -x[1]):
        print(f"  {label:40s} {n:5d} ({100 * n / len(records):.1f}%)")

    print()
    print(f"Уникальных слов в TSV: {len(unique_words)}")
    print(f"  слово есть в графе (хотя бы один target): {len(words_in_graph)}")
    print(f"  слова нет в графе:                     {len(words_not_in_graph)}")
    print(f"  target_id нет в RuWordNet:             {missing_target}")

    in_graph_records = [r for r in records if r.in_graph]
    if in_graph_records:
        print()
        print("=" * 70)
        print("Слова, которые уже есть в графе (но не в gold-target)")
        print("=" * 70)
        for rec in in_graph_records:
            synset_desc = describe_synset(wn, rec.target_id) if rec.target_in_wordnet else rec.target_id
            print(
                f"  row={rec.row_index} {rec.word} -> {rec.target_id} ({rec.target_name}); "
                f"синсеты слова: {', '.join(rec.graph_synset_ids)}; gold: {synset_desc}"
            )

    if args.show_samples or args.show_all_include:
        include_records = [r for r in records if r.suggested_decision == "include_in"]
        hyponym_records = [r for r in records if r.suggested_decision == "hyponym_of"]
        show_include = include_records if args.show_all_include else include_records[:10]

        print()
        print("=" * 70)
        print("Примеры: include_in (слово уже в target-синсете)")
        print("=" * 70)
        if not show_include:
            print("  (нет таких записей)")
        for rec in show_include:
            synset_desc = describe_synset(wn, rec.target_id) if rec.target_in_wordnet else rec.target_id
            print(f"\nrow={rec.row_index} {rec.word} -> {rec.target_id} ({rec.target_name})")
            print(f"  synset: {synset_desc}")
            print(f"  matched forms: {', '.join(rec.matched_forms) or '-'}")
            if rec.graph_synset_ids and rec.target_id not in rec.graph_synset_ids:
                others = [sid for sid in rec.graph_synset_ids if sid != rec.target_id]
                if others:
                    print(f"  также в других synsetах: {', '.join(others[:3])}")

        if not args.show_all_include:
            print()
            print("=" * 70)
            print("Примеры: hyponym_of (слова нет в target-синсете)")
            print("=" * 70)
            for rec in hyponym_records[:10]:
                print(f"\nrow={rec.row_index} {rec.word} -> {rec.target_id} ({rec.target_name})")
                print(f"  forms: {', '.join(sorted(word_forms(rec.word)))}")
                if rec.in_graph:
                    ids = ", ".join(rec.graph_synset_ids[:3])
                    suffix = f" (+{len(rec.graph_synset_ids) - 3})" if len(rec.graph_synset_ids) > 3 else ""
                    print(f"  слово в графе, но не в target: {ids}{suffix}")
                else:
                    print("  слова нет в графе")

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "pos": args.pos,
            "n_records": len(records),
            "n_unique_words": len(unique_words),
            "histogram_decision": dict(decision_hist),
            "histogram_in_graph": dict(in_graph_hist),
            "histogram_detail": dict(cross_hist),
            "records": [asdict(r) for r in records],
        }
        args.output.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"\nJSON сохранён: {args.output}")


if __name__ == "__main__":
    main()
