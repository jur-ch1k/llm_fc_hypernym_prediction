"""
Краевые узлы RuWordNet: корни (без гиперонимов) и листья (без гипонимов).

RuWordNet — не одно дерево с единым корнем, а набор иерархий (DAG).
«Корень» в коде taxoenrich = synset без incoming hypernym-связей.
Первый tool call get_hyponyms(node_id=null) возвращает именно такие узлы.

Запуск из llm_fc_hypernym_prediction:
    python scripts/graph_edges.py
    python scripts/graph_edges.py --pos N --limit 20
    python scripts/graph_edges.py --kind roots --output output/roots_N.txt
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from taxoenrich.core import RuWordNet

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)

DEFAULT_WORDNET = "wordnets/RuWordNet"
FALLBACK_WORDNET = ROOT.parent / "data" / "RuWordNet"


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


def describe_synset(wn: RuWordNet, synset_id: str) -> str:
    s = wn.synsets[synset_id]
    words = ", ".join(w.replace("_", " ") for w in list(s.synset_words)[:4])
    hypo_n = len(s.rels.get("hyponym", []))
    hyper_n = len(s.rels.get("hypernym", []))
    return (
        f"{s.synset_name} `{synset_id}` [{s.synset_type}] "
        f"(hyper={hyper_n}, hypo={hypo_n}) [{words}]"
    )


def collect_roots(wn: RuWordNet, pos: str | None) -> list[str]:
    """Synsets без гиперонимов — верхний уровень иерархии (как get_hyponyms(None))."""
    roots: list[str] = []
    for sid, synset in wn.synsets.items():
        if pos is not None and synset.synset_type != pos:
            continue
        if not synset.rels.get("hypernym"):
            roots.append(sid)
    return sorted(roots)


def collect_leaves(wn: RuWordNet, pos: str | None) -> list[str]:
    """Synsets без гипонимов — самые конкретные понятия."""
    leaves: list[str] = []
    for sid, synset in wn.synsets.items():
        if pos is not None and synset.synset_type != pos:
            continue
        if not synset.rels.get("hyponym"):
            leaves.append(sid)
    return sorted(leaves)


def collect_isolated(wn: RuWordNet, pos: str | None) -> list[str]:
    """Synsets без hypernym и без hyponym — отдельные «острова»."""
    isolated: list[str] = []
    for sid, synset in wn.synsets.items():
        if pos is not None and synset.synset_type != pos:
            continue
        if not synset.rels.get("hypernym") and not synset.rels.get("hyponym"):
            isolated.append(sid)
    return sorted(isolated)


def print_nodes(
    wn: RuWordNet,
    title: str,
    node_ids: list[str],
    limit: int,
    output_path: Path | None,
) -> None:
    log.info("")
    log.info("=" * 70)
    log.info("%s: %d", title, len(node_ids))
    log.info("=" * 70)

    lines: list[str] = []
    shown = node_ids if limit <= 0 else node_ids[:limit]
    for i, sid in enumerate(shown, 1):
        line = f"  {i:4d}. {describe_synset(wn, sid)}"
        log.info(line)
        lines.append(line)

    if limit > 0 and len(node_ids) > limit:
        log.info("  ... ещё %d (используй --limit 0 для полного списка)", len(node_ids) - limit)

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        full_lines = [f"{i:4d}. {describe_synset(wn, sid)}" for i, sid in enumerate(node_ids, 1)]
        output_path.write_text("\n".join(full_lines) + "\n", encoding="utf-8")
        log.info("Полный список записан в %s", output_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Краевые узлы RuWordNet")
    parser.add_argument(
        "--wordnet",
        default=None,
        help=f"Путь к RuWordNet (по умолчанию {DEFAULT_WORDNET} или data/RuWordNet)",
    )
    parser.add_argument(
        "--pos",
        choices=["N", "A", "V"],
        default="N",
        help="Часть речи (по умолчанию N — существительные, как в эксперименте)",
    )
    parser.add_argument(
        "--kind",
        choices=["all", "roots", "leaves", "isolated"],
        default="all",
        help="Какие краевые узлы выводить",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=30,
        help="Сколько строк показать в консоли (0 = все)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Файл для полного списка (если указан — пишется весь kind, не только limit)",
    )
    args = parser.parse_args()

    wordnet_path = resolve_wordnet_path(args.wordnet)
    log.info("Загрузка RuWordNet из %s ...", wordnet_path)
    wn = RuWordNet(wordnet_path)
    log.info("Всего synsets: %d", len(wn.synsets))

    pos = args.pos
    roots = collect_roots(wn, pos)
    leaves = collect_leaves(wn, pos)
    isolated = collect_isolated(wn, pos)

    log.info("")
    log.info("Сводка для pos=%s:", pos)
    log.info("  корни (без hypernym):     %d", len(roots))
    log.info("  листья (без hyponym):    %d", len(leaves))
    log.info("  изолированные (оба 0):   %d", len(isolated))
    log.info(
        "  «корни» через API get_hyponyms(None): %d (должно совпасть с корнями выше)",
        len(wn.get_hyponyms(None, pos=pos)),
    )

    if args.kind in ("all", "roots"):
        print_nodes(
            wn,
            f"КОРНИ (верх таксономии, pos={pos})",
            roots,
            args.limit,
            args.output if args.kind == "roots" else None,
        )
    if args.kind in ("all", "leaves"):
        print_nodes(
            wn,
            f"ЛИСТЬЯ (низ таксономии, pos={pos})",
            leaves,
            args.limit,
            args.output if args.kind == "leaves" else None,
        )
    if args.kind in ("all", "isolated"):
        print_nodes(
            wn,
            f"ИЗОЛИРОВАННЫЕ (без связей hyper/hypo, pos={pos})",
            isolated,
            args.limit,
            args.output if args.kind == "isolated" else None,
        )


if __name__ == "__main__":
    main()
