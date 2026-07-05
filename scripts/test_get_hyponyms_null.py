"""
Проверка: что возвращает get_hyponyms(node_id=null) через API RuWordNet.

Сравни вывод с graph_edges.py (--kind roots): списки должны совпасть.

Запуск:
    python scripts/test_get_hyponyms_null.py
    python scripts/test_get_hyponyms_null.py --pos N
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from taxoenrich.core import RuWordNet

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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--wordnet", default=None)
    parser.add_argument("--pos", choices=["N", "A", "V"], default="N")
    args = parser.parse_args()

    wordnet_path = resolve_wordnet_path(args.wordnet)
    print(f"Загрузка RuWordNet: {wordnet_path}")
    wn = RuWordNet(wordnet_path)

    print(f"\nВызов: wn.get_hyponyms(None, pos={args.pos!r})")
    print("       (это то же самое, что get_hyponyms(node_id=null) в tool call)\n")

    results = wn.get_hyponyms(None, pos=args.pos)
    print(f"Найдено: {len(results)} synsets\n")
    print("-" * 70)

    for i, item in enumerate(results, 1):
        words = "; ".join(item["words"][:4])
        hypos = len(item.get("hyponyms") or [])
        print(f"{i:3d}. {item['name']} `{item['id']}`")
        print(f"     слова: {words}")
        print(f"     гипонимов у узла: {hypos}")
        if item.get("definition"):
            defn = item["definition"]
            if len(defn) > 80:
                defn = defn[:77] + "..."
            print(f"     определение: {defn}")
        print()

    ids = sorted(r["id"] for r in results)
    print("-" * 70)
    print("ID (для сравнения с graph_edges.py):")
    print(", ".join(ids))


if __name__ == "__main__":
    main()
