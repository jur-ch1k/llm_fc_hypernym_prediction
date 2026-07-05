"""
Учебный пример: один реальный путь в RuWordNet для слова АБСЕНТЕИЗМ.

Точка A — первый стартовый узел FastText (119563-N).
Точка B — ground truth из датасета (147309-N).

Путь ищется через networkx на неориентированном графе таксономии
(ребро = связь гипероним/гипоним). На каждом шаге определяем,
какой tool call сделала бы модель: get_hyponyms или get_hypernyms.

Запуск из корня репозитория:
    python scripts/step1_demo_path.py
"""

import logging
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

import networkx as nx

from taxoenrich.core import RuWordNet

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
)
log = logging.getLogger(__name__)

# --- зашитый учебный пример ---
WORD = "АБСЕНТЕИЗМ"
START_ID = "119563-N"   # FastText top-1 для АБСЕНТЕИЗМ
TARGET_ID = "147309-N"  # ground truth #1 из context_analyser_dataset.tsv
WORDNET_PATH = "wordnets/RuWordNet"


def describe_synset(wn: RuWordNet, synset_id: str) -> str:
    s = wn.synsets[synset_id]
    words = ", ".join(w.replace("_", " ") for w in list(s.synset_words)[:4])
    return f"{s.synset_name} `{synset_id}` [{words}]"


def build_undirected_taxonomy(wn: RuWordNet) -> nx.Graph:
    """Граф «родитель — ребёнок» без направления (для поиска маршрута)."""
    graph = nx.Graph()
    for parent_id, synset in wn.synsets.items():
        for child_id in synset.rels.get("hyponym", []):
            graph.add_edge(parent_id, child_id)
    return graph


def tool_call_between(wn: RuWordNet, from_id: str, to_id: str) -> tuple[str, str]:
    """
    Какой tool call переводит модель из from_id в to_id за один шаг.
    Возвращает (имя функции, node_id аргумента).
    """
    if to_id in wn.synsets[from_id].rels.get("hyponym", []):
        return "get_hyponyms", from_id
    if from_id in wn.synsets[to_id].rels.get("hyponym", []):
        return "get_hypernyms", to_id
    raise ValueError(f"Нет прямой связи между {from_id} и {to_id}")


def main():
    log.info("=" * 70)
    log.info("Учебный пример: путь в RuWordNet для «%s»", WORD)
    log.info("=" * 70)

    log.info("\n[1] Загрузка RuWordNet...")
    wn = RuWordNet(WORDNET_PATH)
    log.info("    Synsets: %d", len(wn.synsets))

    log.info("\n[2] Заданные точки маршрута:")
    log.info("    A (старт, FastText #1): %s", describe_synset(wn, START_ID))
    log.info("    B (цель, ground truth): %s", describe_synset(wn, TARGET_ID))
    log.info(
        "    Контекст: «рейтинг абсентеизма» — отсутствие на работе без уважительной причины"
    )

    log.info("\n[3] Поиск кратчайшего пути в графе таксономии...")
    graph = build_undirected_taxonomy(wn)
    path = nx.shortest_path(graph, START_ID, TARGET_ID)
    log.info("    Найден путь из %d узлов (%d шагов навигации)", len(path), len(path) - 1)

    log.info("\n[4] Маршрут — узел за узлом:")
    log.info("-" * 70)
    for i, node_id in enumerate(path):
        marker = {0: "СТАРТ", len(path) - 1: "ЦЕЛЬ"}.get(i, f"шаг {i}")
        log.info("  [%s] %s", marker, describe_synset(wn, node_id))

    log.info("\n[5] Те же шаги как tool calls (так модель ходит по дереву):")
    log.info("-" * 70)
    for i in range(1, len(path)):
        prev_id, next_id = path[i - 1], path[i]
        func, arg_id = tool_call_between(wn, prev_id, next_id)
        direction = "вниз ↓" if func == "get_hyponyms" else "вверх ↑"
        log.info("")
        log.info("  Шаг %d (%s):", i, direction)
        log.info("    Вызов:  %s(node_id=%r)", func, arg_id)
        log.info("    Было:   %s", describe_synset(wn, prev_id))
        log.info("    Стало:  %s", describe_synset(wn, next_id))

    log.info("\n[6] Итог:")
    log.info("-" * 70)
    log.info(
        "  Из «%s» в «%s» можно дойти за %d tool call(s).",
        wn.synsets[START_ID].synset_name,
        wn.synsets[TARGET_ID].synset_name,
        len(path) - 1,
    )
    log.info(
        "  FastText дал далёкую ветку («мировоззрение»), но через граф "
        "маршрут к правильному synset существует."
    )
    log.info(
        "  На шаге 2 модель идёт ВВЕРХ (get_hypernyms) — только вниз путь не находился."
    )
    log.info("=" * 70)


if __name__ == "__main__":
    main()
