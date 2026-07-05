import json
import os
import re
import sys
from pathlib import Path

# При запуске `python scripts\step1_sanity.py` Python не видит корень репозитория.
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

import networkx as nx

from taxoenrich.core import RuWordNet

WORD = "АБСЕНТЕИЗМ"
DATASET_PATH = Path("datasets/context_analyser_dataset.tsv")
FASTTEXT_PATH = Path("examples/fasttext_baseline.json")
CORPUS_DIR = Path("corpus/annotated_texts")
WORDNET_PATH = "wordnets/RuWordNet"


def load_tsv_row(word: str):
    with open(DATASET_PATH, encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if parts[0] == word:
                node_ids = eval(parts[1])
                node_names = eval(parts[2])
                files = eval(parts[3])
                return node_ids, node_names, files
    raise ValueError(f"Слово {word} не найдено в TSV")


def load_fasttext_top3(word: str):
    with open(FASTTEXT_PATH, encoding="utf-8") as f:
        data = json.load(f)
    ids = data[word]
    # top-3 уникальных (как в MAP@3)
    seen = []
    for sid in ids:
        if sid not in seen:
            seen.append(sid)
        if len(seen) == 3:
            break
    return seen


def load_context(filename: str, max_chars: int = 500):
    path = CORPUS_DIR / filename
    text = path.read_text(encoding="utf-8")
    m = re.search(r"<predict_kb>(.*?)</predict_kb>", text)
    tag = m.group(1) if m else "???"
    snippet = text[:max_chars].replace("\n", " ")
    return tag, snippet


def synset_info(wn, sid):
    s = wn.synsets[sid]
    return s.synset_name, list(s.synset_words)[:5]


def find_path_up(wn, start, target):
    """Путь вверх по гиперонимам: start -> ... -> target"""
    G_up = nx.DiGraph()
    for sid, syn in wn.synsets.items():
        for parent in syn.rels.get("hypernym", []):
            G_up.add_edge(sid, parent)
    try:
        return nx.shortest_path(G_up, start, target)
    except nx.NetworkXNoPath:
        return None


def find_path_down(wn, start, target):
    """Путь вниз по гипонимам: start -> ... -> target"""
    try:
        return nx.shortest_path(wn.graph, start, target)
    except nx.NetworkXNoPath:
        return None


def main():
    print("=" * 60)
    print("ШАГ 1: sanity check для одного примера")
    print("=" * 60)

    print("\n[1] Загрузка RuWordNet...")
    wn = RuWordNet(WORDNET_PATH)
    print(f"    Загружено synsets: {len(wn.synsets)}")

    print(f"\n[2] Строка из TSV для слова: {WORD}")
    target_ids, target_names, files = load_tsv_row(WORD)
    print(f"    Целевые synset ID:   {target_ids}")
    print(f"    Имена synsets:       {target_names}")
    print(f"    Файл контекста:      {files[0]}")

    print(f"\n[3] FastText стартовые узлы (top-3 уникальных):")
    start_ids = load_fasttext_top3(WORD)
    for i, sid in enumerate(start_ids, 1):
        name, words = synset_info(wn, sid)
        print(f"    {i}. {sid} — {name} — слова: {words}")

    print(f"\n[4] Контекст из корпуса:")
    tag, snippet = load_context(files[0])
    print(f"    Слово в теге: {tag}")
    print(f"    Начало текста: {snippet}...")

    target = target_ids[0]
    t_name, t_words = synset_info(wn, target)
    print(f"\n[5] Ground truth (первый target): {target} — {t_name}")

    print(f"\n[6] Пути от каждого стартового узла к target:")
    for sid in start_ids:
        s_name, _ = synset_info(wn, sid)
        path_up = find_path_up(wn, sid, target)
        path_down = find_path_down(wn, sid, target)

        print(f"\n    Старт: {sid} ({s_name})")
        if path_up:
            print(f"      Путь ВВЕРХ (hypernyms): {' -> '.join(path_up)}")
            print(f"      Шагов: {len(path_up) - 1}")
        else:
            print("      Путь ВВЕРХ: не найден")

        if path_down:
            print(f"      Путь ВНИЗ (hyponyms):  {' -> '.join(path_down)}")
            print(f"      Шагов: {len(path_down) - 1}")
        else:
            print("      Путь ВНИЗ: не найден")

    print("\n" + "=" * 60)
    print("Готово. Если synsets загрузились и контекст прочитался — шаг 1 OK.")
    print("=" * 60)


if __name__ == "__main__":
    main()