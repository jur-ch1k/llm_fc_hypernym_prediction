"""
Анализ распределения токенов по частям SFT-датасета с TSV-ответами get_hyponyms.

Для каждой записи выводит таблицу с абсолютным числом токенов и долей (%)
по категориям, затем сводку min / max / mean.

Категория tool (ответы get_hyponyms, role=tool) считается суммарно и
разбивается на подгруппы: Определение, Слова, Гипонимы, Имена,
ID синсетов, всего_гипонимов, Прочее.
Парсер ожидает TSV из tools_runtime (колонки id, название, определение,
слова, всего_гипонимов, гипонимы; заголовок строки не включается — описан в system prompt).

Запуск:
    python scripts/analyze_dataset_tokens_tsv.py output/dataset/validate_10_tsv.jsonl
    python scripts/analyze_dataset_tokens_tsv.py output/dataset/validate_10_tsv.jsonl --model Qwen/Qwen3.5-2B
    python scripts/analyze_dataset_tokens_tsv.py output/dataset/validate_10_tsv.jsonl -o tsv_tokens_data.txt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent
ROOT = SCRIPTS.parent
REPO_ROOT = ROOT.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(SCRIPTS))

from analyze_dataset_tokens import (  # noqa: E402
    TokenBreakdown,
    apply_tool_breakdown,
    count_tokens,
    load_records,
    load_tokenizer,
    resolve_output_path,
    serialize_tool_calls,
    write_analysis_report,
)

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

TSV_HEADER_HYPONYMS = (
    "id\tназвание\tопределение\tслова\tвсего_гипонимов\tгипонимы"
)
TSV_HEADER_HYPERNYMS = (
    "id\tназвание\tопределение\tслова\tвсего_гиперонимов\tгиперонимы"
)
TSV_HEADERS = {TSV_HEADER_HYPONYMS, TSV_HEADER_HYPERNYMS}
EMPTY_TOOL_MESSAGES = {"Гипонимов не найдено.", "Гиперонимов не найдено."}

COL_ID = 0
COL_NAME = 1
COL_DEFINITION = 2
COL_WORDS = 3
COL_TOTAL = 4
COL_CHILDREN = 5
EXPECTED_COLS = 6


def _pad_row(cells: list[str]) -> list[str]:
    padded = cells[:EXPECTED_COLS]
    while len(padded) < EXPECTED_COLS:
        padded.append("")
    return padded


def _is_tsv_header_line(line: str) -> bool:
    return line.strip() in TSV_HEADERS


def _empty_sections(extra_other: str = "") -> dict[str, str]:
    return {
        "definition": "",
        "words": "",
        "hyponyms": "",
        "names": "",
        "ids": "",
        "totals": "",
        "other": extra_other,
    }


def parse_tool_sections_tsv(content: str) -> dict[str, str]:
    """Разбивает TSV-ответ get_hyponyms/get_hypernyms на секции для подсчёта токенов."""
    stripped = content.strip()
    if not stripped:
        return _empty_sections()

    if stripped in EMPTY_TOOL_MESSAGES:
        return _empty_sections(stripped)

    ids: list[str] = []
    names: list[str] = []
    definitions: list[str] = []
    words: list[str] = []
    totals: list[str] = []
    children: list[str] = []

    for line in stripped.split("\n"):
        if not line.strip() or _is_tsv_header_line(line):
            continue
        cells = _pad_row(line.split("\t"))
        if cells[COL_ID]:
            ids.append(cells[COL_ID])
        if cells[COL_NAME]:
            names.append(cells[COL_NAME])
        if cells[COL_DEFINITION]:
            definitions.append(cells[COL_DEFINITION])
        if cells[COL_WORDS]:
            words.append(cells[COL_WORDS])
        if cells[COL_TOTAL]:
            totals.append(cells[COL_TOTAL])
        if cells[COL_CHILDREN]:
            children.append(cells[COL_CHILDREN])

    if not ids and not names and not definitions and not words and not totals and not children:
        return _empty_sections(stripped)

    return {
        "definition": "\n".join(definitions).strip(),
        "words": "\n".join(words).strip(),
        "hyponyms": "\n".join(children).strip(),
        "names": "\n".join(names).strip(),
        "ids": "\n".join(ids).strip(),
        "totals": "\n".join(totals).strip(),
        "other": "",
    }


def analyze_messages(messages: list[dict], tokenizer) -> TokenBreakdown:
    breakdown = TokenBreakdown()

    for msg in messages:
        role = msg.get("role")
        content = msg.get("content") or ""

        if role == "system":
            breakdown.system += count_tokens(tokenizer, content)
        elif role == "user":
            breakdown.user += count_tokens(tokenizer, content)
        elif role == "assistant":
            if msg.get("tool_calls"):
                breakdown.tool_call_args += count_tokens(
                    tokenizer, serialize_tool_calls(msg["tool_calls"])
                )
            elif content:
                breakdown.assistant_final += count_tokens(tokenizer, content)
        elif role == "tool":
            apply_tool_breakdown(breakdown, tokenizer, content, parse_tool_sections_tsv(content))

    return breakdown


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Анализ токенов SFT-датасета (TSV-ответы get_hyponyms)",
    )
    parser.add_argument(
        "dataset",
        help="Путь к JSONL или JSON (одна запись / массив)",
    )
    parser.add_argument(
        "--model",
        default="Qwen/Qwen3.5-2B",
        help="HF model id или локальный путь к каталогу модели (default: Qwen/Qwen3.5-2B)",
    )
    parser.add_argument(
        "--tokenizer-json",
        default=None,
        help="Явный путь к tokenizer.json (приоритет над --model)",
    )
    parser.add_argument(
        "--output",
        "-o",
        default=None,
        help="Записать отчёт (таблицы + сводка) в файл вместо stdout",
    )
    args = parser.parse_args()

    dataset_path = Path(args.dataset)
    if not dataset_path.is_absolute():
        dataset_path = ROOT / dataset_path

    records = load_records(dataset_path)
    if not records:
        print("Датасет пуст.")
        sys.exit(1)

    tokenizer, approximate = load_tokenizer(args.model, args.tokenizer_json)

    preamble: list[str] = []
    if approximate:
        preamble.append(
            "(!) Абсолютные числа токенов приблизительны; "
            "для точных — pip install transformers"
        )
    preamble.append(f"Записей: {len(records)}")
    preamble.append("Формат tool-ответов: TSV (get_hyponyms / get_hypernyms)")

    if args.output:
        out_path = resolve_output_path(args.output)
        with out_path.open("w", encoding="utf-8") as out_file:
            write_analysis_report(
                records,
                tokenizer,
                analyze_messages,
                out=out_file,
                preamble=preamble,
            )
        print(f"Отчёт записан в {out_path}")
    else:
        write_analysis_report(
            records,
            tokenizer,
            analyze_messages,
            preamble=preamble,
        )


if __name__ == "__main__":
    main()
