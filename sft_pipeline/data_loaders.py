import ast
import json
import logging
import warnings
from dataclasses import dataclass

from utils.data_processing import load_corpus_text

from .config import CORPUS_DIR, DATASET_PATH, FASTTEXT_PATH, FASTTEXT_TOP_K

log = logging.getLogger(__name__)


@dataclass
class DatasetRow:
    row_index: int
    word: str
    target_ids: list[str]
    target_names: list[str]
    context_files: list[str]


@dataclass
class WordRecord:
    word: str
    target_ids: list[str]
    target_names: list[str]
    context_files: list[str]


def _parse_tsv_line(line: str, row_index: int) -> DatasetRow:
    parts = line.strip().split("\t")
    return DatasetRow(
        row_index=row_index,
        word=parts[0],
        target_ids=ast.literal_eval(parts[1]),
        target_names=ast.literal_eval(parts[2]),
        context_files=ast.literal_eval(parts[3]),
    )


def iter_dataset_rows(
    limit: int | None = None,
    dataset_path: str = DATASET_PATH,
) -> list[DatasetRow]:
    rows: list[DatasetRow] = []
    with open(dataset_path, encoding="utf-8") as f:
        for row_index, line in enumerate(f):
            rows.append(_parse_tsv_line(line, row_index))
            if limit is not None and len(rows) >= limit:
                break
    return rows


def load_dataset_row(
    row_index: int,
    dataset_path: str = DATASET_PATH,
) -> DatasetRow:
    with open(dataset_path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i == row_index:
                return _parse_tsv_line(line, row_index)
    raise ValueError(f"Строка row_index={row_index} не найдена в {dataset_path}")


def iter_rows_by_word(
    word: str,
    dataset_path: str = DATASET_PATH,
) -> list[DatasetRow]:
    rows: list[DatasetRow] = []
    with open(dataset_path, encoding="utf-8") as f:
        for row_index, line in enumerate(f):
            parsed = _parse_tsv_line(line, row_index)
            if parsed.word == word:
                rows.append(parsed)
    return rows


def resolve_context_file(row: DatasetRow) -> str:
    return row.context_files[0]


def load_word_record(word: str, dataset_path: str = DATASET_PATH) -> WordRecord:
    warnings.warn(
        "load_word_record is deprecated for SFT; use load_dataset_row / iter_dataset_rows",
        DeprecationWarning,
        stacklevel=2,
    )
    rows = iter_rows_by_word(word, dataset_path)
    if not rows:
        raise ValueError(f"Слово {word!r} не найдено в {dataset_path}")
    row = rows[0]
    return WordRecord(
        word=row.word,
        target_ids=row.target_ids,
        target_names=row.target_names,
        context_files=row.context_files,
    )


def iter_dataset_words(limit: int | None = None, dataset_path: str = DATASET_PATH) -> list[str]:
    words: list[str] = []
    with open(dataset_path, encoding="utf-8") as f:
        for line in f:
            words.append(line.strip().split("\t")[0])
            if limit is not None and len(words) >= limit:
                break
    return words


def load_fasttext_topk(
    word: str,
    k: int = FASTTEXT_TOP_K,
    fasttext_path: str = FASTTEXT_PATH,
) -> list[str]:
    with open(fasttext_path, encoding="utf-8") as f:
        data = json.load(f)
    if word not in data:
        raise ValueError(f"Слово {word!r} не найдено в {fasttext_path}")
    seen: list[str] = []
    for sid in data[word]:
        if sid not in seen:
            seen.append(sid)
        if len(seen) == k:
            break
    return seen


def load_context(context_file: str, corpus_dir: str = CORPUS_DIR) -> str | None:
    return load_corpus_text(corpus_dir, context_file)
