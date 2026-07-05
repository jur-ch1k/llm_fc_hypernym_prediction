"""
Анализ распределения токенов по частям SFT-датасета.

Для каждой записи выводит таблицу с абсолютным числом токенов и долей (%)
по категориям, затем сводку min / max / mean.

Категория tool (ответы get_hyponyms, role=tool) считается суммарно и
разбивается на подгруппы: Определение, Слова, Гипонимы, Имена,
ID синсетов, всего_гипонимов, Прочее.

Запуск:
    python scripts/analyze_dataset_tokens.py output/dataset/validate_10.jsonl
    python scripts/analyze_dataset_tokens.py output/dataset/train.jsonl --model Qwen/Qwen3.5-2B
    python scripts/analyze_dataset_tokens.py output/dataset/validate_10.jsonl -o tokens_report.txt

Tokenizer ищется локально (без сети): --tokenizer-json, путь к модели,
.cache/huggingface в корне репозитория, затем ~/.cache/huggingface.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, TextIO

ROOT = Path(__file__).resolve().parent.parent
REPO_ROOT = ROOT.parent
sys.path.insert(0, str(ROOT))

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

DEFINITION_PREFIX = "**Определение:**"
WORDS_PREFIX = "**Слова:**"
HYPONYMS_RE = re.compile(r"^\*\*Гипоним[ыа](?:\s*\(\d+\))?:\*\*")
HYPONYMS_NONE_PREFIX = "**Гипонимов:**"
SYNSET_HEADER_RE = re.compile(r"^### \d+\. (.+?) `(\S+)`\s*$")
FOUND_HYPONYMS_RE = re.compile(r"^\*\*Найдено гипонимов: (\d+)\*\*")


@dataclass
class TokenBreakdown:
    system: int = 0
    user: int = 0
    tool_call_args: int = 0
    tool_total: int = 0
    tool_definition: int = 0
    tool_words: int = 0
    tool_hyponyms: int = 0
    tool_names: int = 0
    tool_ids: int = 0
    tool_hyponym_totals: int = 0
    tool_other: int = 0
    assistant_final: int = 0

    @property
    def total(self) -> int:
        return (
            self.system
            + self.user
            + self.tool_call_args
            + self.tool_total
            + self.assistant_final
        )

    @property
    def assistant_tool_calls(self) -> int:
        """Суммарно: ответы get_hyponyms (role=tool)."""
        return self.tool_total

    def as_dict(self) -> dict[str, int]:
        return {
            "system": self.system,
            "user": self.user,
            "tool_call_args": self.tool_call_args,
            "assistant_tool_calls": self.assistant_tool_calls,
            "assistant_tool_calls:Определение": self.tool_definition,
            "assistant_tool_calls:Слова": self.tool_words,
            "assistant_tool_calls:Гипонимы": self.tool_hyponyms,
            "assistant_tool_calls:Имена": self.tool_names,
            "assistant_tool_calls:ID синсетов": self.tool_ids,
            "assistant_tool_calls:всего_гипонимов": self.tool_hyponym_totals,
            "assistant_tool_calls:Прочее": self.tool_other,
            "assistant_final": self.assistant_final,
            "total": self.total,
        }


def load_records(path: Path) -> list[dict]:
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []

    if path.suffix == ".jsonl":
        return [json.loads(line) for line in text.splitlines() if line.strip()]

    data = json.loads(text)
    if isinstance(data, list):
        return data
    if isinstance(data, dict) and "messages" in data:
        return [data]
    raise ValueError(f"Неизвестный формат файла: {path}")


def parse_tool_sections(content: str) -> dict[str, str]:
    """Разбивает markdown-ответ get_hyponyms на секции."""
    parts: dict[str, list[str]] = {
        "definition": [],
        "words": [],
        "hyponyms": [],
        "names": [],
        "ids": [],
        "totals": [],
        "other": [],
    }
    current = "other"

    def flush_line(line: str, section: str) -> None:
        if line:
            parts[section].append(line)

    for line in content.split("\n"):
        if (m := SYNSET_HEADER_RE.match(line)):
            flush_line(m.group(1).strip(), "names")
            flush_line(m.group(2).strip(), "ids")
            current = "other"
        elif (m := FOUND_HYPONYMS_RE.match(line)):
            flush_line(m.group(1), "totals")
            current = "other"
        elif line.startswith(DEFINITION_PREFIX):
            flush_line(line[len(DEFINITION_PREFIX) :].strip(), "definition")
            current = "definition"
        elif line.startswith(WORDS_PREFIX):
            flush_line(line[len(WORDS_PREFIX) :].strip(), "words")
            current = "words"
        elif (m := HYPONYMS_RE.match(line)) or line.startswith(HYPONYMS_NONE_PREFIX):
            if m:
                count_match = re.search(r"\((\d+)\)", line)
                if count_match:
                    flush_line(count_match.group(1), "totals")
                elif line.startswith(HYPONYMS_NONE_PREFIX):
                    flush_line("0", "totals")
                rest = line[m.end() :].strip()
            else:
                flush_line("0", "totals")
                rest = line[len(HYPONYMS_NONE_PREFIX) :].strip()
            flush_line(rest, "hyponyms")
            current = "hyponyms"
        elif line.strip() in ("---", "") or line.startswith("### "):
            flush_line(line, "other")
            current = "other"
        else:
            flush_line(line, current)

    return {key: "\n".join(lines).strip() for key, lines in parts.items()}


def _hf_hub_cache_roots() -> list[Path]:
    roots: list[Path] = []
    for base in (REPO_ROOT, Path.home()):
        cache = base / ".cache" / "huggingface" / "hub"
        if cache.is_dir() and cache not in roots:
            roots.append(cache)
    return roots


def find_local_tokenizer_json(model_id: str) -> Path | None:
    """Ищет tokenizer.json локально, без обращения к HuggingFace Hub."""
    explicit = Path(model_id)
    if explicit.is_file() and explicit.name == "tokenizer.json":
        return explicit.resolve()
    if explicit.is_dir():
        candidate = explicit / "tokenizer.json"
        if candidate.is_file():
            return candidate.resolve()

    repo_key = f"models--{model_id.replace('/', '--')}"
    for cache_root in _hf_hub_cache_roots():
        snapshots = cache_root / repo_key / "snapshots"
        if not snapshots.is_dir():
            continue
        snaps = sorted(snapshots.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True)
        for snap in snaps:
            candidate = snap / "tokenizer.json"
            if candidate.is_file():
                return candidate.resolve()
    return None


def _wrap_tokenizers_file(path: Path):
    from tokenizers import Tokenizer

    tok = Tokenizer.from_file(str(path))

    class _Wrapper:
        def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
            return tok.encode(text).ids

    return _Wrapper()


def _wrap_transformers_local(model_path: Path):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(
        str(model_path),
        trust_remote_code=True,
        local_files_only=True,
    )


def load_tokenizer(model_id: str, tokenizer_json: str | None):
    """Загружает tokenizer только из локальных файлов (без сети)."""
    candidates: list[Path] = []
    if tokenizer_json:
        candidates.append(Path(tokenizer_json))
    found = find_local_tokenizer_json(model_id)
    if found:
        candidates.append(found)

    seen: set[Path] = set()
    for path in candidates:
        path = path.resolve()
        if path in seen:
            continue
        seen.add(path)

        json_path = path if path.name == "tokenizer.json" else path / "tokenizer.json"
        if not json_path.is_file():
            print(f"WARNING: tokenizer.json не найден: {json_path}")
            continue

        model_dir = json_path.parent

        try:
            tok = _wrap_tokenizers_file(json_path)
            print(f"Tokenizer: tokenizers ({json_path})")
            return tok, False
        except ImportError:
            pass
        except Exception as exc:
            print(f"WARNING: tokenizers не загрузил {json_path}: {exc}")

        try:
            tok = _wrap_transformers_local(model_dir)
            print(f"Tokenizer: transformers local ({model_dir})")
            return tok, False
        except ImportError:
            pass
        except Exception as exc:
            print(f"WARNING: transformers local не загрузил {model_dir}: {exc}")

    print(
        "WARNING: локальный tokenizer не найден — "
        "используется приближение по числу символов.\n"
        f"  Искали model={model_id!r}, cache roots: {_hf_hub_cache_roots()}\n"
        "  Укажите явно: --tokenizer-json путь/к/tokenizer.json"
    )

    class _CharFallback:
        def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
            return list(text)

    return _CharFallback(), True


def count_tokens(tokenizer, text: str) -> int:
    if not text:
        return 0
    encoded = tokenizer.encode(text, add_special_tokens=False)
    return len(encoded)


def serialize_tool_calls(tool_calls: list | None) -> str:
    if not tool_calls:
        return ""
    return json.dumps(tool_calls, ensure_ascii=False)


def apply_tool_breakdown(
    breakdown: TokenBreakdown,
    tokenizer,
    content: str,
    sections: dict[str, str],
) -> None:
    tool_tokens = count_tokens(tokenizer, content)
    breakdown.tool_total += tool_tokens
    d = count_tokens(tokenizer, sections.get("definition", ""))
    w = count_tokens(tokenizer, sections.get("words", ""))
    h = count_tokens(tokenizer, sections.get("hyponyms", ""))
    n = count_tokens(tokenizer, sections.get("names", ""))
    i = count_tokens(tokenizer, sections.get("ids", ""))
    t = count_tokens(tokenizer, sections.get("totals", ""))
    breakdown.tool_definition += d
    breakdown.tool_words += w
    breakdown.tool_hyponyms += h
    breakdown.tool_names += n
    breakdown.tool_ids += i
    breakdown.tool_hyponym_totals += t
    breakdown.tool_other += max(0, tool_tokens - d - w - h - n - i - t)


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
            apply_tool_breakdown(breakdown, tokenizer, content, parse_tool_sections(content))

    return breakdown


def pct(part: int, total: int) -> float:
    if total == 0:
        return 0.0
    return 100.0 * part / total


def format_cell(tokens: int, total: int) -> str:
    return f"{tokens:>6} ({pct(tokens, total):5.1f}%)"


def print_record_table(
    index: int,
    label: str,
    breakdown: TokenBreakdown,
    *,
    out: TextIO | None = None,
) -> None:
    sink = out or sys.stdout
    total = breakdown.total
    rows = [
        ("system", breakdown.system),
        ("user", breakdown.user),
        ("tool_call_args", breakdown.tool_call_args),
        ("assistant_tool_calls", breakdown.assistant_tool_calls),
        ("  └ Определение", breakdown.tool_definition),
        ("  └ Слова", breakdown.tool_words),
        ("  └ Гипонимы", breakdown.tool_hyponyms),
        ("  └ Имена", breakdown.tool_names),
        ("  └ ID синсетов", breakdown.tool_ids),
        ("  └ всего_гипонимов", breakdown.tool_hyponym_totals),
        ("  └ Прочее", breakdown.tool_other),
        ("assistant_final", breakdown.assistant_final),
    ]

    print(f"\n{'=' * 72}", file=sink)
    print(f"Запись #{index + 1}: {label}", file=sink)
    print(f"{'=' * 72}", file=sink)
    print(f"{'Категория':<22} {'Токены':>8}  {'Доля':>8}", file=sink)
    print(f"{'-' * 22} {'-' * 8}  {'-' * 8}", file=sink)
    for name, tokens in rows:
        print(f"{name:<22} {tokens:>8}  {pct(tokens, total):>7.1f}%", file=sink)
    print(f"{'-' * 22} {'-' * 8}  {'-' * 8}", file=sink)
    print(f"{'ИТОГО':<22} {total:>8}  {'100.0':>7}%", file=sink)


@dataclass
class SummaryStats:
    values: dict[str, list[int]] = field(default_factory=dict)

    def add(self, breakdown: TokenBreakdown) -> None:
        for key, val in breakdown.as_dict().items():
            self.values.setdefault(key, []).append(val)

    def print_summary(self, *, out: TextIO | None = None) -> None:
        sink = out or sys.stdout
        print(f"\n{'=' * 72}", file=sink)
        print("СВОДКА (min / max / mean) по всем записям", file=sink)
        print(f"{'=' * 72}", file=sink)
        print(f"{'Категория':<22} {'min':>8} {'max':>8} {'mean':>10}", file=sink)
        print(f"{'-' * 22} {'-' * 8} {'-' * 8} {'-' * 10}", file=sink)

        order = [
            "system",
            "user",
            "tool_call_args",
            "assistant_tool_calls",
            "assistant_tool_calls:Определение",
            "assistant_tool_calls:Слова",
            "assistant_tool_calls:Гипонимы",
            "assistant_tool_calls:Имена",
            "assistant_tool_calls:ID синсетов",
            "assistant_tool_calls:всего_гипонимов",
            "assistant_tool_calls:Прочее",
            "assistant_final",
            "total",
        ]
        for key in order:
            vals = self.values.get(key, [])
            if not vals:
                continue
            label = key.replace("assistant_tool_calls:", "  └ ")
            print(
                f"{label:<22} {min(vals):>8} {max(vals):>8} {sum(vals) / len(vals):>10.1f}",
                file=sink,
            )

        print(f"\n{'Категория':<22} {'min %':>8} {'max %':>8} {'mean %':>10}", file=sink)
        print(f"{'-' * 22} {'-' * 8} {'-' * 8} {'-' * 10}", file=sink)
        totals = self.values.get("total", [])
        for key in order[:-1]:
            vals = self.values.get(key, [])
            if not vals or not totals:
                continue
            percents = [pct(v, t) for v, t in zip(vals, totals)]
            label = key.replace("assistant_tool_calls:", "  └ ")
            print(
                f"{label:<22} {min(percents):>7.1f}% {max(percents):>7.1f}% "
                f"{sum(percents) / len(percents):>9.1f}%",
                file=sink,
            )


def record_label(record: dict, index: int) -> str:
    messages = record.get("messages", [])
    for msg in reversed(messages):
        if msg.get("role") == "assistant" and msg.get("content"):
            preview = msg["content"][:60].replace("\n", " ")
            return preview
    return f"record_{index}"


def write_analysis_report(
    records: list[dict],
    tokenizer,
    analyze_fn: Callable[[list[dict], object], TokenBreakdown],
    *,
    out: TextIO | None = None,
    preamble: list[str] | None = None,
) -> SummaryStats:
    """Пишет таблицы по записям и сводку в out (по умолчанию stdout)."""
    sink = out or sys.stdout
    for line in preamble or []:
        print(line, file=sink)

    summary = SummaryStats()
    for i, record in enumerate(records):
        messages = record.get("messages", [])
        breakdown = analyze_fn(messages, tokenizer)
        summary.add(breakdown)
        print_record_table(i, record_label(record, i), breakdown, out=sink)

    summary.print_summary(out=sink)
    return summary


def resolve_output_path(path: str) -> Path:
    out_path = Path(path)
    if not out_path.is_absolute():
        out_path = ROOT / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Анализ токенов SFT-датасета")
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
            "(!) Абсолютные числа токенов приблизительны; для точных — pip install transformers"
        )
    preamble.append(f"Записей: {len(records)}")

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
