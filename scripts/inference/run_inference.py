"""Локальный инференс LoRA-адаптера с agent loop и TSV-тулзами."""

from __future__ import annotations

import argparse
import gc
import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

SCRIPTS = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPTS.parent.parent
REPO_ROOT = PROJECT_ROOT.parent
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "output"
DEFAULT_ADAPTER_ROOT = REPO_ROOT / "output"
HF_CACHE_ROOT = REPO_ROOT / ".cache" / "huggingface"
HF_HUB_CACHE = HF_CACHE_ROOT / "hub"

# Только локальный HF-кэш в репозитории (не ~/.cache на диске C:).
# os.environ["HF_HOME"] = str(HF_CACHE_ROOT)
# os.environ["HF_HUB_CACHE"] = str(HF_HUB_CACHE)

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(PROJECT_ROOT))

from sft_pipeline.config import CORPUS_DIR, WORDNET_PATH  # noqa: E402
from sft_pipeline.data_loaders import iter_dataset_rows, load_context  # noqa: E402
from sft_pipeline.prompts import load_system_prompt  # noqa: E402
from sft_pipeline.tools_runtime import execute_tool, is_virtual_root_token  # noqa: E402
from taxoenrich.core import RuWordNet  # noqa: E402
from utils.tools import hyponym_only  # noqa: E402

TOOLS = hyponym_only

TOOL_CALL_RE = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL | re.IGNORECASE)
FUNCTION_RE = re.compile(r"<function=([^>\n]+)>", re.DOTALL | re.IGNORECASE)
PARAM_RE = re.compile(
    r"<parameter=([^>\n]+)>\s*(.*?)\s*</parameter>",
    re.DOTALL | re.IGNORECASE,
)
JSON_TOOL_RE = re.compile(
    r'\{\s*"name"\s*:\s*"([^"]+)"\s*,\s*"arguments"\s*:\s*(\{.*?\})\s*\}',
    re.DOTALL,
)
FINAL_RE = re.compile(
    r"(hyponym of|include in|not_found)",
    re.IGNORECASE,
)
SYNSET_ID_RE = re.compile(r"\d{1,6}-[ANV]")
SPECIAL_TOKEN_RE = re.compile(
    r"<\|[^|>]+?\|>|<\|endoftext\|>|<think>\s*</think>",
    re.DOTALL | re.IGNORECASE,
)

MAX_NEW_TOKENS = 128
GENERATION_STOP_STRINGS = ["</tool_call>"]


def truncate_at_first_tool_call(text: str) -> str:
    match = re.search(r"</tool_call>", text, re.IGNORECASE)
    if match:
        return text[: match.end()].strip()
    return text.strip()


def clean_final_response(text: str) -> str:
    return SPECIAL_TOKEN_RE.sub("", text).strip()


def sanitize_assistant_response(text: str) -> str:
    """Обрезка для agent loop: tool call — до </tool_call>, финал — без служебных токенов."""
    if TOOL_CALL_RE.search(text):
        return truncate_at_first_tool_call(text)
    return clean_final_response(text)


def parse_qwen_tool_call(text: str) -> tuple[str, dict[str, Any]] | None:
    """Парсит Qwen XML tool-call; fallback — JSON внутри <tool_call>."""
    match = TOOL_CALL_RE.search(text)
    if not match:
        return None

    body = match.group(1).strip()
    fn_match = FUNCTION_RE.search(body)
    if fn_match:
        function_name = fn_match.group(1).strip()
        args: dict[str, Any] = {}
        for key, value in PARAM_RE.findall(body):
            raw = value.strip()
            if key == "node_id" and is_virtual_root_token(raw):
                args[key] = "null"
            else:
                args[key.strip()] = raw
        return function_name, args

    json_match = JSON_TOOL_RE.search(body)
    if json_match:
        function_name = json_match.group(1)
        args = json.loads(json_match.group(2))
        return function_name, args

    return None


def is_final_response(text: str) -> bool:
    cleaned = text.strip()
    if not cleaned:
        return False
    if TOOL_CALL_RE.search(cleaned):
        return False
    return bool(FINAL_RE.search(cleaned))


def extract_synset_id(text: str) -> str | None:
    matches = SYNSET_ID_RE.findall(text)
    return matches[-1] if matches else None


def normalize_node_id(node_id: str | None) -> str | None:
    if node_id is None:
        return None
    raw = str(node_id).strip()
    if is_virtual_root_token(raw):
        return "null"
    return raw


def append_final_synset(selected_synsets: list[dict[str, Any]], final_result: str) -> None:
    final_synset = extract_synset_id(final_result)
    if not final_synset:
        return
    if any(item.get("synset_id") == final_synset for item in selected_synsets):
        return
    selected_synsets.append(
        {
            "synset_id": final_synset,
            "function": "final_result",
            "args": {"node_id": final_synset},
            "timestamp": datetime.now().isoformat(),
        }
    )


def load_model_and_tokenizer(adapter_dir: Path):
    with open(adapter_dir / "adapter_config.json", encoding="utf-8") as f:
        adapter_cfg = json.load(f)
    base_model_id = adapter_cfg["base_model_name_or_path"]

    tokenizer = AutoTokenizer.from_pretrained(
        base_model_id,
        trust_remote_code=True,
        cache_dir=str(HF_HUB_CACHE),
    )
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        cache_dir=str(HF_HUB_CACHE),
    )
    model = PeftModel.from_pretrained(base_model, str(adapter_dir))
    model.eval()
    return model, tokenizer, base_model_id


def generate_response(
    model,
    tokenizer,
    messages: list[dict[str, Any]],
    *,
    temperature: float,
    max_new_tokens: int = MAX_NEW_TOKENS,
    stop_strings: list[str] | None = GENERATION_STOP_STRINGS,
) -> str:
    prompt = tokenizer.apply_chat_template(
        messages,
        tools=TOOLS,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    gen_kwargs: dict[str, Any] = {
        "max_new_tokens": max_new_tokens,
        "do_sample": temperature > 0,
        "pad_token_id": tokenizer.pad_token_id or tokenizer.eos_token_id,
    }
    if temperature > 0:
        gen_kwargs["temperature"] = temperature
    if stop_strings:
        gen_kwargs["stop_strings"] = stop_strings
        gen_kwargs["tokenizer"] = tokenizer

    with torch.no_grad():
        output_ids = model.generate(**inputs, **gen_kwargs)

    new_tokens = output_ids[0, inputs["input_ids"].shape[1] :]
    return tokenizer.decode(new_tokens, skip_special_tokens=False).strip()


def run_agent_loop(
    model,
    tokenizer,
    wn: RuWordNet,
    context_text: str,
    *,
    max_iters: int,
    temperature: float,
) -> dict[str, Any]:
    system_prompt = load_system_prompt()
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": context_text},
    ]
    selected_synsets: list[dict[str, Any]] = []
    visited: set[str | None] = set()
    final_result: str | None = None
    status = "failed"
    tool_call_count = 0
    iteration = 0

    for iteration in range(1, max_iters + 1):
        raw_response = generate_response(
            model,
            tokenizer,
            messages,
            temperature=temperature,
        )
        if not raw_response:
            break

        response_text = sanitize_assistant_response(raw_response)

        if is_final_response(response_text):
            final_result = clean_final_response(response_text)
            messages.append(
                {
                    "role": "assistant",
                    "content": final_result,
                    "raw_content": raw_response,
                }
            )
            append_final_synset(selected_synsets, final_result)
            status = "ok"
            break

        parsed = parse_qwen_tool_call(response_text)
        if parsed is None:
            if FINAL_RE.search(response_text):
                final_result = clean_final_response(response_text)
                messages.append(
                    {
                        "role": "assistant",
                        "content": final_result,
                        "raw_content": raw_response,
                    }
                )
                append_final_synset(selected_synsets, final_result)
                status = "ok"
                break
            break

        function_name, args = parsed
        if function_name != "get_hyponyms":
            break

        node_id = normalize_node_id(args.get("node_id"))
        if node_id in visited:
            break
        visited.add(node_id)

        try:
            tool_result = execute_tool(
                wn,
                function_name,
                node_id,
                max_words=3,
                max_hyponyms=5,
            )
        except ValueError:
            break

        if node_id and node_id != "null":
            selected_synsets.append(
                {
                    "synset_id": node_id,
                    "function": function_name,
                    "args": {"node_id": node_id},
                    "timestamp": datetime.now().isoformat(),
                }
            )

        call_id = f"call_{tool_call_count}"
        tool_call_count += 1
        messages.append(
            {
                "role": "assistant",
                "content": response_text,
                "raw_content": raw_response,
            }
        )
        messages.append(
            {
                "role": "tool",
                "tool_call_id": call_id,
                "content": tool_result,
            }
        )
    else:
        if final_result is None:
            final_result = "Достигнут лимит итераций"

    return {
        "final_result": final_result,
        "selected_synsets": selected_synsets,
        "total_selections": len(selected_synsets),
        "iterations": iteration,
        "status": status,
        "messages": messages,
    }


def save_result(path: Path, payload: dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump([payload], f, ensure_ascii=False, indent=2)


def extract_final_answer(messages: list[dict[str, Any]]) -> str | None:
    """Последний ответ модели (assistant), без синтетических final_result."""
    for msg in reversed(messages):
        if msg.get("role") == "assistant":
            content = msg.get("content")
            if content is not None and str(content).strip():
                return str(content).strip()
    return None


def _message_direction(role: str) -> str:
    if role == "assistant":
        return "получено"
    return "отправлено"


def format_dialog_markdown(
    word: str,
    row_index: int,
    messages: list[dict[str, Any]],
    *,
    status: str,
    final_result: str | None,
) -> str:
    lines = [
        f"# Диалог: {word}",
        "",
        f"- row_index: {row_index}",
        f"- status: {status}",
        f"- final_result: {final_result or ''}",
        "",
    ]
    for i, msg in enumerate(messages, start=1):
        role = msg.get("role", "unknown")
        direction = _message_direction(role)
        lines.append(f"## [{i}] {role} — {direction}")
        lines.append("")
        if role == "assistant" and msg.get("raw_content") is not None:
            content = msg["raw_content"]
        else:
            content = msg.get("content", "")
        if content is None:
            content = ""
        content = str(content)
        if role == "assistant":
            lines.append("```xml")
            lines.append(content)
            lines.append("```")
        else:
            lines.append(content)
        if role == "tool" and msg.get("tool_call_id"):
            lines.append("")
            lines.append(f"tool_call_id: {msg['tool_call_id']}")
        lines.append("")
        lines.append("---")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def save_verbose_dialog(
    path: Path,
    word: str,
    row_index: int,
    messages: list[dict[str, Any]],
    *,
    status: str,
    final_result: str | None,
) -> None:
    path.write_text(
        format_dialog_markdown(
            word,
            row_index,
            messages,
            status=status,
            final_result=final_result,
        ),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="LoRA inference с agent loop")
    parser.add_argument("--adapter", required=True, help="Имя папки адаптера в output/")
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--max-iters", type=int, default=15)
    parser.add_argument(
        "--seed-dir",
        default="datasets/seed_42",
        help="Папка с test_dataset.tsv относительно llm_fc_hypernym_prediction",
    )
    parser.add_argument(
        "--out",
        default=str(PROJECT_ROOT / "output" / "inference"),
        help="Корневая папка для прогонов inference",
    )
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument(
        "--adapter-root",
        default=str(DEFAULT_ADAPTER_ROOT),
        help="Корень с адаптерами (по умолчанию S:/hw/aspa/output)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help=(
            "Сохранить полный диалог инференса в .md для каждого слова "
            "(сырой текст отправленных и полученных сообщений в папке прогона)"
        ),
    )
    parser.add_argument(
        "--add-final",
        action="store_true",
        help=(
            "Добавить в JSON поле final_answer с последним ответом модели; "
            "не используется при подсчёте метрик"
        ),
    )
    args = parser.parse_args()

    adapter_dir = Path(args.adapter_root) / args.adapter
    if not adapter_dir.exists():
        raise FileNotFoundError(f"Адаптер не найден: {adapter_dir}")

    dataset_path = PROJECT_ROOT / args.seed_dir / "test_dataset.tsv"
    if not dataset_path.exists():
        raise FileNotFoundError(f"Test dataset не найден: {dataset_path}")

    run_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(args.out) / f"run_{run_ts}"
    run_dir.mkdir(parents=True, exist_ok=True)

    print(f"HF cache: {HF_HUB_CACHE}")
    print(f"Loading adapter from {adapter_dir}")
    model, tokenizer, base_model_id = load_model_and_tokenizer(adapter_dir)

    wordnet_path = str(PROJECT_ROOT / WORDNET_PATH)
    wn = RuWordNet(wordnet_path)
    rows = iter_dataset_rows(args.limit, str(dataset_path))

    run_meta = {
        "adapter": args.adapter,
        "adapter_dir": str(adapter_dir),
        "base_model": base_model_id,
        "limit": args.limit,
        "max_iters": args.max_iters,
        "temperature": args.temperature,
        "seed_dir": str(dataset_path),
        "verbose": args.verbose,
        "add_final": args.add_final,
        "timestamp": datetime.now().isoformat(),
        "run_dir": str(run_dir),
    }
    with open(run_dir / "run_meta.json", "w", encoding="utf-8") as f:
        json.dump(run_meta, f, ensure_ascii=False, indent=2)

    ok_count = 0
    failed_count = 0

    for row in rows:
        context_file = row.context_files[0]
        context_text = load_context(context_file, str(PROJECT_ROOT / CORPUS_DIR))
        word = row.word
        print(f"[{row.row_index}] {word}")

        out_stem = f"{row.row_index}_{word}"

        if not context_text:
            payload = {
                "target_word": word,
                "target_ids": row.target_ids,
                "selected_synsets": [],
                "final_result": None,
                "status": "failed",
                "iterations": 0,
                "error": f"Контекст не загружен: {context_file}",
            }
            if args.add_final:
                payload["final_answer"] = None
            failed_count += 1
            if args.verbose:
                save_verbose_dialog(
                    run_dir / f"{out_stem}.md",
                    word,
                    row.row_index,
                    [
                        {
                            "role": "system",
                            "content": f"Ошибка: контекст не загружен ({context_file})",
                        }
                    ],
                    status="failed",
                    final_result=None,
                )
        else:
            loop_result = run_agent_loop(
                model,
                tokenizer,
                wn,
                context_text,
                max_iters=args.max_iters,
                temperature=args.temperature,
            )
            payload = {
                "target_word": word,
                "target_ids": row.target_ids,
                "selected_synsets": loop_result["selected_synsets"],
                "final_result": loop_result["final_result"],
                "status": loop_result["status"],
                "iterations": loop_result["iterations"],
                "total_selections": loop_result["total_selections"],
            }
            if args.add_final:
                payload["final_answer"] = extract_final_answer(loop_result["messages"])
            if loop_result["status"] == "ok":
                ok_count += 1
            else:
                failed_count += 1
            if args.verbose:
                save_verbose_dialog(
                    run_dir / f"{out_stem}.md",
                    word,
                    row.row_index,
                    loop_result["messages"],
                    status=loop_result["status"],
                    final_result=loop_result["final_result"],
                )

        save_result(run_dir / f"{out_stem}.json", payload)

    run_meta["ok"] = ok_count
    run_meta["failed"] = failed_count
    with open(run_dir / "run_meta.json", "w", encoding="utf-8") as f:
        json.dump(run_meta, f, ensure_ascii=False, indent=2)

    print(f"Done: {run_dir} (ok={ok_count}, failed={failed_count})")

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
