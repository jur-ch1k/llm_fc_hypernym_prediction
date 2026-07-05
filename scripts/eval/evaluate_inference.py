"""Оценка прогона inference: P@1/P@3, MAP@3, MRR@3."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

SCRIPTS = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPTS.parent.parent


def compute_p(true: list[str], pred: list[str], k: int = 1) -> float:
    if not true:
        return 0.0
    k = min(k, len(true))
    top_k_pred = pred[:k]
    num_relevant = np.sum(np.isin(top_k_pred, true))
    return float(num_relevant / k)


def compute_ap(actual: list[list[str]], predicted: list[str], k: int = 10) -> float:
    """Average Precision@k (как в taxoenrich.utils)."""
    if not actual:
        return 0.0

    predicted = predicted[:k]

    score = 0.0
    num_hits = 0.0
    already_predicted: set[str] = set()
    skipped = 0
    for i, p in enumerate(predicted):
        if p in already_predicted:
            skipped += 1
            continue
        for parents in actual:
            if p in parents:
                num_hits += 1.0
                score += num_hits / (i + 1.0 - skipped)
                already_predicted.update(parents)
                break

    return score / min(len(actual), k)


def compute_rr(true: list[str], predicted: list[str], k: int = 10) -> float:
    """Reciprocal rank@k (как в taxoenrich.utils)."""
    for i, synset in enumerate(predicted[:k]):
        if synset in true:
            return 1.0 / (i + 1.0)
    return 0.0


def path_predictions(selected_synsets: list[dict[str, Any]]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for item in reversed(selected_synsets):
        node_id = item.get("synset_id")
        if node_id and node_id not in seen:
            seen.add(node_id)
            result.append(node_id)
    return result


def load_run_record(path: Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        return data[0]
    return data


def find_latest_run(inference_root: Path) -> Path:
    runs = sorted(
        (p for p in inference_root.glob("run_*") if p.is_dir()),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not runs:
        raise FileNotFoundError(f"Прогоны не найдены в {inference_root}")
    return runs[0]


def evaluate_run(run_dir: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    result_files = sorted(
        p for p in run_dir.glob("*.json") if p.name != "run_meta.json"
    )
    if not result_files:
        raise FileNotFoundError(f"JSON-результаты не найдены в {run_dir}")

    per_example: list[dict[str, Any]] = []
    ok_count = 0
    failed_count = 0

    for path in result_files:
        record = load_run_record(path)
        word = record.get("target_word", path.stem)
        target_ids = record.get("target_ids") or []
        selected = record.get("selected_synsets") or []
        status = record.get("status", "unknown")
        predicted = path_predictions(selected)

        if status == "ok":
            ok_count += 1
        else:
            failed_count += 1

        metrics = {
            "P@1": compute_p(target_ids, predicted, k=1),
            "MAP@1": compute_ap([target_ids], predicted, k=1),
            "MRR@1": compute_rr(target_ids, predicted, k=1),
            "P@3": compute_p(target_ids, predicted, k=3),
            "MAP@3": compute_ap([target_ids], predicted, k=3),
            "MRR@3": compute_rr(target_ids, predicted, k=3),
        }

        per_example.append(
            {
                "word": word,
                "target_ids": target_ids,
                "predicted": predicted,
                "final_result": record.get("final_result"),
                "status": status,
                **metrics,
            }
        )

    metric_keys = ["P@1", "MAP@1", "MRR@1", "P@3", "MAP@3", "MRR@3"]
    summary: dict[str, Any] = {
        "run_dir": str(run_dir),
        "num_examples": len(per_example),
        "ok": ok_count,
        "failed": failed_count,
    }
    for key in metric_keys:
        summary[key] = float(np.mean([row[key] for row in per_example])) if per_example else 0.0

    return per_example, summary


def write_metrics(run_dir: Path, per_example: list[dict], summary: dict[str, Any]) -> None:
    with open(run_dir / "metrics_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    header = [
        "word",
        "target_ids",
        "predicted",
        "final_result",
        "status",
        "P@1",
        "MAP@1",
        "MRR@1",
        "P@3",
        "MAP@3",
        "MRR@3",
    ]
    lines = ["\t".join(header)]
    for row in per_example:
        lines.append(
            "\t".join(
                [
                    str(row["word"]),
                    json.dumps(row["target_ids"], ensure_ascii=False),
                    json.dumps(row["predicted"], ensure_ascii=False),
                    str(row.get("final_result") or ""),
                    str(row["status"]),
                    f"{row['P@1']:.4f}",
                    f"{row['MAP@1']:.4f}",
                    f"{row['MRR@1']:.4f}",
                    f"{row['P@3']:.4f}",
                    f"{row['MAP@3']:.4f}",
                    f"{row['MRR@3']:.4f}",
                ]
            )
        )

    with open(run_dir / "metrics_per_example.tsv", "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Оценка прогона inference")
    parser.add_argument(
        "--run",
        default=None,
        help="Путь к output/inference/run_<ts> (по умолчанию — последний)",
    )
    parser.add_argument(
        "--inference-root",
        default=str(PROJECT_ROOT / "output" / "inference"),
        help="Корневая папка inference-прогонов",
    )
    args = parser.parse_args()

    if args.run:
        run_dir = Path(args.run)
    else:
        run_dir = find_latest_run(Path(args.inference_root))

    if not run_dir.is_dir():
        raise FileNotFoundError(f"Папка прогона не найдена: {run_dir}")

    per_example, summary = evaluate_run(run_dir)
    write_metrics(run_dir, per_example, summary)

    print(f"Run: {run_dir}")
    print(f"Examples: {summary['num_examples']} (ok={summary['ok']}, failed={summary['failed']})")
    for key in ["P@1", "P@3", "MAP@3", "MRR@3"]:
        print(f"  {key}: {summary[key]:.4f}")
    print(f"Wrote {run_dir / 'metrics_summary.json'}")
    print(f"Wrote {run_dir / 'metrics_per_example.tsv'}")


if __name__ == "__main__":
    main()
