"""
Построение одного SFT-примера из trajectory JSON.

Запуск:
    python scripts/build_sample.py --trajectory output/trajectories/АБСЕНТЕИЗМ_r0_147309-N.json
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from taxoenrich.core import RuWordNet

from sft_pipeline.config import OUTPUT_DATASET, WORDNET_PATH
from sft_pipeline.data_loaders import load_context
from sft_pipeline.messages import messages_to_jsonl_record, trajectory_to_messages
from sft_pipeline.trajectory import load_trajectory

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Построить один SFT-пример из trajectory")
    parser.add_argument("--trajectory", required=True, help="Путь к trajectory JSON")
    parser.add_argument(
        "--out",
        default=str(Path(OUTPUT_DATASET) / "sample.jsonl"),
        help="Выходной JSONL",
    )
    args = parser.parse_args()

    traj = load_trajectory(args.trajectory)
    if traj.status != "ok":
        log.error("Trajectory status=%s, ожидался ok", traj.status)
        sys.exit(1)

    context_text = load_context(traj.context_file)
    if not context_text:
        log.error("Не удалось загрузить контекст: %s", traj.context_file)
        sys.exit(1)

    log.info("Загрузка RuWordNet...")
    wn = RuWordNet(WORDNET_PATH)

    messages = trajectory_to_messages(wn, traj, context_text)
    record = messages_to_jsonl_record(messages)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

    log.info("Записано: %s (%d messages)", out_path, len(messages))


if __name__ == "__main__":
    main()
