import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from taxoenrich.core import RuWordNet

from .config import MAX_HYPONYMS, MAX_WORDS, VIRTUAL_ROOT
from .data_loaders import DatasetRow, resolve_context_file
from .graph import (
    collect_roots,
    find_ancestor_roots,
    find_shortest_downward_path,
    pick_best_root,
    virtual_root_path,
)
from .tools_runtime import execute_tool

log = logging.getLogger(__name__)


@dataclass
class TrajectoryStep:
    function: str
    node_id: str | None
    next_node_id: str
    tool_result: str


@dataclass
class TrajectoryFinal:
    decision: str
    synset_id: str
    synset_name: str
    content: str


@dataclass
class Trajectory:
    word: str
    row_index: int
    target_id: str
    target_name: str
    context_file: str
    root_id: str | None
    status: str
    path: list[str] = field(default_factory=list)
    steps: list[TrajectoryStep] = field(default_factory=list)
    final: TrajectoryFinal | None = None


def trajectory_filename(word: str, row_index: int, target_id: str) -> str:
    return f"{word}_r{row_index}_{target_id}.json"


def _step_node_id(path_node: str) -> str | None:
    return None if path_node == VIRTUAL_ROOT else path_node


def build_trajectory(
    wn: RuWordNet,
    row: DatasetRow,
    target_id: str,
    target_name: str,
    *,
    max_words: int = MAX_WORDS,
    max_hyponyms: int = MAX_HYPONYMS,
) -> Trajectory:
    context_file = resolve_context_file(row)
    base = Trajectory(
        word=row.word,
        row_index=row.row_index,
        target_id=target_id,
        target_name=target_name,
        context_file=context_file,
        root_id=None,
        status="invalid_target",
    )

    if target_id not in wn.synsets:
        log.error(
            "invalid_target: word=%s row=%d target=%s",
            row.word,
            row.row_index,
            target_id,
        )
        return base

    roots = collect_roots(wn)
    ancestor_roots = find_ancestor_roots(wn, target_id, roots)
    if not ancestor_roots:
        log.error(
            "no_path: word=%s row=%d target=%s reason=no_ancestor_roots",
            row.word,
            row.row_index,
            target_id,
        )
        return Trajectory(
            word=row.word,
            row_index=row.row_index,
            target_id=target_id,
            target_name=target_name,
            context_file=context_file,
            root_id=None,
            status="no_path",
        )

    root_id = pick_best_root(wn, target_id, ancestor_roots)
    if root_id is None:
        log.error(
            "no_path: word=%s row=%d target=%s reason=no_downward_path_from_roots",
            row.word,
            row.row_index,
            target_id,
        )
        return Trajectory(
            word=row.word,
            row_index=row.row_index,
            target_id=target_id,
            target_name=target_name,
            context_file=context_file,
            root_id=None,
            status="no_path",
        )

    down_path = find_shortest_downward_path(wn, root_id, target_id)
    if down_path is None:
        log.error(
            "no_path: word=%s row=%d target=%s reason=downward_path_reconstruction_failed root=%s",
            row.word,
            row.row_index,
            target_id,
            root_id,
        )
        return Trajectory(
            word=row.word,
            row_index=row.row_index,
            target_id=target_id,
            target_name=target_name,
            context_file=context_file,
            root_id=root_id,
            status="no_path",
        )

    full_path = virtual_root_path(down_path)
    steps: list[TrajectoryStep] = []
    for i in range(len(full_path) - 1):
        node_id = _step_node_id(full_path[i])
        next_node_id = full_path[i + 1]
        tool_result = execute_tool(
            wn,
            "get_hyponyms",
            node_id,
            max_words=max_words,
            max_hyponyms=max_hyponyms,
        )
        steps.append(
            TrajectoryStep(
                function="get_hyponyms",
                node_id=node_id,
                next_node_id=next_node_id,
                tool_result=tool_result,
            )
        )

    synset = wn.synsets[target_id]
    synset_name = synset.synset_name
    final = TrajectoryFinal(
        decision="hyponym_of",
        synset_id=target_id,
        synset_name=synset_name,
        content=f"hyponym of {target_id} ({synset_name})",
    )

    return Trajectory(
        word=row.word,
        row_index=row.row_index,
        target_id=target_id,
        target_name=target_name,
        context_file=context_file,
        root_id=root_id,
        status="ok",
        path=full_path,
        steps=steps,
        final=final,
    )


def _trajectory_to_dict(traj: Trajectory) -> dict[str, Any]:
    data = asdict(traj)
    if traj.final is None:
        data["final"] = None
    return data


def _dict_to_trajectory(data: dict[str, Any]) -> Trajectory:
    final_data = data.get("final")
    final = TrajectoryFinal(**final_data) if final_data else None
    steps = [TrajectoryStep(**s) for s in data.get("steps", [])]
    return Trajectory(
        word=data["word"],
        row_index=data["row_index"],
        target_id=data["target_id"],
        target_name=data["target_name"],
        context_file=data["context_file"],
        root_id=data.get("root_id"),
        status=data["status"],
        path=data.get("path", []),
        steps=steps,
        final=final,
    )


def save_trajectory(traj: Trajectory, out_dir: str | Path) -> Path:
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    filename = trajectory_filename(traj.word, traj.row_index, traj.target_id)
    file_path = out_path / filename
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(_trajectory_to_dict(traj), f, ensure_ascii=False, indent=2)
    return file_path


def load_trajectory(path: str | Path) -> Trajectory:
    with open(path, encoding="utf-8") as f:
        return _dict_to_trajectory(json.load(f))


def build_trajectories_for_row(
    wn: RuWordNet,
    row: DatasetRow,
    out_dir: str | Path,
    *,
    max_words: int = MAX_WORDS,
    max_hyponyms: int = MAX_HYPONYMS,
) -> tuple[int, int, int]:
    """Строит trajectory для каждого target строки TSV. Возвращает (ok, no_path, invalid_target)."""
    ok_count = 0
    no_path_count = 0
    invalid_target_count = 0

    for target_id, target_name in zip(row.target_ids, row.target_names):
        traj = build_trajectory(
            wn,
            row,
            target_id,
            target_name,
            max_words=max_words,
            max_hyponyms=max_hyponyms,
        )
        save_trajectory(traj, out_dir)
        if traj.status == "ok":
            ok_count += 1
        elif traj.status == "no_path":
            no_path_count += 1
        elif traj.status == "invalid_target":
            invalid_target_count += 1

    return ok_count, no_path_count, invalid_target_count
