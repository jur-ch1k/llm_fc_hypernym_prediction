from taxoenrich.core import RuWordNet

from .config import POS, VIRTUAL_ROOT


def collect_roots(wn: RuWordNet, pos: str = POS) -> set[str]:
    """Synsets без hypernym с заданным pos — реальные корни таксономии."""
    roots: set[str] = set()
    for sid, synset in wn.synsets.items():
        if synset.synset_type != pos:
            continue
        if not synset.rels.get("hypernym"):
            roots.add(sid)
    return roots


def find_ancestor_roots(
    wn: RuWordNet,
    target_id: str,
    roots: set[str],
) -> set[str]:
    """BFS вверх по hypernym от target; возвращает корни-предки target."""
    if target_id not in wn.synsets:
        return set()

    found: set[str] = set()
    queue = [target_id]
    visited: set[str] = set()

    while queue:
        node_id = queue.pop(0)
        if node_id in visited:
            continue
        visited.add(node_id)

        if node_id in roots:
            found.add(node_id)

        synset = wn.synsets.get(node_id)
        if synset is None:
            continue

        for parent_id in synset.rels.get("hypernym", []):
            if parent_id not in visited:
                queue.append(parent_id)

    return found


def _downward_distance(wn: RuWordNet, start_id: str, target_id: str) -> int | None:
    """BFS вниз по hyponym от start до target. None если недостижим."""
    if start_id not in wn.synsets or target_id not in wn.synsets:
        return None
    if start_id == target_id:
        return 0

    dist: dict[str, int] = {start_id: 0}
    queue = [start_id]

    while queue:
        node_id = queue.pop(0)
        synset = wn.synsets.get(node_id)
        if synset is None:
            continue
        for child_id in synset.rels.get("hyponym", []):
            if child_id in dist:
                continue
            dist[child_id] = dist[node_id] + 1
            if child_id == target_id:
                return dist[child_id]
            queue.append(child_id)

    return None


def pick_best_root(
    wn: RuWordNet,
    target_id: str,
    ancestor_roots: set[str],
) -> str | None:
    """
    Корень с минимальной downward distance до target;
    при равенстве — min root_id (лексикографически).
    """
    best_root: str | None = None
    best_dist: int | None = None

    for root_id in ancestor_roots:
        dist = _downward_distance(wn, root_id, target_id)
        if dist is None:
            continue
        if best_dist is None or dist < best_dist or (dist == best_dist and root_id < best_root):
            best_dist = dist
            best_root = root_id

    return best_root


def _downward_distances_from(wn: RuWordNet, start_id: str) -> dict[str, int]:
    """BFS вниз по hyponym: dist[node] = расстояние от start_id."""
    dist: dict[str, int] = {start_id: 0}
    queue = [start_id]
    while queue:
        node_id = queue.pop(0)
        synset = wn.synsets.get(node_id)
        if synset is None:
            continue
        for child_id in synset.rels.get("hyponym", []):
            if child_id not in dist:
                dist[child_id] = dist[node_id] + 1
                queue.append(child_id)
    return dist


def find_shortest_downward_path(
    wn: RuWordNet,
    root_id: str,
    target_id: str,
) -> list[str] | None:
    """
    Кратчайший путь root → target только по hyponym.
    На развилках — min synset_id среди детей на кратчайшем пути.
    """
    if root_id not in wn.synsets or target_id not in wn.synsets:
        return None

    dist = _downward_distances_from(wn, root_id)
    if target_id not in dist:
        return None

    target_dist = dist[target_id]
    path = [root_id]
    current = root_id

    while current != target_id:
        candidates: list[str] = []
        current_dist = dist[current]
        for child_id in wn.synsets[current].rels.get("hyponym", []):
            if child_id not in dist:
                continue
            if dist[child_id] != current_dist + 1:
                continue
            remaining = target_dist - dist[child_id]
            if remaining == 0 and child_id == target_id:
                candidates.append(child_id)
            elif remaining > 0:
                child_to_target = _downward_distance(wn, child_id, target_id)
                if child_to_target == remaining:
                    candidates.append(child_id)

        if not candidates:
            return None

        next_node = min(candidates)
        path.append(next_node)
        current = next_node

    return path


def virtual_root_path(down_path: list[str]) -> list[str]:
    """Добавляет виртуальный корень null перед реальным root."""
    return [VIRTUAL_ROOT] + down_path
