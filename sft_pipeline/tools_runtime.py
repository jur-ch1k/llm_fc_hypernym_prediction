from taxoenrich.core import RuWordNet

VIRTUAL_ROOT_ALIASES = frozenset({"null", "none"})


def is_virtual_root_token(node_id: str) -> bool:
    return str(node_id).strip().lower() in VIRTUAL_ROOT_ALIASES


def resolve_synset_id(node_id: str | None) -> str | None:
    """RuWordNet lookup: виртуальный корень (null/None/…) -> None."""
    if node_id is None:
        return None
    if is_virtual_root_token(node_id):
        return None
    return str(node_id).strip()


def _sanitize_cell(value: str) -> str:
    return value.replace("\t", " ").replace("\n", " ").replace("\r", " ")


def _format_list(items: list[str], max_items: int) -> str:
    return "; ".join(_sanitize_cell(item) for item in items[:max_items])


def _format_synset_rows(
    results: list[dict],
    *,
    max_words: int,
    max_children: int,
    children_key: str,
) -> str:
    rows: list[str] = []
    for item in results:
        children = item.get(children_key, [])
        total = len(children)
        children_str = _format_list(children, max_children) if total else ""
        definition = _sanitize_cell(item.get("definition") or "")
        rows.append(
            "\t".join(
                [
                    _sanitize_cell(item["id"]),
                    _sanitize_cell(item["name"]),
                    definition,
                    _format_list(item["words"], max_words),
                    str(total),
                    children_str,
                ]
            )
        )
    return "\n".join(rows)


def format_hyponyms(
    wn: RuWordNet,
    node_id: str | None,
    *,
    max_words: int = 3,
    max_children: int = 5,
) -> str:
    """Sync TSV для get_hyponyms (без reranking, без заголовка — колонки в system prompt)."""
    node_id = resolve_synset_id(node_id)

    results = wn.get_hyponyms(node_id, pos="N")
    if not results:
        return "Гипонимов не найдено."

    return _format_synset_rows(
        results,
        max_words=max_words,
        max_children=max_children,
        children_key="hyponyms",
    )


def format_hypernyms(
    wn: RuWordNet,
    node_id: str | None,
    *,
    max_words: int = 3,
    max_children: int = 5,
) -> str:
    """Sync TSV для get_hypernyms (без reranking, без заголовка — колонки в system prompt)."""
    node_id = resolve_synset_id(node_id)

    results = wn.get_hypernyms(node_id, pos="N")
    if not results:
        return "Гиперонимов не найдено."

    return _format_synset_rows(
        results,
        max_words=max_words,
        max_children=max_children,
        children_key="hypernyms",
    )


def execute_tool(
    wn: RuWordNet,
    function: str,
    node_id: str | None,
    *,
    max_words: int = 3,
    max_hyponyms: int = 5,
) -> str:
    if function == "get_hyponyms":
        return format_hyponyms(wn, node_id, max_words=max_words, max_children=max_hyponyms)
    if function == "get_hypernyms":
        return format_hypernyms(wn, node_id, max_words=max_words, max_children=max_hyponyms)
    raise ValueError(f"Неизвестная функция: {function}")
