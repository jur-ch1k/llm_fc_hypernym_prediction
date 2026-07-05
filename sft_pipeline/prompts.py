from .config import SYSTEM_PROMPT_PATH


def load_system_prompt(path: str = SYSTEM_PROMPT_PATH) -> str:
    with open(path, encoding="utf-8") as f:
        return f.read()
