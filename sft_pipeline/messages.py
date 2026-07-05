import json

from taxoenrich.core import RuWordNet

from .prompts import load_system_prompt
from .trajectory import Trajectory


def build_user_message(context_text: str) -> str:
    return context_text


def trajectory_to_messages(
    wn: RuWordNet,
    trajectory: Trajectory,
    context_text: str,
) -> list[dict]:
    system_prompt = load_system_prompt()
    user_content = build_user_message(context_text)

    messages: list[dict] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]

    for i, step in enumerate(trajectory.steps):
        call_id = f"call_{i}"
        arguments = json.dumps({"node_id": step.node_id}, ensure_ascii=False)
        messages.append(
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": call_id,
                        "type": "function",
                        "function": {
                            "name": step.function,
                            "arguments": arguments,
                        },
                    }
                ],
            }
        )
        messages.append(
            {
                "role": "tool",
                "tool_call_id": call_id,
                "content": step.tool_result,
            }
        )

    if trajectory.final:
        messages.append(
            {
                "role": "assistant",
                "content": trajectory.final.content,
            }
        )

    return messages


def messages_to_jsonl_record(messages: list[dict]) -> dict:
    return {"messages": messages}
