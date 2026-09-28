"""Utilities for inspecting agent run results in notebooks."""

from __future__ import annotations

import json
from typing import Any


def _get_attr(obj: Any, name: str, default: Any = None) -> Any:
    """Read an attribute from an object or key from a dict-like payload."""
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _extract_text_from_message_item(message_item: Any) -> str:
    """Flatten text blocks from a message item into a single string."""
    content = _get_attr(message_item, "content", []) or []
    texts: list[str] = []
    for block in content:
        text = _get_attr(block, "text")
        if text:
            texts.append(text)
    return "\n".join(texts).strip()


def _extract_reasoning_summary(summary_items: Any) -> list[str]:
    """Flatten reasoning summary items into readable text strings."""
    if not summary_items:
        return []

    texts: list[str] = []
    for item in summary_items:
        if isinstance(item, str):
            if item.strip():
                texts.append(item.strip())
            continue

        text = _get_attr(item, "text")
        if text and str(text).strip():
            texts.append(str(text).strip())
            continue

        summary = _get_attr(item, "summary")
        if summary and str(summary).strip():
            texts.append(str(summary).strip())

    return texts


def inspect_agent_result(result: Any, verbose: bool = True) -> list[dict[str, Any]]:
    """Inspect a run result and collect reasoning, tool calls, and tool outputs.

    Args:
        result: A run result object such as the return value from ``Runner.run(...)``.
        verbose: If True, print a readable inspection trace.

    Returns:
        A list of normalized step dictionaries. Function-call steps include the
        search query, tool name, call id, and matching tool output when found.
    """
    tool_outputs: dict[str, Any] = {}
    for item in _get_attr(result, "new_items", []) or []:
        item_type = _get_attr(item, "type")
        if item_type != "tool_call_output_item":
            continue

        raw_item = _get_attr(item, "raw_item", {}) or {}
        call_id = _get_attr(raw_item, "call_id") or _get_attr(item, "call_id")
        output = _get_attr(raw_item, "output") or _get_attr(item, "output")
        if call_id:
            tool_outputs[call_id] = output

    all_steps: list[dict[str, Any]] = []

    for resp_idx, resp in enumerate(_get_attr(result, "raw_responses", []) or [], start=1):
        if verbose:
            print(f"\n{'=' * 100}")
            print(f"RAW RESPONSE #{resp_idx}")
            print(f"response_id: {_get_attr(resp, 'response_id')}")
            print(f"request_id : {_get_attr(resp, 'request_id')}")

        output_items = _get_attr(resp, "output", []) or []
        for item_idx, item in enumerate(output_items, start=1):
            item_type = _get_attr(item, "type")

            if verbose:
                print(f"\n--- item #{item_idx} | type={item_type} ---")

            if item_type == "reasoning":
                summary = _get_attr(item, "summary", [])
                content = _get_attr(item, "content")
                encrypted_content = _get_attr(item, "encrypted_content")
                summary_texts = _extract_reasoning_summary(summary)
                summary_text = "\n\n".join(summary_texts).strip()

                if verbose:
                    if summary_texts:
                        print("reasoning.summary:")
                        for summary_item in summary_texts:
                            print(summary_item)
                    elif content:
                        print("reasoning.content:")
                        print(content)
                    elif encrypted_content:
                        print("reasoning.encrypted_content exists but is not human-readable.")
                    else:
                        print("reasoning: <empty / not exposed by model>")

                all_steps.append(
                    {
                        "response_idx": resp_idx,
                        "item_type": "reasoning",
                        "reasoning_summary": summary_text,
                        "reasoning_content": content,
                        "reasoning_encrypted_content": encrypted_content,
                    }
                )
                continue

            if item_type == "function_call":
                call_id = _get_attr(item, "call_id")
                tool_name = _get_attr(item, "name")
                raw_args = _get_attr(item, "arguments", "{}")

                try:
                    args = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
                except Exception:
                    args = {"_raw_arguments": raw_args}

                query = args.get("query")
                max_results = args.get("max_results")
                filter_year = args.get("filter_year")
                search_output = tool_outputs.get(call_id)

                if verbose:
                    print(f"tool_name   : {tool_name}")
                    print(f"call_id     : {call_id}")
                    print(f"query       : {query}")
                    print(f"max_results : {max_results}")
                    print(f"filter_year : {filter_year}")
                    print("\nsearch output:")
                    print(search_output or "<no matching tool output found in result.new_items>")

                all_steps.append(
                    {
                        "response_idx": resp_idx,
                        "item_type": "function_call",
                        "tool_name": tool_name,
                        "call_id": call_id,
                        "query": query,
                        "max_results": max_results,
                        "filter_year": filter_year,
                        "search_output": search_output,
                    }
                )
                continue

            if item_type == "message":
                final_text = _extract_text_from_message_item(item)
                parsed_json = None
                if final_text:
                    try:
                        parsed_json = json.loads(final_text)
                    except json.JSONDecodeError:
                        parsed_json = None

                if verbose:
                    print("message output:")
                    print(final_text)

                message_step = {
                    "response_idx": resp_idx,
                    "item_type": "message",
                    "message_text": final_text,
                }
                if parsed_json is not None:
                    message_step["parsed_json"] = parsed_json
                all_steps.append(message_step)
                continue

            if verbose:
                print("unhandled item:")
                print(item)

            all_steps.append(
                {
                    "response_idx": resp_idx,
                    "item_type": item_type,
                    "raw_item": item,
                }
            )

    return all_steps


def extract_search_steps(result: Any) -> list[dict[str, Any]]:
    """Return only the function-call steps from an inspected run result."""
    return [
        step
        for step in inspect_agent_result(result, verbose=False)
        if step.get("item_type") == "function_call"
    ]


def extract_reasoning_steps(result: Any) -> list[dict[str, Any]]:
    """Return only the reasoning steps from an inspected run result."""
    return [
        step
        for step in inspect_agent_result(result, verbose=False)
        if step.get("item_type") == "reasoning"
    ]
