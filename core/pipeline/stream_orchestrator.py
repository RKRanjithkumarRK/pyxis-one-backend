"""
Stream orchestrator — the main request pipeline.

Flow per request:
  1. classify intent
  2. select model
  3. prepare context (trim + RAG inject)
  4. build tools list
  5. stream from model
  6. intercept tool events → execute tool → reinject → continue
  7. emit SSE StreamEvents to router
"""

from __future__ import annotations
import asyncio
import json
from typing import AsyncGenerator

from core.pipeline.intent_classifier import classify, RouterDecision
from core.pipeline.model_router import select_model, get_system_prompt, ModelSelection
from core.pipeline.context_manager import prepare, build_tool_result_message
from engines.unified_client import StreamEvent, stream as llm_stream
from tools.executor import execute_tool
from tools.definitions import get_tool_schemas


async def orchestrate(
    *,
    session_id: str,
    message: str,
    history: list[dict],
    feature_mode: str = "standard",
    psyche_context: str = "",
    rag_chunks: list[str] | None = None,
    user_tier: str = "free",
    manual_model: str | None = None,
    enable_web_search: bool = False,
    has_attachments: bool = False,
    file_contexts: list[dict] | None = None,
    temperature_boost: float = 0.0,   # for regeneration (+0.1 per attempt)
) -> AsyncGenerator[StreamEvent, None]:
    """
    Full request pipeline. Yields StreamEvents directly to the SSE router.
    Handles multi-turn tool loops internally (model → tool → model → ...).
    """

    # 1. Classify intent
    decision: RouterDecision = classify(message, has_attachments=has_attachments)

    # 2. Select model
    selection: ModelSelection = select_model(
        decision,
        user_tier=user_tier,
        manual_model=manual_model,
        enable_web_search=enable_web_search,
    )
    selection.temperature = min(1.0, selection.temperature + temperature_boost)

    # Emit model info to frontend
    yield StreamEvent(
        type="model_selected",
        content=json.dumps({
            "model": selection.model,
            "provider": selection.provider,
            "intent": decision.intent,
        }),
    )

    # 3. Build system prompt
    feature_context = _feature_system(feature_mode) if feature_mode != "standard" else ""
    system = get_system_prompt(selection.persona, feature_context, psyche_context)

    # 4. Add file contexts to message if any
    full_message = message
    if file_contexts:
        file_block = "\n\n".join(
            f"[Uploaded file: {f['filename']}]\n{f['content'][:8000]}"
            for f in file_contexts
        )
        full_message = f"{message}\n\n{file_block}"

    # 5. Prepare context (trim history, inject RAG)
    enriched_system, trimmed_history = await prepare(
        model=selection.model,
        system=system,
        messages=history + [{"role": "user", "content": full_message}],
        rag_chunks=rag_chunks,
        psyche_context=psyche_context,
    )

    # 6. Get tool schemas
    tool_schemas = get_tool_schemas(selection.inject_tools) if selection.inject_tools else None

    # 7. Multi-turn tool loop (max 5 iterations to prevent infinite loops)
    working_messages = trimmed_history
    max_tool_iterations = 5

    for iteration in range(max_tool_iterations):
        tool_calls_this_round: list[dict] = []
        tool_args_accumulator: dict[str, str] = {}  # tool_call_id → args json string

        # Stream from model
        async for event in llm_stream(
            messages=working_messages,
            system=enriched_system,
            model=selection.model,
            max_tokens=selection.max_tokens,
            tools=tool_schemas,
            temperature=selection.temperature,
        ):
            if event.type == "tool_start":
                # Frontend shows "Searching..." / "Running code..."
                yield event
                tool_args_accumulator[event.tool_call_id or ""] = ""

            elif event.type == "tool_delta":
                tool_args_accumulator[event.tool_call_id or ""] = (
                    tool_args_accumulator.get(event.tool_call_id or "", "")
                    + (event.content or "")
                )

            elif event.type == "tool_done":
                # Parse accumulated args
                raw_args = tool_args_accumulator.get(event.tool_call_id or "", "{}")
                try:
                    parsed_args = json.loads(raw_args) if raw_args else {}
                except json.JSONDecodeError:
                    parsed_args = {}
                tool_calls_this_round.append({
                    "name": event.tool_name,
                    "id": event.tool_call_id,
                    "input": parsed_args or (event.tool_input or {}),
                })

            elif event.type == "done":
                yield event
                stop_reason = (event.usage or {}).get("stop_reason", "stop")
                if stop_reason != "tool_use" and not tool_calls_this_round:
                    return  # Natural completion, no tools needed

            else:
                yield event  # text, thinking, system, error

        # If no tool calls, we're done
        if not tool_calls_this_round:
            return

        # Execute all tool calls (parallel)
        tool_results = await asyncio.gather(*[
            _run_tool(tc) for tc in tool_calls_this_round
        ])

        # Emit tool results to frontend
        for tc, result in zip(tool_calls_this_round, tool_results):
            yield StreamEvent(
                type="tool_result",
                tool_name=tc["name"],
                tool_call_id=tc["id"],
                content=result[:2000],  # truncate for display
            )

        # Build assistant message with tool calls + inject results
        if selection.provider == "anthropic":
            assistant_content = [
                {
                    "type": "tool_use",
                    "id": tc["id"],
                    "name": tc["name"],
                    "input": tc["input"],
                }
                for tc in tool_calls_this_round
            ]
            working_messages = working_messages + [
                {"role": "assistant", "content": assistant_content},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": tc["id"],
                            "content": result,
                        }
                        for tc, result in zip(tool_calls_this_round, tool_results)
                    ],
                },
            ]
        else:  # openai
            tool_call_objects = [
                {
                    "id": tc["id"],
                    "type": "function",
                    "function": {
                        "name": tc["name"],
                        "arguments": json.dumps(tc["input"]),
                    },
                }
                for tc in tool_calls_this_round
            ]
            working_messages = working_messages + [
                {"role": "assistant", "content": None, "tool_calls": tool_call_objects},
            ] + [
                {
                    "role": "tool",
                    "tool_call_id": tc["id"],
                    "content": result,
                }
                for tc, result in zip(tool_calls_this_round, tool_results)
            ]

        tool_calls_this_round.clear()
        # Loop back → model continues with tool results

    yield StreamEvent(type="system", content="Tool iteration limit reached.")


async def _run_tool(tool_call: dict) -> str:
    """Execute a single tool call, return string result."""
    try:
        result = await execute_tool(
            name=tool_call["name"],
            input_data=tool_call["input"],
        )
        return result
    except Exception as e:
        return f"Tool error ({tool_call['name']}): {str(e)[:300]}"


def _feature_system(feature_mode: str) -> str:
    """Return feature-specific system prompt addendum."""
    mode_map = {
        "forge":          "You are in Cognitive Forge mode. Guide the user through metallurgical stages of concept mastery.",
        "oracle":         "You are in Oracle mode. Predict learning obstacles and provide preemptive scaffolding.",
        "nemesis":        "You are in Nemesis mode. Challenge the user's weakest knowledge points mercilessly but constructively.",
        "helix":          "You are in Helix mode. Apply spaced repetition principles in your responses.",
        "parliament":     "You are in Philosopher Parliament mode. Channel historical thinkers when presenting perspectives.",
        "trident":        "You are in Trident mode. Respond as three simultaneous perspectives: Architect, Street Fighter, Heretic.",
        "apex":           "You are in Apex mode. All cognitive dimensions converge simultaneously. Be comprehensive and transcendent.",
        "dominion":       "You are in Dominion mode. Map the user's knowledge territory forensically.",
        "eternal":        "You are in Eternal Archive mode. Crystallize insights for cross-session memory.",
        "transcendence":  "You are in Transcendence mode. Analyze through math, philosophy, engineering, art, cosmic, and street lenses.",
    }
    return mode_map.get(feature_mode, "")
