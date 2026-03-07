"""OpenAI-compatible LLM provider.

Works with any API that follows the OpenAI chat-completions contract:
DeepSeek, Grok (xAI), Kimi (Moonshot), GLM (Zhipu), and others.
"""

from __future__ import annotations

from collections.abc import Generator
from typing import Any

from openai import OpenAI

from .base import LLMProvider
from .response import MockChoice, MockFunction, MockMessage, MockResponse, MockToolCall


class OpenAICompatibleProvider(LLMProvider):
    """Thin wrapper around the OpenAI SDK for chat completions."""

    def __init__(self, api_key: str, base_url: str, model_name: str) -> None:
        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=120.0,
        )
        self.model_name = model_name

    def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        tool_choice: Any = "auto",
        **kwargs: Any,
    ) -> Any:
        req: dict[str, Any] = {
            "model": self.model_name,
            "messages": messages,
            **kwargs,
        }
        if tools:
            req["tools"] = tools
            req["tool_choice"] = tool_choice

        return self.client.chat.completions.create(**req)

    def chat_stream(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        tool_choice: Any = "auto",
        **kwargs: Any,
    ) -> Generator[dict[str, Any], None, Any]:
        """Stream OpenAI-compatible responses, yielding text deltas."""
        req: dict[str, Any] = {
            "model": self.model_name,
            "messages": messages,
            "stream": True,
            **kwargs,
        }
        if tools:
            req["tools"] = tools
            req["tool_choice"] = tool_choice

        content_text = ""
        tool_calls_acc: dict[int, dict] = {}

        for chunk in self.client.chat.completions.create(**req):
            delta = chunk.choices[0].delta if chunk.choices else None
            if delta is None:
                continue

            if delta.content:
                content_text += delta.content
                yield {"type": "text_delta", "text": delta.content}

            if delta.tool_calls:
                for tc_delta in delta.tool_calls:
                    idx = tc_delta.index
                    if idx not in tool_calls_acc:
                        tool_calls_acc[idx] = {
                            "id": tc_delta.id or "",
                            "name": "",
                            "args": "",
                        }
                    if tc_delta.id:
                        tool_calls_acc[idx]["id"] = tc_delta.id
                    if tc_delta.function:
                        if tc_delta.function.name:
                            tool_calls_acc[idx]["name"] = tc_delta.function.name
                        if tc_delta.function.arguments:
                            tool_calls_acc[idx]["args"] += (
                                tc_delta.function.arguments
                            )

        mock_tool_calls = [
            MockToolCall(
                id=v["id"],
                function=MockFunction(name=v["name"], arguments=v["args"] or "{}"),
            )
            for v in sorted(tool_calls_acc.values(), key=lambda x: x["id"])
        ]

        return MockResponse(choices=[
            MockChoice(message=MockMessage(
                content=content_text or None,
                tool_calls=mock_tool_calls or None,
            ))
        ])
