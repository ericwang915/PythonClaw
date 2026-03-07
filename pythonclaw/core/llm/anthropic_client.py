"""
Anthropic (Claude) provider — adapts the Anthropic API to the OpenAI-compatible
response format used by Agent.
"""

from __future__ import annotations

import json
from collections.abc import Generator
from typing import Any, Dict, List, Optional

import anthropic

from .base import LLMProvider
from .response import MockChoice, MockFunction, MockMessage, MockResponse, MockToolCall


class AnthropicProvider(LLMProvider):
    """Anthropic Claude provider with API key or setup-token support.

    Supports all authentication methods:
    - Standard API key (``sk-ant-...``)
    - Setup token (from ``claude setup-token``, long-lived session token)
    - Environment variable ``ANTHROPIC_API_KEY``
    """

    supports_images = True

    def __init__(self, api_key: str, model_name: str = "claude-sonnet-4-20250514"):
        self.client = anthropic.Anthropic(
            api_key=api_key,
            timeout=120.0,
        )
        self.model_name = model_name
        self._auth_type = (
            "setup-token" if not api_key.startswith("sk-ant-") else "api-key"
        )

    # ── shared helpers ────────────────────────────────────────────────────

    def _prepare_request(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]],
        tool_choice: Any,
        **kwargs: Any,
    ) -> dict:
        """Build the kwargs dict for ``messages.create`` / ``stream``."""
        system_prompt = ""
        filtered_messages: list[dict] = []

        for msg in messages:
            if msg["role"] == "system":
                system_prompt += msg["content"] + "\n"

            elif msg["role"] == "tool":
                filtered_messages.append({
                    "role": "user",
                    "content": [{
                        "type": "tool_result",
                        "tool_use_id": msg["tool_call_id"],
                        "content": msg["content"],
                    }],
                })

            elif msg["role"] == "assistant" and "tool_calls" in msg:
                content_block: list[dict] = []
                if msg.get("content"):
                    content_block.append({"type": "text", "text": msg["content"]})
                for tc in msg["tool_calls"]:
                    tc_id = tc["id"] if isinstance(tc, dict) else tc.id
                    func = tc["function"] if isinstance(tc, dict) else tc.function
                    fname = func["name"] if isinstance(func, dict) else func.name
                    fargs = json.loads(
                        func["arguments"] if isinstance(func, dict) else func.arguments
                    )
                    content_block.append({
                        "type": "tool_use",
                        "id": tc_id,
                        "name": fname,
                        "input": fargs,
                    })
                filtered_messages.append({"role": "assistant", "content": content_block})

            elif msg["role"] == "user" and isinstance(msg.get("content"), list):
                filtered_messages.append({
                    "role": "user",
                    "content": self._convert_user_content(msg["content"]),
                })

            else:
                filtered_messages.append(msg)

        filtered_messages = self._merge_consecutive(filtered_messages)

        anthropic_tools = []
        if tools:
            for t in tools:
                if t["type"] == "function":
                    anthropic_tools.append({
                        "name": t["function"]["name"],
                        "description": t["function"]["description"],
                        "input_schema": t["function"]["parameters"],
                    })

        max_tokens = kwargs.get("max_tokens", 4096)
        api_kwargs: dict = {
            "model": self.model_name,
            "messages": filtered_messages,
            "max_tokens": max_tokens,
        }
        if system_prompt:
            api_kwargs["system"] = system_prompt
        if anthropic_tools:
            api_kwargs["tools"] = anthropic_tools
            if tool_choice == "required":
                api_kwargs["tool_choice"] = {"type": "any"}
            elif tool_choice == "none":
                pass
            else:
                api_kwargs["tool_choice"] = {"type": "auto"}

        return api_kwargs

    @staticmethod
    def _response_from_blocks(
        content_text: str,
        tool_calls: list[MockToolCall],
    ) -> MockResponse:
        return MockResponse(choices=[
            MockChoice(message=MockMessage(
                content=content_text or None,
                tool_calls=tool_calls or None,
            ))
        ])

    # ── non-streaming ─────────────────────────────────────────────────────

    def chat(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Any = "auto",
        **kwargs: Any,
    ) -> Any:
        api_kwargs = self._prepare_request(
            messages, tools, tool_choice, **kwargs
        )
        response = self.client.messages.create(**api_kwargs)

        content_text = ""
        tool_calls: list[MockToolCall] = []
        for block in response.content:
            if block.type == "text":
                content_text += block.text
            elif block.type == "tool_use":
                tool_calls.append(MockToolCall(
                    id=block.id,
                    function=MockFunction(
                        name=block.name,
                        arguments=json.dumps(block.input),
                    ),
                ))

        return self._response_from_blocks(content_text, tool_calls)

    # ── streaming ─────────────────────────────────────────────────────────

    def chat_stream(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Any = "auto",
        **kwargs: Any,
    ) -> Generator[dict[str, Any], None, MockResponse]:
        """Stream Anthropic responses, yielding text deltas."""
        api_kwargs = self._prepare_request(
            messages, tools, tool_choice, **kwargs
        )

        content_text = ""
        tool_calls: list[MockToolCall] = []
        current_tool: dict[str, Any] | None = None

        with self.client.messages.stream(**api_kwargs) as stream:
            for event in stream:
                if event.type == "content_block_start":
                    block = event.content_block
                    if block.type == "tool_use":
                        current_tool = {
                            "id": block.id,
                            "name": block.name,
                            "args": "",
                        }
                elif event.type == "content_block_delta":
                    delta = event.delta
                    if delta.type == "text_delta":
                        content_text += delta.text
                        yield {"type": "text_delta", "text": delta.text}
                    elif delta.type == "input_json_delta":
                        if current_tool is not None:
                            current_tool["args"] += delta.partial_json
                elif event.type == "content_block_stop":
                    if current_tool is not None:
                        tool_calls.append(MockToolCall(
                            id=current_tool["id"],
                            function=MockFunction(
                                name=current_tool["name"],
                                arguments=current_tool["args"] or "{}",
                            ),
                        ))
                        current_tool = None

        return self._response_from_blocks(content_text, tool_calls)

    # ── multimodal conversion ────────────────────────────────────────────

    @staticmethod
    def _convert_user_content(parts: list[dict]) -> list[dict]:
        """Convert OpenAI-style content parts to Anthropic format.

        Handles ``image_url`` with ``data:`` URIs (base64) or plain URLs.
        """
        import base64 as _b64
        import re as _re

        out: list[dict] = []
        for p in parts:
            if p.get("type") == "text":
                out.append({"type": "text", "text": p["text"]})
            elif p.get("type") == "image_url":
                url = p["image_url"]["url"]
                m = _re.match(
                    r"data:(image/\w+);base64,(.+)", url, _re.DOTALL
                )
                if m:
                    out.append({
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": m.group(1),
                            "data": m.group(2),
                        },
                    })
                else:
                    try:
                        import urllib.request

                        resp = urllib.request.urlopen(url, timeout=15)
                        data = resp.read()
                        ct = resp.headers.get(
                            "Content-Type", "image/jpeg"
                        ).split(";")[0]
                        out.append({
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": ct,
                                "data": _b64.b64encode(data).decode(),
                            },
                        })
                    except Exception:
                        out.append({
                            "type": "text",
                            "text": f"[image: {url}]",
                        })
            else:
                out.append(p)
        return out

    # ── utilities ─────────────────────────────────────────────────────────

    @staticmethod
    def _merge_consecutive(messages: list[dict]) -> list[dict]:
        """Merge consecutive messages with the same role (Anthropic requirement)."""
        if not messages:
            return messages
        merged: list[dict] = [messages[0]]
        for msg in messages[1:]:
            if msg["role"] == merged[-1]["role"]:
                prev_content = merged[-1].get("content", "")
                curr_content = msg.get("content", "")
                if isinstance(prev_content, str) and isinstance(curr_content, str):
                    merged[-1]["content"] = prev_content + "\n" + curr_content
                elif isinstance(prev_content, list) and isinstance(curr_content, list):
                    merged[-1]["content"] = prev_content + curr_content
                elif isinstance(prev_content, str) and isinstance(curr_content, list):
                    merged[-1]["content"] = [
                        {"type": "text", "text": prev_content}
                    ] + curr_content
                elif isinstance(prev_content, list) and isinstance(curr_content, str):
                    merged[-1]["content"] = prev_content + [
                        {"type": "text", "text": curr_content}
                    ]
            else:
                merged.append(msg)
        return merged
