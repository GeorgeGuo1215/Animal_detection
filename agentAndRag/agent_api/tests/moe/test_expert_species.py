"""Verify run_expert injects species into the RAG query and the expert payload.

Uses fake registry + fake LLM (no DB / no network). ASCII sentinels only.
Run: pytest tests/moe/test_expert_species.py
"""
import asyncio
import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from app.services.moe.experts import EXPERTS, run_expert


class FakeRegistry:
    def __init__(self):
        self.calls = []

    async def call(self, name, args):
        self.calls.append((name, args))
        return {"hits": []}


class FakeLLM:
    model = "fake"

    def __init__(self):
        self.last_messages = None

    async def chat(self, messages=None, **kwargs):
        self.last_messages = messages
        content = json.dumps(
            {"conclusion": "ok", "evidence": [], "risks": [], "confidence": 0.5}
        )
        return {"choices": [{"message": {"content": content}}]}


def _user_content(messages):
    return [m for m in messages if m["role"] == "user"][0]["content"]


def test_run_expert_injects_species():
    reg, llm = FakeRegistry(), FakeLLM()
    res = asyncio.run(
        run_expert(
            expert=EXPERTS["clinical"],
            query="my pet keeps vomiting",
            weight=0.8,
            registry=reg,
            llm=llm,
            species_en="cat",
            species_zh="ZH_SENTINEL",
            recorder=None,
        )
    )
    assert reg.calls, "rag.search was not called"
    rag_args = reg.calls[0][1]
    assert "cat" in rag_args["query"]

    user_content = _user_content(llm.last_messages)
    assert "ZH_SENTINEL" in user_content
    assert "species" in user_content
    assert res["expert"] == "clinical"


def test_run_expert_without_species():
    reg, llm = FakeRegistry(), FakeLLM()
    asyncio.run(
        run_expert(
            expert=EXPERTS["clinical"],
            query="my pet keeps vomiting",
            weight=0.8,
            registry=reg,
            llm=llm,
            recorder=None,
        )
    )
    user_content = _user_content(llm.last_messages)
    assert "species" not in user_content
