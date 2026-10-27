"""Runs the real agent with a scripted model (no API key): checks tools are wired and return the right data."""
import asyncio
from typing import AsyncGenerator

from google.adk.models.base_llm import BaseLlm
from google.adk.models.llm_response import LlmResponse
from google.adk.runners import InMemoryRunner
from google.genai import types

from oprisk_agent import agent


class StepLlm(BaseLlm):
    model: str = "steps"
    steps: list = []
    requests: list = []

    async def generate_content_async(self, llm_request, stream=False) -> AsyncGenerator[LlmResponse, None]:
        self.requests.append(llm_request)
        step = self.steps[len(self.requests) - 1]
        part = (types.Part(function_call=types.FunctionCall(name=step[1], args=step[2]))
                if step[0] == "call" else types.Part(text=step[1]))
        yield LlmResponse(content=types.Content(role="model", parts=[part]))


def ask(question, steps):
    llm = StepLlm(steps=steps, requests=[])
    agent.root_agent.model = llm

    async def go():
        r = InMemoryRunner(agent=agent.root_agent, app_name="t")
        s = await r.session_service.create_session(app_name="t", user_id="u")
        results = {}
        msg = types.Content(role="user", parts=[types.Part(text=question)])
        async for e in r.run_async(user_id="u", session_id=s.id, new_message=msg):
            for p in (e.content.parts if e.content else []):
                if p.function_response:
                    results.setdefault(p.function_response.name, []).append(p.function_response.response)
        return results

    return asyncio.run(go()), llm


def test_level0_only():
    results, llm = ask("Which tables exist for KRIs?", [("text", "kri_definitions and kri_values.")])
    prompt = str(llm.requests[0].config.system_instruction)
    assert "kri_values" in prompt and "red_threshold" not in prompt  # datasets listed, no columns
    assert results == {}


def test_level1_then_level2():
    results, _ = ask("How many red KRI breaches last quarter?", [
        ("call", "search_knowledge", {"query": "red breach", "domain": "kri"}),
        ("call", "load_tables", {"model": "oprisk_kri", "datasets": ["kri_values"]}),
        ("text", "Use red_breach_count on kri_values."),
    ])
    section = results["search_knowledge"][0]["sections"][0]
    assert section["path"] == "kri.rag_and_breaches" and "oprisk_kri.kri_values" in section["tables"]
    tables = results["load_tables"][0]
    assert [d["name"] for d in tables["datasets"]] == ["kri_values"]
    assert [m["name"] for m in tables["metrics"]] == ["red_breach_count"]


def test_bad_table_returns_error_to_model():
    results, _ = ask("x", [("call", "load_tables", {"model": "oprisk_kri", "datasets": ["kri"]}), ("text", "ok")])
    assert "Options" in results["load_tables"][0]["error"]
