"""One LLM agent. L0 is in the prompt; it loads L1 sections and L2 tables only when needed."""
import json
import os

from google.adk import Agent

import kb

MODEL = os.getenv("KB_AGENT_MODEL", "gemini-2.5-flash")


def search_knowledge(query: str, domain: str = "") -> dict:
    """Level 1: find and load the business-knowledge sections relevant to a question
    (definitions, rules, thresholds, processes). Returns at most 3 sections, full content.

    Args:
        query: key terms from the question, e.g. "red kri escalation cro".
        domain: optional domain name from the index to search only there, e.g. "kri".
    """
    results = kb.search_sections(query, domain=domain)
    return {"sections": results} if results else {"sections": [], "hint": "No match. Try other terms or no domain."}


def load_tables(model: str, datasets: list[str]) -> dict:
    """Level 2: load table-level detail (columns, joins, metrics) from an Apache Ossie model.
    Only call when the question needs data or table/column details. Load only the datasets you need.

    Args:
        model: Ossie model name from the index, e.g. "oprisk_losses".
        datasets: dataset names in that model, e.g. ["op_loss_events"].
    """
    try:
        return kb.load_tables(model, datasets)
    except KeyError as e:
        return {"error": str(e)}


INSTRUCTION = """
You answer operational-risk questions. Knowledge has three levels; load only what the question needs.

Level 0 (below) lists the domains and their datasets. It is all you know without tools.
<knowledge_index>
{index}
</knowledge_index>

Decide:
1. Meta questions ("what do you cover?", "which tables exist for KRIs?") -> answer from Level 0. No tools.
2. Meaning, definitions, rules, thresholds, processes -> search_knowledge with focused key terms.
   Pass `domain` when it is obvious. If the sections don't cover it, search again with different terms.
3. Data, tables, columns, joins, metrics, or SQL -> load_tables with only the datasets needed.
   A section's "tables" list (model.dataset) tells you which datasets relate to it.
   Questions about a business metric often need both: search_knowledge for the meaning, then load_tables.
4. Never answer definitions, thresholds or rules from general knowledge. If not found, say what is missing.

Answer concisely, quote numbers exactly, and end with "Sources:" listing section paths and
model.dataset names you used.
"""


def instruction(ctx) -> str:
    # A function (not a string) so ADK skips {state} templating on the JSON below.
    return INSTRUCTION.format(index=json.dumps(kb.level0(), indent=1))


root_agent = Agent(
    name="oprisk_knowledge",
    model=MODEL,
    description="Answers operational risk questions, loading only the knowledge depth needed.",
    instruction=instruction,
    tools=[search_knowledge, load_tables],
)
