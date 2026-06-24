import json
import logging
import re
from time import perf_counter
from typing import Callable

from multi_agent_research_lab.core.schemas import BenchmarkMetrics
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.services.llm_client import LLMClient

logger = logging.getLogger(__name__)

Runner = Callable[[str], ResearchState]


def run_benchmark(run_name: str, query: str, runner: Runner) -> tuple[ResearchState, BenchmarkMetrics]:
    """Measure latency, token cost, quality score (LLM-as-a-judge), and citation coverage."""
    logger.info(f"Running benchmark for '{run_name}' with query: '{query}'")
    started = perf_counter()
    state = runner(query)
    latency = perf_counter() - started

    # 1. Gather cost
    cost = state.estimated_cost_usd

    # 2. Compute citation coverage
    final_answer = state.final_answer or ""
    total_sources = len(state.sources)
    citation_count = 0
    if total_sources > 0:
        # Check for citation markers like [1], [2] in final answer
        cited_sources = set()
        for idx in range(1, total_sources + 1):
            pattern = rf"\[{idx}\]"
            if re.search(pattern, final_answer):
                cited_sources.add(idx)
        citation_count = len(cited_sources)
        citation_coverage = citation_count / total_sources
    else:
        citation_coverage = 0.0

    # 3. LLM-as-a-judge quality scoring (0 to 10)
    quality_score = 5.0
    reason_notes = f"Citations: {citation_count}/{total_sources} ({citation_coverage:.0%})."
    
    if final_answer and final_answer != "No answer produced.":
        try:
            llm = LLMClient()
            system_prompt = (
                "You are an expert evaluator. Evaluate the quality of the research summary "
                "provided by the assistant for the given query. Grade the summary on a scale "
                "of 0.0 to 10.0 based on criteria: accuracy, relevance, structure, and depth.\n"
                "Respond ONLY with a JSON object containing:\n"
                "{\n"
                "  \"score\": 8.5,\n"
                "  \"reason\": \"Brief explanation of the score.\"\n"
                "}\n"
                "Do not include markdown or backticks."
            )
            user_prompt = f"Query: {query}\n\nSummary:\n{final_answer}"
            res = llm.complete(system_prompt, user_prompt)
            
            clean_content = res.content.strip("` \n").replace("json\n", "")
            data = json.loads(clean_content)
            quality_score = float(data.get("score", 5.0))
            judge_reason = data.get("reason", "Evaluated by judge.")
            reason_notes += f" Judge review: {judge_reason}"
        except Exception as e:
            logger.error(f"LLM quality score evaluation failed: {e}")
            reason_notes += f" (Quality scoring failed: {e})"

    metrics = BenchmarkMetrics(
        run_name=run_name,
        latency_seconds=latency,
        estimated_cost_usd=cost,
        quality_score=quality_score,
        notes=reason_notes,
    )
    
    return state, metrics
