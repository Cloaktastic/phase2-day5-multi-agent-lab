import os
import sys
from pathlib import Path

# Add src/ folder to path to make multi_agent_research_lab discoverable
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from multi_agent_research_lab.core.schemas import ResearchQuery
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.services.llm_client import LLMClient
from multi_agent_research_lab.graph.workflow import MultiAgentWorkflow
from multi_agent_research_lab.evaluation.benchmark import run_benchmark
from multi_agent_research_lab.evaluation.report import render_markdown_report


def run_baseline_agent(query: str) -> ResearchState:
    state = ResearchState(request=ResearchQuery(query=query))
    llm = LLMClient()
    system_prompt = (
        "You are a helpful research assistant. Synthesize a comprehensive response "
        "answering the user query directly. Output your answer in clean Markdown."
    )
    res = llm.complete(system_prompt, query)
    state.final_answer = res.content
    state.input_tokens = res.input_tokens or 0
    state.output_tokens = res.output_tokens or 0
    state.estimated_cost_usd = res.cost_usd or 0.0
    return state


def run_multi_agent(query: str) -> ResearchState:
    state = ResearchState(request=ResearchQuery(query=query))
    workflow = MultiAgentWorkflow()
    return workflow.run(state)


def main() -> None:
    query = "Research GraphRAG state-of-the-art and write a 500-word summary"
    
    print(f"Starting evaluations for query: '{query}'")
    print("--------------------------------------------------")
    
    print("Running Single-Agent Baseline...")
    _, baseline_metrics = run_benchmark("Single-Agent Baseline", query, run_baseline_agent)
    print("Single-Agent Baseline complete.\n")
    
    print("Running Multi-Agent Workflow...")
    _, multi_metrics = run_benchmark("Multi-Agent Workflow", query, run_multi_agent)
    print("Multi-Agent Workflow complete.\n")
    
    report_md = render_markdown_report([baseline_metrics, multi_metrics])
    
    report_path = Path(__file__).parent.parent / "reports" / "benchmark_report.md"
    report_path.parent.mkdir(exist_ok=True)
    
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_md)
        
    print(f"Evaluation report written successfully to: {report_path.resolve()}")
    print("--------------------------------------------------")
    print(report_md)


if __name__ == "__main__":
    main()
