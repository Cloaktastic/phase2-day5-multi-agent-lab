"""Command-line entrypoint for the lab starter."""

from typing import Annotated

import typer
from rich.console import Console
from rich.panel import Panel

from multi_agent_research_lab.core.config import get_settings
from multi_agent_research_lab.core.errors import StudentTodoError
from multi_agent_research_lab.core.schemas import ResearchQuery
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.graph.workflow import MultiAgentWorkflow
from multi_agent_research_lab.observability.logging import configure_logging

app = typer.Typer(help="Multi-Agent Research Lab starter CLI")
console = Console()


def _init() -> None:
    settings = get_settings()
    configure_logging(settings.log_level)


from time import perf_counter
from multi_agent_research_lab.services.llm_client import LLMClient

@app.command()
def baseline(
    query: Annotated[str, typer.Option("--query", "-q", help="Research query")],
) -> None:
    """Run a real single-agent baseline using the LLM client directly."""

    _init()
    console.print("[bold green]Starting Single-Agent Baseline...[/bold green]")
    request = ResearchQuery(query=query)
    state = ResearchState(request=request)

    start_time = perf_counter()
    llm = LLMClient()
    
    system_prompt = (
        "You are a helpful research assistant. Synthesize a comprehensive response "
        "answering the user query directly. Output your answer in clean Markdown."
    )
    
    try:
        res = llm.complete(system_prompt, query)
        latency = perf_counter() - start_time
        
        state.final_answer = res.content
        state.input_tokens = res.input_tokens or 0
        state.output_tokens = res.output_tokens or 0
        state.estimated_cost_usd = res.cost_usd or 0.0

        console.print(Panel(state.final_answer, title="Single-Agent Baseline Answer", border_style="green"))
        console.print(
            f"[bold blue]Metrics:[/bold blue]\n"
            f"- Latency: {latency:.2f}s\n"
            f"- Prompt Tokens: {state.input_tokens}\n"
            f"- Completion Tokens: {state.output_tokens}\n"
            f"- Estimated Cost: ${state.estimated_cost_usd:.6f}"
        )
    except Exception as e:
        console.print(f"[bold red]Baseline execution failed:[/bold red] {e}")
        raise typer.Exit(code=1)


@app.command("multi-agent")
def multi_agent(
    query: Annotated[str, typer.Option("--query", "-q", help="Research query")],
) -> None:
    """Run the multi-agent workflow skeleton."""

    _init()
    console.print("[bold green]Starting Multi-Agent System...[/bold green]")
    state = ResearchState(request=ResearchQuery(query=query))
    workflow = MultiAgentWorkflow()
    
    start_time = perf_counter()
    try:
        result = workflow.run(state)
        latency = perf_counter() - start_time
        
        console.print(Panel(result.final_answer or "No answer produced.", title="Multi-Agent Final Answer", border_style="cyan"))
        
        console.print(
            f"[bold blue]Multi-Agent Run Execution Summary:[/bold blue]\n"
            f"- Total Iterations: {result.iteration}\n"
            f"- Route History: {' -> '.join(result.route_history)}\n"
            f"- Latency: {latency:.2f}s\n"
            f"- Total Prompt Tokens: {result.input_tokens}\n"
            f"- Total Completion Tokens: {result.output_tokens}\n"
            f"- Estimated Total Cost: ${result.estimated_cost_usd:.6f}"
        )
    except StudentTodoError as exc:
        console.print(Panel.fit(str(exc), title="Expected TODO", style="yellow"))
        raise typer.Exit(code=2) from exc
    except Exception as e:
        console.print(f"[bold red]Multi-agent execution failed:[/bold red] {e}")
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
