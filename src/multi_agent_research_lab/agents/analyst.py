import logging
from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.services.llm_client import LLMClient

logger = logging.getLogger(__name__)


class AnalystAgent(BaseAgent):
    """Turns research notes into structured insights."""

    name = "analyst"

    def __init__(self, llm_client: LLMClient | None = None) -> None:
        self.llm_client = llm_client or LLMClient()

    def run(self, state: ResearchState) -> ResearchState:
        """Populate `state.analysis_notes`."""
        research_notes = state.research_notes or "No research notes available."
        query = state.request.query

        logger.info("Analyst: Analyzing research notes and synthesizing insights.")

        system_prompt = (
            "You are the Analyst Agent. Your task is to analyze raw research notes and extract "
            "deep, structured insights.\n"
            "1. Extract key assertions or technical claims.\n"
            "2. Evaluate the strength of evidence supporting each claim.\n"
            "3. Identify conflicting viewpoints or gaps in the notes.\n"
            "4. Maintain correct citation references (e.g. [1], [2]) pointing to their original sources.\n"
            "Be objective, highly critical, and structured."
        )
        user_prompt = f"Target Query: {query}\n\nResearch Notes:\n{research_notes}"

        try:
            res = self.llm_client.complete(system_prompt, user_prompt)
            
            # Record LLM usage in state metrics
            state.input_tokens += res.input_tokens or 0
            state.output_tokens += res.output_tokens or 0
            state.estimated_cost_usd += res.cost_usd or 0.0

            state.analysis_notes = res.content
            logger.info("Analyst: Successfully generated analysis notes.")
        except Exception as e:
            logger.error(f"Analyst run failed: {e}")
            state.errors.append(f"Analyst error: {e}")

        state.add_trace_event("analyst", {})
        return state
