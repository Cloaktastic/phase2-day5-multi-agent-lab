import logging
from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.services.llm_client import LLMClient

logger = logging.getLogger(__name__)


class WriterAgent(BaseAgent):
    """Produces final answer from research and analysis notes."""

    name = "writer"

    def __init__(self, llm_client: LLMClient | None = None) -> None:
        self.llm_client = llm_client or LLMClient()

    def run(self, state: ResearchState) -> ResearchState:
        """Populate `state.final_answer`."""
        query = state.request.query
        audience = state.request.audience
        research_notes = state.research_notes or "No research notes available."
        analysis_notes = state.analysis_notes or "No analysis notes available."

        logger.info(f"Writer: Writing final response for audience: '{audience}'")

        # Compile references section
        references_str = ""
        if state.sources:
            references_str = "\n## References\n"
            for idx, src in enumerate(state.sources, 1):
                references_str += f"- [{idx}] **{src.title}**"
                if src.url:
                    references_str += f" - [Source Link]({src.url})"
                references_str += "\n"

        system_prompt = (
            f"You are the Writer Agent. Your task is to write a final, comprehensive, and polished "
            f"research summary tailored specifically to the target audience: '{audience}'.\n"
            "Format the output beautifully in Markdown, including clear headings, bullet points, "
            "and logical sections.\n"
            "Ensure you integrate the facts from the research notes and structured insights from "
            "the analysis notes.\n"
            "Keep the bracketed citation markers (e.g., [1], [2]) intact within the text. "
            "Do not output a References section at the end; that will be appended automatically."
        )

        user_prompt = (
            f"User Query: {query}\n\n"
            f"Research Notes:\n{research_notes}\n\n"
            f"Analysis Notes:\n{analysis_notes}"
        )

        try:
            res = self.llm_client.complete(system_prompt, user_prompt)
            
            # Record LLM usage in state metrics
            state.input_tokens += res.input_tokens or 0
            state.output_tokens += res.output_tokens or 0
            state.estimated_cost_usd += res.cost_usd or 0.0

            state.final_answer = res.content + references_str
            logger.info("Writer: Successfully generated final answer.")
        except Exception as e:
            logger.error(f"Writer run failed: {e}")
            state.errors.append(f"Writer error: {e}")

        state.add_trace_event("writer", {})
        return state
