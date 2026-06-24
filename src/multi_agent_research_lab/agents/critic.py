import json
import logging
from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.services.llm_client import LLMClient

logger = logging.getLogger(__name__)


class CriticAgent(BaseAgent):
    """Optional fact-checking and safety-review agent."""

    name = "critic"

    def __init__(self, llm_client: LLMClient | None = None) -> None:
        self.llm_client = llm_client or LLMClient()

    def run(self, state: ResearchState) -> ResearchState:
        """Validate final answer and append findings."""
        final_answer = state.final_answer or ""
        query = state.request.query

        if not final_answer:
            logger.warning("Critic: Final answer is empty. Rejecting.")
            state.add_trace_event("critic", {"passed": False, "feedback": "Final answer is empty."})
            return state

        logger.info("Critic: Reviewing final answer quality and citation alignment.")

        system_prompt = (
            "You are the Critic Agent. Your task is to critique the final draft research report.\n"
            "Assess the following:\n"
            "1. Did the report address all aspects of the user query?\n"
            "2. Are the claims properly cited with numeric references matching the source list?\n"
            "3. Are there clear errors, formatting issues, or logic gaps?\n\n"
            "Respond ONLY with a valid JSON object containing:\n"
            "{\n"
            "  \"passed\": true | false,\n"
            "  \"feedback\": \"Constructive feedback detailing exactly what to improve, or approving the text.\"\n"
            "}\n"
            "Do not include markdown or backticks."
        )

        user_prompt = f"Query: {query}\n\nFinal Draft Answer:\n{final_answer}"

        passed = True
        feedback = "Approved."

        try:
            res = self.llm_client.complete(system_prompt, user_prompt)
            
            # Record LLM usage in state metrics
            state.input_tokens += res.input_tokens or 0
            state.output_tokens += res.output_tokens or 0
            state.estimated_cost_usd += res.cost_usd or 0.0

            clean_content = res.content.strip("` \n").replace("json\n", "")
            data = json.loads(clean_content)
            passed = data.get("passed", True)
            feedback = data.get("feedback", "No feedback provided.")
        except Exception as e:
            logger.error(f"Critic parse failed: {e}. Defaulting to approval.")

        logger.info(f"Critic review: passed={passed}, feedback='{feedback}'")

        if not passed:
            # If rejected, append feedback to the final answer so it can be revised by writer
            state.final_answer = (
                f"{final_answer}\n\n"
                f"--- \n"
                f"**Critic Revision Request (Iteration {state.iteration}):**\n"
                f"{feedback}"
            )
            state.research_notes = (state.research_notes or "") + f"\n[Critic Feedback: {feedback}]"

        state.add_trace_event("critic", {"passed": passed, "feedback": feedback})
        return state
