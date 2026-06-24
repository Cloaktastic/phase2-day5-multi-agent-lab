import json
import logging
from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.services.llm_client import LLMClient
from multi_agent_research_lab.core.config import get_settings

logger = logging.getLogger(__name__)


class SupervisorAgent(BaseAgent):
    """Decides which worker should run next and when to stop."""

    name = "supervisor"

    def __init__(self, llm_client: LLMClient | None = None) -> None:
        self.llm_client = llm_client or LLMClient()
        self.settings = get_settings()

    def run(self, state: ResearchState) -> ResearchState:
        """Update `state.route_history` with the next route.
        
        Evaluates the current state (research, analysis, final answer) and routes
        to researcher, analyst, writer, critic, or done.
        """
        # Guard against exceeding maximum iterations
        if state.iteration >= self.settings.max_iterations:
            logger.warning(f"Supervisor: Max iterations ({self.settings.max_iterations}) reached. Forcing 'done'.")
            state.record_route("done")
            state.add_trace_event("supervisor", {"decision": "done", "reason": "Max iterations reached"})
            return state

        # Create a summary of current research state
        state_summary = {
            "query": state.request.query,
            "audience": state.request.audience,
            "iteration": state.iteration,
            "route_history": state.route_history,
            "has_sources": len(state.sources) > 0,
            "has_research_notes": state.research_notes is not None and len(state.research_notes) > 0,
            "has_analysis_notes": state.analysis_notes is not None and len(state.analysis_notes) > 0,
            "has_final_answer": state.final_answer is not None and len(state.final_answer) > 0,
        }

        system_prompt = (
            "You are the Supervisor Agent of a Multi-Agent Research System.\n"
            "Your role is to decide the next step in the research workflow. Select exactly one of the following next agents:\n"
            "- 'researcher': If you do not have sufficient raw research sources or need to gather facts.\n"
            "- 'analyst': If you have research notes but have not yet synthesized key claims, evidence, or structured insights.\n"
            "- 'writer': If you have research/analysis insights but haven't written the final markdown response, or need to compile it.\n"
            "- 'critic': If a final answer is drafted and needs fact-checking or citation review.\n"
            "- 'done': If the critic has approved or the final answer is completely accurate, well-structured, properly cited, and answers the query.\n\n"
            "Respond ONLY with a valid JSON object containing:\n"
            "{\n"
            "  \"next_agent\": \"researcher\" | \"analyst\" | \"writer\" | \"critic\" | \"done\",\n"
            "  \"reason\": \"A concise explanation of why this routing choice was made.\"\n"
            "}\n"
            "Do not include markdown or backticks."
        )

        user_prompt = f"Current workflow state:\n{json.dumps(state_summary, indent=2)}"

        next_agent = "done"
        reason = "Fallback"
        try:
            res = self.llm_client.complete(system_prompt, user_prompt)
            
            # Record LLM usage in state metrics
            state.input_tokens += res.input_tokens or 0
            state.output_tokens += res.output_tokens or 0
            state.estimated_cost_usd += res.cost_usd or 0.0

            clean_content = res.content.strip("` \n").replace("json\n", "")
            data = json.loads(clean_content)
            next_agent = data.get("next_agent", "done").lower()
            reason = data.get("reason", "Parsed routing choice.")
        except Exception as e:
            logger.error(f"Supervisor routing failed: {e}. Executing default fallback routing.")
            # Fallback state machine logic
            if len(state.sources) == 0:
                next_agent = "researcher"
                reason = "Fallback: No sources collected."
            elif not state.research_notes:
                next_agent = "researcher"
                reason = "Fallback: No research notes."
            elif not state.analysis_notes:
                next_agent = "analyst"
                reason = "Fallback: No analysis notes."
            elif not state.final_answer:
                next_agent = "writer"
                reason = "Fallback: No final answer written."
            else:
                next_agent = "done"
                reason = "Fallback: Finished."

        # Validate agent name
        allowed = ["researcher", "analyst", "writer", "critic", "done"]
        if next_agent not in allowed:
            logger.warning(f"Supervisor generated invalid agent '{next_agent}'. Defaulting to 'done'.")
            next_agent = "done"

        logger.info(f"Supervisor decision: {next_agent} (Reason: {reason})")
        state.record_route(next_agent)
        state.add_trace_event("supervisor", {"decision": next_agent, "reason": reason})
        return state
