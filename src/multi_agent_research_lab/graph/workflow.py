import logging
from langgraph.graph import StateGraph, END
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.agents.supervisor import SupervisorAgent
from multi_agent_research_lab.agents.researcher import ResearcherAgent
from multi_agent_research_lab.agents.analyst import AnalystAgent
from multi_agent_research_lab.agents.writer import WriterAgent
from multi_agent_research_lab.agents.critic import CriticAgent
from multi_agent_research_lab.services.llm_client import LLMClient

logger = logging.getLogger(__name__)


class MultiAgentWorkflow:
    """Builds and runs the multi-agent graph.

    Keep orchestration here; keep agent internals in `agents/`.
    """

    def __init__(self) -> None:
        self.llm_client = LLMClient()
        self.supervisor = SupervisorAgent(llm_client=self.llm_client)
        self.researcher = ResearcherAgent(llm_client=self.llm_client)
        self.analyst = AnalystAgent(llm_client=self.llm_client)
        self.writer = WriterAgent(llm_client=self.llm_client)
        self.critic = CriticAgent(llm_client=self.llm_client)
        self.compiled_graph = self.build()

    def build(self) -> StateGraph:
        """Create a LangGraph graph."""
        logger.info("Building multi-agent LangGraph workflow...")
        workflow = StateGraph(ResearchState)

        # Add nodes (agents)
        workflow.add_node("supervisor", self.supervisor.run)
        workflow.add_node("researcher", self.researcher.run)
        workflow.add_node("analyst", self.analyst.run)
        workflow.add_node("writer", self.writer.run)
        workflow.add_node("critic", self.critic.run)

        # entry node is the supervisor
        workflow.set_entry_point("supervisor")

        # worker nodes always route back to supervisor to check progress
        workflow.add_edge("researcher", "supervisor")
        workflow.add_edge("analyst", "supervisor")
        workflow.add_edge("writer", "supervisor")
        workflow.add_edge("critic", "supervisor")

        # supervisor uses conditional edges
        def route_decision(state: ResearchState) -> str:
            if not state.route_history:
                logger.warning("No route history found in state. Routing back to supervisor.")
                return "supervisor"
            
            return state.route_history[-1]

        workflow.add_conditional_edges(
            "supervisor",
            route_decision,
            {
                "researcher": "researcher",
                "analyst": "analyst",
                "writer": "writer",
                "critic": "critic",
                "done": END,
            }
        )

        return workflow.compile()

    def run(self, state: ResearchState) -> ResearchState:
        """Execute the graph and return final state."""
        logger.info("Executing multi-agent LangGraph workflow...")

        # Setup tracing callbacks if available
        callbacks = []
        try:
            from multi_agent_research_lab.observability.tracing import get_langfuse_handler
            handler = get_langfuse_handler()
            if handler:
                callbacks.append(handler)
        except Exception as e:
            logger.debug(f"Langfuse callback handler load skipped: {e}")

        # Run the workflow
        result = self.compiled_graph.invoke(
            state,
            config={"callbacks": callbacks}
        )

        # Flush any callback handlers to ensure traces are delivered before process exits
        for cb in callbacks:
            if hasattr(cb, "flush"):
                try:
                    logger.info("Flushing Langfuse traces...")
                    cb.flush()
                except Exception as e:
                    logger.debug(f"Failed to flush callback: {e}")

        # Ensure we return a proper ResearchState
        if isinstance(result, dict):
            return ResearchState(**result)
        return result
