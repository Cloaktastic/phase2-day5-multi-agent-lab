import logging
from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.services.llm_client import LLMClient
from multi_agent_research_lab.services.search_client import SearchClient

logger = logging.getLogger(__name__)


class ResearcherAgent(BaseAgent):
    """Collects sources and creates concise research notes."""

    name = "researcher"

    def __init__(self, llm_client: LLMClient | None = None, search_client: SearchClient | None = None) -> None:
        self.llm_client = llm_client or LLMClient()
        self.search_client = search_client or SearchClient(llm_client=self.llm_client)

    def run(self, state: ResearchState) -> ResearchState:
        """Populate `state.sources` and `state.research_notes`."""
        query = state.request.query
        max_sources = state.request.max_sources

        logger.info(f"Researcher: Executing search for query: '{query}'")
        sources = self.search_client.search(query, max_results=max_sources)
        state.sources.extend(sources)

        # De-duplicate sources by URL or title
        seen = set()
        unique_sources = []
        for src in state.sources:
            key = src.url or src.title
            if key not in seen:
                seen.add(key)
                unique_sources.append(src)
        state.sources = unique_sources

        # Format sources as context
        context_str = ""
        for idx, src in enumerate(state.sources, 1):
            context_str += f"Source [{idx}]:\nTitle: {src.title}\nURL: {src.url or 'N/A'}\nContent: {src.snippet}\n\n"

        system_prompt = (
            "You are the Researcher Agent. Your task is to analyze the gathered search results and "
            "compile comprehensive, structured research notes.\n"
            "Include key definitions, technical details, facts, and relevant claims. "
            "Cite sources explicitly in your notes using brackets like [1], [2], corresponding to their indices. "
            "Do not fabricate any information. If details are missing, state so clearly."
        )

        user_prompt = f"Query: {query}\n\nRetrieved Search Results:\n{context_str}"

        try:
            res = self.llm_client.complete(system_prompt, user_prompt)
            
            # Record LLM usage in state metrics
            state.input_tokens += res.input_tokens or 0
            state.output_tokens += res.output_tokens or 0
            state.estimated_cost_usd += res.cost_usd or 0.0

            state.research_notes = res.content
            logger.info("Researcher: Successfully compiled research notes.")
        except Exception as e:
            logger.error(f"Researcher run failed: {e}")
            state.errors.append(f"Researcher error: {e}")

        state.add_trace_event("researcher", {"sources_count": len(state.sources)})
        return state
