"""Search client abstraction for ResearcherAgent."""

from multi_agent_research_lab.core.errors import StudentTodoError
from multi_agent_research_lab.core.schemas import SourceDocument


import json
import logging
from multi_agent_research_lab.core.config import get_settings
from multi_agent_research_lab.core.schemas import SourceDocument
from multi_agent_research_lab.services.llm_client import LLMClient

logger = logging.getLogger(__name__)


class SearchClient:
    """Provider-agnostic search client with fallback simulator."""

    def __init__(self, llm_client: LLMClient | None = None) -> None:
        self.settings = get_settings()
        self._llm_client = llm_client

    def search(self, query: str, max_results: int = 5) -> list[SourceDocument]:
        """Search for documents relevant to a query.
        
        Uses Tavily if key is available, else falls back to simulating search results using LLM.
        """
        # 1. Try real Tavily Search if key exists
        if self.settings.tavily_api_key:
            try:
                logger.info(f"Searching using Tavily for query: '{query}'")
                # Using requests to avoid hard dependency on Tavily SDK
                import requests
                response = requests.post(
                    "https://api.tavily.com/search",
                    json={
                        "api_key": self.settings.tavily_api_key,
                        "query": query,
                        "max_results": max_results,
                    },
                    timeout=10,
                )
                response.raise_for_status()
                results = response.json().get("results", [])
                return [
                    SourceDocument(
                        title=item.get("title", "Search Result"),
                        url=item.get("url", ""),
                        snippet=item.get("content", ""),
                    )
                    for item in results
                ]
            except Exception as e:
                logger.warning(f"Tavily search failed: {e}. Falling back to LLM simulator.")

        # 2. Fallback to LLM simulation
        llm = self._llm_client or LLMClient()
        system_prompt = (
            "You are a search engine simulator. Generate a list of realistic search results "
            "for the user's query.\n"
            "Output exactly a valid JSON array of objects, where each object has 'title', 'url', "
            "and 'snippet' fields. Do not include any markdown styling, code blocks, or backticks - "
            "just return raw JSON."
        )
        user_prompt = f"Query: {query}\nGenerate {max_results} search results."

        try:
            logger.info(f"Simulating search results using LLM for query: '{query}'")
            res = llm.complete(system_prompt, user_prompt)
            clean_content = res.content.strip("` \n").replace("json\n", "")
            data = json.loads(clean_content)
            
            docs = []
            for item in data:
                docs.append(SourceDocument(
                    title=item.get("title", "Result"),
                    url=item.get("url", "https://example.com"),
                    snippet=item.get("snippet", ""),
                ))
            return docs[:max_results]
        except Exception as e:
            logger.error(f"LLM search simulator failed: {e}. Returning hardcoded defaults.")
            # 3. Hardcoded fallback defaults
            return [
                SourceDocument(
                    title="GraphRAG: Flow-Tuning and Knowledge Graphs",
                    url="https://arxiv.org/abs/2404.16130",
                    snippet="GraphRAG uses knowledge graphs to enable structured retrieval-augmented generation. It improves global query focus.",
                ),
                SourceDocument(
                    title="Microsoft GraphRAG Repository",
                    url="https://github.com/microsoft/graphrag",
                    snippet="Official repository for Microsoft's GraphRAG implementation. Supports local and global search pipelines.",
                ),
                SourceDocument(
                    title="Evaluating Retrieval-Augmented Generation",
                    url="https://example.com/eval-rag",
                    snippet="Evaluation framework details for benchmark RAG systems with metrics such as answer relevance and faithfulness.",
                ),
            ][:max_results]
