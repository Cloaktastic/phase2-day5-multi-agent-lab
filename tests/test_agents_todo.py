import pytest

from multi_agent_research_lab.agents import SupervisorAgent
from multi_agent_research_lab.core.errors import StudentTodoError
from multi_agent_research_lab.core.schemas import ResearchQuery
from multi_agent_research_lab.core.state import ResearchState


def test_supervisor_routing() -> None:
    state = ResearchState(request=ResearchQuery(query="Explain multi-agent systems"))
    updated_state = SupervisorAgent().run(state)
    assert len(updated_state.route_history) == 1
    assert updated_state.route_history[0] in ["researcher", "analyst", "writer", "critic", "done"]
