# Quickstart Guide: Multi-Agent Research System

This guide explains how to set up, configure, run, and evaluate the Multi-Agent Research Lab application.

---

## 1. Setup & Environment Activation

First, ensure your virtual environment is activated in your terminal:

```powershell
# Windows PowerShell
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate
```

Next, ensure you have all requirements installed:

```powershell
pip install -e ".[dev,llm]" langfuse
```

---

## 2. Configuration (`.env`)

Open the [.env](file:///.env) file in the root directory and configure your keys:

* **OpenAI API Key**: Set `OPENAI_API_KEY` to run the baseline and multi-agent systems via OpenAI.
* **Ollama (Optional Local LLM)**: To run the system locally using Ollama:
  1. Set `USE_OLLAMA=true`.
  2. Set `OLLAMA_MODEL="llama3.2:1b"` (or `llama3:8b`, `qwen2.5:1.5b`, etc. depending on your pulled model).
  3. Ensure Ollama is running (`ollama run <model_name>`).
* **Langfuse Tracing (Optional)**: Fill in `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY`, and `LANGFUSE_HOST` to enable execution tracing.
* **Tavily Search (Optional)**: Fill in `TAVILY_API_KEY` to use live search. If not provided, the system will fall back to simulated search.

---

## 3. Running the CLI Applications

### Run Single-Agent Baseline
The baseline agent uses a single direct call to the LLM to write a summary:
```powershell
python -m multi_agent_research_lab.cli baseline --query "Research GraphRAG state-of-the-art and write a 500-word summary"
```

### Run Multi-Agent System
The multi-agent system orchestrates a Supervisor, Researcher, Analyst, Writer, and Critic using a LangGraph workflow:
```powershell
python -m multi_agent_research_lab.cli multi-agent --query "Research GraphRAG state-of-the-art and write a 500-word summary"
```

---

## 4. Benchmarking & Evaluation

To run a direct performance evaluation comparing the single-agent baseline against the multi-agent system:

```powershell
python scripts/run_eval.py
```

This script will run both workflows, rate their quality using an LLM-as-a-judge, compute costs and citation coverages, and output a detailed comparison report to [reports/benchmark_report.md](file:///reports/benchmark_report.md).

---

## 5. Running Unit Tests

To run the unit tests:

```powershell
pytest
```
or
```powershell
python -m pytest
```
