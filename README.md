# Chatbot Juridique Intelligent

An intelligent legal chatbot focused on Moroccan labor law. This project provides a Streamlit-powered chat assistant that uses language models and knowledge-graph/ vector retrieval to answer legal questions, help construct knowledge graphs from PDFs, and support Arabic UI rendering.

**Key features**
- **Specialized domain:** tailored to Moroccan labor law and related regulations.
- **Multilingual UI:** Arabic support via `streamlit-arabic-support-wrapper`.
- **Hybrid retrieval:** uses Neo4j knowledge graph and vector retrieval for context-aware answers.
- **Streamlit interface:** lightweight web app for interactive conversations.

**Quick Start**
1. Install Python 3.12 or newer.
2. Create a virtual environment and activate it:

	```powershell
	python -m venv .venv; .\.venv\Scripts\Activate.ps1
	```

3. Install dependencies (uses `pyproject.toml`):

	```powershell
	pip install -U pip
	pip install -e .
	```

4. Prepare environment variables (create a `.env` file in project root). Typical variables:

	- `OPENAI_API_KEY` — API key for OpenAI (or whichever LLM provider is used)
	- `NEO4J_URI`, `NEO4J_USER`, `NEO4J_PASSWORD` — Neo4j connection details (if using knowledge graph features)

5. Run the Streamlit app:

	```powershell
	streamlit run app/bot.py
	```

**Project Structure (important files)**
- `app/` — Streamlit application and agents
  - `bot.py` — Streamlit UI entrypoint
  - `agent.py` — agent logic and orchestration
  - `helpers/` — utility functions, llm wrappers, graph helpers
  - `tools/` — retrievers and knowledge-graph connectors
- `kg_construction/` — helpers and scripts to build a knowledge graph from PDFs
- `pyproject.toml` — project metadata and dependencies

**Environment & Notes**
- The project targets Python 3.12+ as declared in `pyproject.toml`.
- If you don't use the Neo4j-based retrieval, configure the app to use only vector retrieval or mock connectors.
- The app includes Arabic UI support; ensure your browser and OS can render Arabic text correctly.

**Contributing**
- Please open issues or pull requests. Follow standard GitHub workflows and add tests for non-trivial changes.

**License**
- Add a license file or choose a license and update this section accordingly.

---
If you want, I can also generate a `requirements.txt` from `pyproject.toml`, add a sample `.env` file, or update the Streamlit prompts and translations. Which would you like next?