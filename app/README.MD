# App — Streamlit Chatbot

This folder contains the Streamlit front-end and agent code for the Chatbot Juridique Intelligent.

## Purpose
The Streamlit app provides a simple chat interface to ask legal questions about Moroccan labor law. It uses the project's agent modules and retrievers to gather information from a knowledge graph and vector store.

## How to run (development)
1. Make sure dependencies are installed and your virtual environment is activated (see project root `README.md`).
2. Create a `.env` in the project root with required environment variables (example below).
3. Start the app:

	```powershell
	streamlit run app/bot.py
	```

4. Open the local URL printed by Streamlit in your browser.

## Important environment variables
- `OPENAI_API_KEY` — API key for LLM access
- `NEO4J_URI`, `NEO4J_USER`, `NEO4J_PASSWORD` — (optional) Neo4j connection for knowledge-graph retrieval

## Files of interest
- `bot.py` — Streamlit UI, session management, and call to `agent.generate_response`
- `agent.py` — contains core conversation/agent logic used by the app
- `helpers/` — utilities used across the app (LLM wrappers, writing messages, graph helpers)
- `tools/` — retriever implementations (`kg_retriever.py`, `vector_retriever.py`)

## Notes and Recommendations
- The UI includes Arabic support via `streamlit-arabic-support-wrapper`; if you see rendering issues, check that the package is installed and that your environment supports Arabic fonts.
- The app stores messages in `st.session_state` and assigns a `thread_id` per session for continuity across agent calls.

If you'd like, I can add a sample `.env.example` file, create a `requirements.txt`, or update `bot.py` to include a language selector and Arabic default. Tell me which addition you'd prefer.
