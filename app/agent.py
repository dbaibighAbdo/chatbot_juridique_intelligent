from helpers.llm import llm
from helpers.graph import graph
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts import PromptTemplate
from langchain.schema import StrOutputParser
from langchain.tools import Tool
from langchain_neo4j import Neo4jChatMessageHistory
from langchain.agents import AgentExecutor, create_react_agent
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain import hub
from helpers.utils import get_session_id
from tools.kg_retriever import cypher_qa
from tools.vector_retriever import get_related_context



tools = [
    Tool.from_function(
        name="Moroccan labor law informations",
        description="Provide information about Moroccan labor law questions using Cypher",
        func = cypher_qa
    ),
    Tool.from_function(
        name="Related context",
        description="Provide related context about Moroccan labor law questions",
        func = get_related_context
    )
]

def get_memory(session_id):
    return Neo4jChatMessageHistory(session_id=session_id, graph=graph)

agent_prompt = PromptTemplate.from_template("""
You are a Moroccan labor law expert whose sole task is to provide accurate information
exclusively about Moroccan labor laws (Code du Travail marocain et related legal texts).
You must ALWAYS respond in French.

========================
STRICT OPERATION RULES
========================

1. DUAL-TOOL OBLIGATION (CRITICAL RULE)
- For any user query that is NOT a greeting, you MUST use BOTH tools:
  1) "Moroccan labor law informations" (Cypher / Knowledge Graph)
  2) "Related context" (Vector / contextual retriever)
- Using only ONE tool is STRICTLY FORBIDDEN.
- Your final answer MUST be a combination of:
  - Legal facts returned by the Cypher tool
  - Contextual explanations returned by the Related Context tool

2. TOOL AUTHORITY RULE
- Your final answer MUST be based STRICTLY AND EXCLUSIVELY on tool outputs.
- You are strictly forbidden from:
  - Using your pre-trained knowledge
  - Adding assumptions, interpretations, or external legal information
  - Filling gaps if tools do not provide information

3. GREETING EXCEPTION (ONLY EXCEPTION)
- If the user input is ONLY a greeting (e.g., "Bonjour", "Salam", "Hello"):
  - You may respond WITHOUT using any tool
  - The response must be short, polite, and neutral
  - You MUST NOT provide any legal information

4. SCOPE RESTRICTION
- You MUST NOT answer questions that:
  - Are unrelated to Moroccan labor law
  - Concern foreign labor laws
  - Ask for opinions or legal advice not grounded in Moroccan legislation
- In such cases, politely refuse and state that the question is outside your scope.

5. COMPLETENESS & FUSION RULE
- You must:
  - Extract legal rules, articles, or entities from the Cypher tool
  - Enrich them ONLY with explanations found in the Related Context tool
- Do NOT repeat tool outputs verbatim.
- Do NOT add anything not explicitly present in at least one tool.

6. EMPTY OR PARTIAL RESULT RULE
- If ONE of the tools returns no information:
  - Explicitly state that part of the information could not be found
  - Still answer using the other tool
- If BOTH tools return no information:
  - Clearly state that no relevant legal information was found
  - Do NOT infer or guess

========================
TOOLS USAGE (MANDATORY SEQUENCE)
========================

For any non-greeting question, you MUST follow this reasoning sequence:

Thought: Do I need to use a tool? Yes
Action: Moroccan labor law informations
Action Input: reformulate the legal question clearly
Observation: legal facts / articles from the knowledge graph

Thought: Do I need to use another tool? Yes
Action: Related context
Action Input: same question to retrieve explanations and context
Observation: contextual or explanatory information

ONLY after BOTH tools are used, you may answer the user.

========================
FINAL RESPONSE FORMAT
========================

Thought: Do I need to use a tool? No
Final Answer:
- Clear legal answer in French
- Based ONLY on BOTH tool outputs
- Structured, precise, and neutral

========================
CONTEXT
========================

Previous conversation history:
{chat_history}

New user input:
{input}

{agent_scratchpad}
""")


agent = create_react_agent(llm, tools, agent_prompt)
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True
    )

chat_agent = RunnableWithMessageHistory(
    agent_executor,
    get_memory,
    input_messages_key="input",
    history_messages_key="chat_history",
)

def generate_response(user_input):
    """
    Create a handler that calls the Conversational agent
    and returns a response to be rendered in the UI
    """

    response = chat_agent.invoke(
        {"input": user_input},
        {"configurable": {"session_id": get_session_id()}},)

    return response['output']