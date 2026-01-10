import os
from tools.kg_retriever import cypher_qa
from tools.vector_retriever import get_related_context
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver
from pydantic import BaseModel, Field


from langgraph.graph import END, MessagesState, START, StateGraph

class SearchQuery(BaseModel):
    search_query: str = Field(..., description="Search query for retrieval.")

class GraphAgentState(MessagesState):
    vector_db_context: str
    cypher_db_context: str
    web_context: str

llm = ChatOpenAI(model="gpt-4.1-mini", temperature=1.0, api_key=os.getenv("OPENAI_API_KEY"))

answer_generator_prompt = """You are a highly qualified Moroccan law expert.

Your sole mission is to provide precise, reliable, and legally accurate information based strictly on the provided data.

========================
LANGUAGE RULE (STRICT)
========================
- You MUST ALWAYS respond in Arabic (الفصحى).
- Do NOT use French or any other language in your final answer.

========================
QUERY CLASSIFICATION (INTERNAL RULE – STRICT)
========================
Before generating the final answer, internally determine the type of the user's query:

1. GREETING / CHITCHAT:
   - Examples: السلام عليكم، مرحبا، شكرا، كيف حالك، Bonjour, Hello
   - In this case:
     - IGNORE ALL provided legal contexts and tools outputs.
     - Respond politely and briefly in Arabic.
     - Do NOT mention any legal information.
     - Do NOT say that you are ignoring the context.

2. NON-LEGAL QUERY (OUT OF LAW FIELD):
   - Any question that is not related to Moroccan law, legal rules, legal procedures, or legal rights.
   - In this case:
     - IGNORE ALL provided legal contexts and tools outputs.
     - Respond EXACTLY with the following sentence (Arabic only):
       "عذرًا، لا يمكنني الإجابة عن الأسئلة التي لا تندرج ضمن المجال القانوني."

3. LEGAL QUERY (MOROCCAN LAW ONLY):
   - Questions related strictly to Moroccan law.
   - ONLY in this case, use the provided legal contexts below.

========================
CONTENT RULES (STRICT – FOR LEGAL QUERIES ONLY)
========================
- Base your answer ONLY on the extracted legal information and context provided below.
- Do NOT add assumptions, interpretations, or external knowledge.
- Do NOT fabricate legal rules or articles.
- Do NOT mix personal opinions with legal facts.

========================
STRUCTURE & STYLE REQUIREMENTS
========================
- The answer must be clear, well-structured, and easy to read.
- Use rich paragraphs and, when appropriate, bullet points.
- Use the maximum detail possible from the provided information related to the question.
- Use formal and professional legal Arabic.
- Ensure logical flow and coherence.
- Avoid redundancy and unnecessary explanations.

========================
INSUFFICIENT INFORMATION RULE (LEGAL QUERIES ONLY)
========================
- If the provided legal information is insufficient to answer the legal question, respond EXACTLY with:
  "عذرًا، لا تتوفر معلومات كافية للإجابة على سؤالك بناءً على البيانات المقدمة"

========================
 Extracted Legal Information (Neo4j / Cypher):
========================
{cypher_db_context}

========================
 Extracted Context (Vector Database):
========================
{vector_db_context}

========================
 Extracted Web Search Results (if any):
========================
{web_context}

========================
 User Question:
========================
{user_input}

========================
✍️ Final Answer (Arabic only):
========================

"""

def cypher_retriever(state: GraphAgentState):
    try:
        graph_result = cypher_qa.invoke({"query": state["messages"][-1].content})
        graph_answer = graph_result.get("result", "").strip()
        if not graph_answer:
            graph_answer = ""
    except Exception:
        graph_answer = ""

    return {"cypher_db_context": graph_answer}


def vector_retriever(state: GraphAgentState):
    try:
        context = get_related_context(str(state["messages"][-1].content))
        if not context:
            context = ""
    except Exception:
        context = ""

    return {"vector_db_context": context}



search_instructions = SystemMessage(content=f"""You will be given a conversation between an user and an assistant. 

Your goal is to generate a well-structured ARABIC query for use in retrieval and / or web-search related to the conversation.
First, analyze the full conversation.
Convert this final question into a well-structured web search query""")

llm_for_search = ChatOpenAI(model="gpt-4o-mini", temperature=1.0, api_key=os.getenv("OPENAI_API_KEY"))


def search_web(state: GraphAgentState): 
    """ Retrieve docs from web search """ 
    tavily_search = TavilySearchResults(max_results=3) 
    # Search query 
    structured_llm = llm_for_search.with_structured_output(SearchQuery) 
    search_query = structured_llm.invoke([search_instructions]+state['messages']) 
    # Search 
    search_docs = tavily_search.invoke(search_query.search_query) 
    # Format 
    formatted_search_docs = "\n\n---\n\n".join( [ f'<Document href="{doc["url"]}"/>\n{doc["content"]}\n</Document>' for doc in search_docs ] ) 
    return {"web_context": [formatted_search_docs]}



def get_final_answer(state: GraphAgentState):
    prompt = answer_generator_prompt.format(
        cypher_db_context=state["cypher_db_context"],
        vector_db_context=state["vector_db_context"],
        web_context=state["web_context"],
        user_input=state["messages"][-1].content
    )

    response = llm.invoke([
        SystemMessage(content=prompt)
    ])

    return {"messages": [response]}




graph = StateGraph(GraphAgentState)

graph.add_node("Cypher Retrieval", cypher_retriever)
graph.add_node("Vector Retrieval", vector_retriever)
graph.add_node("Web Search Retrieval", search_web)
graph.add_node("Final Answer Generation", get_final_answer)

graph.add_edge(START, "Cypher Retrieval")
graph.add_edge(START, "Vector Retrieval")
graph.add_edge(START, "Web Search Retrieval")
graph.add_edge("Cypher Retrieval", "Final Answer Generation")
graph.add_edge("Vector Retrieval", "Final Answer Generation")
graph.add_edge("Web Search Retrieval", "Final Answer Generation")
graph.add_edge("Final Answer Generation", END)

memory = InMemorySaver()
workflow = graph.compile(checkpointer=memory)

import uuid  # Add this import

def generate_response(user_input: str, config):
    # Append new user message
    messages = [HumanMessage(content=user_input)]
    
    input_state = {
        "messages": messages,
        "vector_db_context": "",
        "cypher_db_context": "",
        "web_context": ""
    }
    
    # Stream the graph
    for event in workflow.stream(input_state, config):
        pass  # State accumulates in checkpointer
    
    # Get final state and return last assistant message
    final_state = workflow.get_state(config)
    final_messages = final_state.values.get("messages", [])
    
    # Find last non-retriever message (assistant response)
    for msg in reversed(final_messages):
        if hasattr(msg, 'content') and msg.content:  # HumanMessage/AIMessage
            return msg.content
    
    return "No response generated"
