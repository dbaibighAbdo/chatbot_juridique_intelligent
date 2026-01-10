from helpers.llm import embeddings
from helpers.graph import graph
import os
from langchain_neo4j import Neo4jVector
from dotenv import load_dotenv
load_dotenv()


neo4jvector = Neo4jVector.from_existing_index(
    embedding=embeddings,
    graph=graph,
    index_name="legal_vector_index",
)

retriever = neo4jvector.as_retriever(search_type="similarity", k=3)


def get_related_context(query: str) -> str:
    """
    Retrieve related contextual passages from the Neo4j vector index.
    Returns a clean, well-structured string in Arabic.
    """
    docs = retriever.invoke(query)

    if not docs:
        return ""

    clean_docs = []
    for doc in docs:
        # Remove unnecessary newlines and extra spaces
        text = doc.page_content.replace("\n", " ").strip()
        # Optional: collapse multiple spaces into one
        text = " ".join(text.split())
        clean_docs.append(text)

    # Use explicit separator for multiple docs to improve readability
    return "\n---\n".join(clean_docs)