from helpers.graph import graph
from langchain_openai import OpenAIEmbeddings

from langchain_neo4j import Neo4jVector
from dotenv import load_dotenv
load_dotenv()


neo4jvector = Neo4jVector.from_existing_index(
    embedding=OpenAIEmbeddings(model="text-embedding-ada-002"),                              
    graph=graph,                             
    index_name="vector_nodes",        # Node index
    keyword_index_name="keyword_nodes",
    search_type="similarity",
    embedding_node_property="embedding",
)

retriever = neo4jvector.as_retriever()


def get_related_context(input):
    return retriever.invoke(input, kwargs={'k': 5})[0]