import os
from langchain_openai import ChatOpenAI
from neo4j_graphrag.embeddings import OpenAIEmbeddings

import dotenv

dotenv.load_dotenv()

embeddings = OpenAIEmbeddings(
    model="text-embedding-ada-002",
    api_key=os.getenv("OPENAI_API_KEY")
)
llm = ChatOpenAI(
    model="gpt-4o-mini",
    api_key=os.getenv("OPENAI_API_KEY")
)