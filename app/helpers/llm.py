import os
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
import dotenv

dotenv.load_dotenv()

embeddings = OpenAIEmbeddings(
    model="text-embedding-ada-002",
)
llm = ChatOpenAI(
    model="gpt-4o-mini",
)