from langchain_community.tools.tavily_search import TavilySearchResults

from langchain_core.tools import tool 
@tool("web_search", return_direct=True)
def web_search(query)-> str: 
    """ Retrieve docs from web search """ 
    tavily_search = TavilySearchResults(max_results=3) 
    # Search 
    search_docs = tavily_search.invoke(query) 
    # Format 
    formatted_search_docs = "\n\n---\n\n".join( [ f'<Document href="{doc["url"]}"/>\n{doc["content"]}\n</Document>' for doc in search_docs ] ) 
    return formatted_search_docs