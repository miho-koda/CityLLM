import sys
import os

from langchain_community.tools.tavily_search import TavilySearchResults

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from config import API_KEYS  

from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages
class State(TypedDict):
    messages: Annotated[list, add_messages]  # User conversation history
    task_type: str  # "file_io", "code_writer", "search_web"
    file_results: str
    code_results: str
    search_results: str

os.environ["TAVILY_API_KEY"] = API_KEYS["tavily"]

tavily_search = TavilySearchResults(max_results=5, include_answer=True, include_raw_content=True, include_images=True)

def search_web(state: State):
    """
    Uses TavilySearchResults to fetch relevant online information.
    """
    search_query = state["messages"][-1]["content"]
    search_results = tavily_search.invoke({"query": search_query})
    
    return {"search_results": search_results}
