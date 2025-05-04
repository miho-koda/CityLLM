import sys
import os

from langchain_community.tools.tavily_search import TavilySearchResults

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from config import API_KEYS  

from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages

from workflow.state import State, FileIOArgs, CodeWriterArgs, SearchWebArgs

os.environ["TAVILY_API_KEY"] = API_KEYS["tavily"]

tavily_search = TavilySearchResults(max_results=5, include_answer=True, include_raw_content=True, include_images=True)

def search_web(state: State):
    """
    Uses TavilySearchResults to fetch relevant online information.
    """
    search_query = state["messages"][0]
    search_results = tavily_search.invoke({"query": search_query})
    
    return {"search_results": search_results}


'''
#TEST SCRIPT 

import json
test_queries = [
    "Latest advancements in AI for 2025",
    "Who won the last FIFA World Cup?",
    "What are the key features of Python 3.12?",
    "Explain the theory of general relativity in simple terms",
    "Current stock price of Tesla",
]

def test_search_agent():
    for i, query in enumerate(test_queries, 1):
        print(f"\n **Test Case {i}:** {query}")

        state = {"messages": [{"role": "user", "content": query}]}

        result = search_web(state)

        # Pretty-print JSON output
        print("🔎 **Search Results:**")
        print(json.dumps(result, indent=4))

# Run the test
test_search_agent()
'''