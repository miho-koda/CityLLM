from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages

from agents.router import router
from agents.file_io_agent import file_io
from agents.search_agent import search_web
from agents.code_agent import code_writer
from agents.response_builder_agent import response_builder  # Import the response_builder function
from agents.llm_extraction_agent import llm_extraction  # New agent for LLM extraction



from workflow.state import State, FileIOArgs, CodeWriterArgs, SearchWebArgs

from langgraph.graph import StateGraph, START, END

graph_builder = StateGraph(State)

# Add nodes
graph_builder.add_node("router", router)
graph_builder.add_node("file_io", file_io)
graph_builder.add_node("code_writer", code_writer)
graph_builder.add_node("search_web", search_web)
graph_builder.add_node("llm_extraction", llm_extraction)
graph_builder.add_node("response_builder", response_builder)   


# Define edges
graph_builder.add_edge(START, "router")

graph_builder.add_conditional_edges(
    "router",
    lambda state: state["task_type"],  # Condition based on task_type
    {
        "file_io": "file_io",  # If task_type is "file_io", go to "file_io" node
        "code_writer": "code_writer",  # If task_type is "code_writer", go to "code_writer" node
        "search_web": "search_web",  # If task_type is "search_web", go to "search_web" node
        "llm_extraction": "llm_extraction",  # If task_type is "llm_extraction", go to "llm_extraction" node
    },
)

# graph_builder.add_edge("router", "file_io", condition=lambda s: s["task_type"] == "file_io")
# graph_builder.add_edge("router", "code", condition=lambda s: s["task_type"] == "code_writer")
# graph_builder.add_edge("router", "search_web", condition=lambda s: s["task_type"] == "search_web")
# graph_builder.add_edge("router", "llm_extraction", condition=lambda s: s["task_type"] == "llm_extraction")  # New edge



graph_builder.add_edge("file_io", "response_builder")
graph_builder.add_edge("code_writer", "response_builder")
graph_builder.add_edge("search_web", "response_builder")
graph_builder.add_edge("llm_extraction", "response_builder")  # New edge


graph_builder.add_edge("response_builder", END)

graph = graph_builder.compile()
