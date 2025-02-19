from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages

from agents.router import router
from agents.file_io_agent import file_io
from agents.search_agent import search_web
from agents.code_agent import code_writer


class State(TypedDict):
    messages: Annotated[list, add_messages]  # User conversation history
    task_type: str  # "file_io", "code_writer", "search_web"
    file_results: str
    code_results: str
    search_results: str


from langgraph.graph import StateGraph, START, END

graph_builder = StateGraph(State)

# Add nodes
graph_builder.add_node("router", router)
graph_builder.add_node("file_io", file_io)
graph_builder.add_node("code_writer", code_writer)
graph_builder.add_node("search_web", search_web)

# Define edges
graph_builder.add_edge(START, "router")

graph_builder.add_edge("router", "file_io", condition=lambda s: s["task_type"] == "file_io")
graph_builder.add_edge("router", "code", condition=lambda s: s["task_type"] == "code_writer")
graph_builder.add_edge("router", "search_web", condition=lambda s: s["task_type"] == "search_web")

graph_builder.add_edge("file_io", "response_builder")
graph_builder.add_edge("code", "response_builder")
graph_builder.add_edge("search_web", "response_builder")

graph_builder.add_edge("response_builder", END)

graph = graph_builder.compile()
