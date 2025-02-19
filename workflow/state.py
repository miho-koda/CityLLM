from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages

class State(TypedDict):
    messages: Annotated[list, add_messages]  # User conversation history
    task_type: str  # "file_io", "code_writer", "search_web"
    file_results: str
    code_results: str
    search_results: str
