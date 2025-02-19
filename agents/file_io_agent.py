import os


from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages
class State(TypedDict):
    messages: Annotated[list, add_messages]  # User conversation history
    task_type: str  # "file_io", "code_writer", "search_web"
    file_results: str
    code_results: str
    search_results: str



def file_io(state: State):
    """
    Handles file operations (reading/writing).
    """
    user_request = state["messages"][-1]["content"]
    file_path = "storage/input_data/data.txt"

    # Read file
    if "read" in user_request.lower():
        if os.path.exists(file_path):
            with open(file_path, "r") as file:
                content = file.read()
            return {"file_results": content}
        return {"file_results": "File not found."}

    # Write file
    elif "write" in user_request.lower():
        with open(file_path, "w") as file:
            file.write(user_request)
        return {"file_results": "File updated successfully."}

    return {"file_results": "Invalid file operation."}
