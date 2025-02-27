from typing import TypedDict, Union, Literal, Annotated, List, Dict, Any, Optional

# Optional input for all tasks
class BaseTaskArgs(TypedDict, total=False):
    input: Optional[Union[str, Dict, List]]  # Optional input data from a previous task

# Task-specific argument structures
class FileIOArgs(BaseTaskArgs):
    operation: Literal["read", "write"]
    file_path: str
    content: Optional[Union[str, Dict, List]]  # Only used for "write" operations

class CodeWriterArgs(BaseTaskArgs):
    language: str
    requirements: str
    context: Optional[Union[str, Dict, List]]  # Additional context for the task

class SearchWebArgs(BaseTaskArgs):
    focus_area: str

class LLMExtractionArgs(BaseTaskArgs):
    instructions: str  

# Task type literals
TaskType = Literal["file_io", "code_writer", "search_web", "llm_extraction"]

# Union type for all possible task arguments
TaskArgs = Union[FileIOArgs, CodeWriterArgs, SearchWebArgs, LLMExtractionArgs]

# Define the main state class with task-specific result types
class State(TypedDict):
    messages: Annotated[List[Dict[str, Any]], "add_messages"]  # User conversation history
    task_type: TaskType
    task_args: TaskArgs  # Task-specific arguments
    
    # Results from various operations
    file_results: Optional[Any]  # Results from file operations
    code_results: Optional[Any]  # Results from code generation
    search_results: Optional[Any]  # Results from web searches
    llm_results: Optional[Any]  # Results from LLM extraction