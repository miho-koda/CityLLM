from langchain_openai import ChatOpenAI  # OpenAI model
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from config import API_KEYS
from workflow import graph
import json


from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages
from typing import TypedDict, Union, Literal, Annotated

from workflow.state import State, FileIOArgs, CodeWriterArgs, SearchWebArgs


llm = ChatOpenAI(model="gpt-4-turbo", api_key=API_KEYS["openai"])    

def router(state: State):
    user_message = state["messages"][-1]["content"]
    prompt = f"""
    Analyze the user request below and decompose it into a structured workflow of interdependent tasks in JSON format:

    Request: "{user_message}"

    Task Types:
    1. **file_io**: For reading from or writing to files.
    - Arguments:
        - `operation`: Must be either `"read"` or `"write"`.
        - For `"read"`: Specify the file to read from using `file_path`.
        - For `"write"`: Specify the file to write to using `file_path` and provide the `content` to write.
        - `file_path`: The path to the file (e.g., `"data/input.csv"`).
        - `content` (only for `"write"`): The data to write to the file (e.g., a string, JSON, or DataFrame).

    2. **code_writer**: For generating code or providing code-based solutions.
    - Arguments:
        - `language`: The programming language to use (e.g., `"python"`, `"R"`, `"javascript"`).
        - `requirements`: A clear description of what the code should accomplish (e.g., `"Parse a CSV file and calculate averages"`).
        - `context`: Additional context or input data needed for the task (e.g., file paths, sample data, or specific instructions).

    3. **search_web**: For retrieving information from the web.
    - Arguments:
        - `query`: The search query to execute (e.g., `"public transport stations near [latitude, longitude]"`).
        - `focus_area`: The domain or focus area of the search (e.g., `"city planning"`, `"environmental data"`).

    4. **llm_extraction**: For processing raw data or text using an LLM to extract relevant information.
    - Arguments:
        - `input`: The raw data or text from a previous task (e.g., `"Raw search results from task 0"`).
        - `instructions`: Specific instructions for the LLM to extract or process the input data (e.g., `"Extract the names, latitudes, and longitudes of public transport stations"`).

    Return a JSON array of tasks where each task has:
    1. A descriptive `task_type` (one of the above).
    2. A unique integer `id` (0, 1, 2, ...).
    3. `dep`: An array of task IDs that must complete before this one (empty if no dependencies).
    4. `args`: A dictionary of task-specific arguments as described above.

    Example format:
    [
        {{
            "task_type": "file_io",
            "id": 0,
            "dep": [],
            "args": {{
                "operation": "read",
                "file_path": "data/input.csv"
            }}
        }},
        {{
            "task_type": "llm_extraction",
            "id": 1,
            "dep": [0],
            "args": {{
                "input": "Raw data from task 0",
                "instructions": "Extract relevant columns and clean the data"
            }}
        }},
        {{
            "task_type": "code_writer",
            "id": 2,
            "dep": [1],
            "args": {{
                "language": "python",
                "requirements": "Calculate averages and generate a summary report",
                "context": "Use the cleaned data from task 1"
            }}
        }}
    ]

    Ensure tasks are logically sequenced and dependencies are correctly specified to produce the desired result.
    Return only valid, executable JSON.
    """

    # Invoke LLM to generate structured tasks
    tasks_json = llm.invoke(prompt).content
    if tasks_json.startswith("```json"):
        tasks_json = tasks_json[7:]  # Remove the first 7 characters (` ```json `)
    if tasks_json.endswith("```"):
        tasks_json = tasks_json[:-3]
        
    # Attempt to parse the response as JSON
    try:
        structured_tasks = json.loads(tasks_json)
        if isinstance(structured_tasks, list):  
            return {"tasks": structured_tasks}
    except json.JSONDecodeError:
        pass  # If LLM output is invalid, default to a fallback task

    # Default to a web search task if the response is invalid or ambiguous
    return {
        "tasks": [{
            "task_type": "search_web",
            "id": "default_search",
            "dep": [],
            "args": {"query": user_message}
        }]
    }




# TEST SCRIPT 

test_inputs = [
    "Search for public transport stations near cambridge massachusetts, extract their names and coordinates, and generate a map visualizing their locations.",
    #"Compare land use data from two different years (stored in separate shapefiles) and identify areas where land use has changed."
]

def test_router():
    for i, user_input in enumerate(test_inputs, 1):
        print(f"\n **Test Case {i}:** {user_input}")
        
        # Mocking state format expected by router function
        state = {"messages": [{"role": "user", "content": user_input}]}

        # Call the router function
        result = router(state)

        # Pretty-print JSON output
        print("🔍 **Decomposed Tasks:**")
        print(json.dumps(result, indent=4))

# Run the test
test_router()
