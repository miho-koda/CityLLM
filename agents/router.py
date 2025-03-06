from langchain_openai import ChatOpenAI  # OpenAI model
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from config import API_KEYS
import json


from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages
from typing import TypedDict, Union, Literal, Annotated

from workflow.state import State, FileIOArgs, CodeWriterArgs, SearchWebArgs


llm = ChatOpenAI(model="gpt-4-turbo", api_key=API_KEYS["openai"])    

def router(user_message):
    prompt = f"""
    You are an advanced task planner responsible for breaking down a high-level user request into a structured, interdependent workflow using a minimal number of AI agents while ensuring **maximal efficiency**.

    ### **User Request**
    "{user_message}"

    ### **Your Task**
    Analyze the request and decompose it into a **logical sequence of interdependent tasks**, ensuring that:
    - **Dependencies are correctly defined** (a task may depend on the output of multiple tasks).
    - **The fewest number of AI agents are used** while maintaining **optimal performance**.
    - **Tasks are ordered efficiently** to avoid redundant computation.
    - **Minimize multiple calls to `code_writer` by writing a single, comprehensive script that can handle multiple operations at once.**

    ### **Available Task Types**
    Your decomposition should use the following structured task types:

    #### **1. file_io** (Read/Write operations)
    - Used when reading from or writing to files.
    - **Arguments:**
    - `operation`: Must be either `"read"` or `"write"`.
    - **For `"read"`:** Specify the file to read from using `file_path`.
    - **For `"write"`:** Specify `file_path` and the `content` to be written.
    - `file_path`: The file's location (e.g., `"data/input.csv"`).
    - `content` (for `"write"` only): The actual data to write (e.g., a string, JSON, or DataFrame).

    #### **2. code_writer** (Generate a single, efficient script)
    - Used for writing **one** optimized script that performs multiple tasks.
    - **Arguments:**
    - `language`: The programming language to use (default to `"python"`).
    - `requirements`: A clear description of what the code should accomplish.
    - `context`: Any **necessary input data, constraints, or dependencies**.
    - `input_files`: A list of files required for the script.
    - `output_files`: A list of expected output files.

    #### **3. search_web** (Retrieve external information)
    - Used when information retrieval is necessary.
    - **Arguments:**
    - `query`: The specific web search to perform.
    - `focus_area`: The domain of interest (e.g., `"urban planning"`, `"environmental data"`).
    - `instructions`: How to **process** the search result using LLM (e.g., `"Extract the names and coordinates of metro stations"`).

    ---

    ### **Task Dependencies & Efficiency Rules**
    - **Minimize redundant `code_writer` calls**: Instead of multiple code-writing steps, generate a **single Python script** that:
    - Reads all required files.
    - Processes data as needed.
    - Outputs all final results.
    - **Tasks should be dependent only when necessary.** Use the minimum number of dependencies to maximize efficiency.
    - **A task can have multiple dependencies** (e.g., the output of two different tasks can be combined as input for another task).
    - **Ensure outputs are reusable** to avoid unnecessary recomputation.

    ---

    ### **Example JSON Output**
    Return a structured JSON list where each task has:
    1. A descriptive `"task_type"` (one of the above).
    2. A unique integer `"id"` (e.g., `0, 1, 2, ...`).
    3. A `"dep"` array specifying which tasks must complete before this one.
    4. An `"args"` dictionary with task-specific arguments.

    ```json
    [
        {{
            "task_type": "file_io",
            "id": 0,
            "dep": [],
            "args": {{
                "operation": "read",
                "file_path": "data/city_traffic.csv"
            }}
        }},
        {{
            "task_type": "file_io",
            "id": 1,
            "dep": [],
            "args": {{
                "operation": "read",
                "file_path": "data/road_network.geojson"
            }}
        }},
        {{
            "task_type": "search_web",
            "id": 2,
            "dep": [],
            "args": {{
                "query": "Current public transport stations in San Francisco",
                "focus_area": "city planning",
                "instructions": "Extract the names, latitudes, and longitudes of all public transport stations."
            }}
        }},
        {{
            "task_type": "code_writer",
            "id": 3,
            "dep": [0, 1, 2],
            "args": {{
                "language": "python",
                "requirements": "Analyze congestion hotspots, generate public transport accessibility zones, and determine underserved areas in one script.",
                "context": "Use GeoPandas for spatial analysis and overlay congestion data onto road networks. Compare areas with high congestion to public transport buffer zones to identify underserved regions.",
                "input_files": ["data/city_traffic.csv", "data/road_network.geojson", "transport_stations.json"],
                "output_files": ["output/congestion_hotspots.geojson", "output/underserved_areas.geojson"]
            }}
        }},
        {{
            "task_type": "file_io",
            "id": 4,
            "dep": [3],
            "args": {{
                "operation": "write",
                "file_path": "output/underserved_areas.geojson",
                "content": "Results from Task 3"
            }}
        }}
    ]```

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

