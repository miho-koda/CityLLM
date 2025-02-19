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
class State(TypedDict):
    messages: Annotated[list, add_messages]  # User conversation history
    task_type: str  # "file_io", "code_writer", "search_web"
    file_results: str
    code_results: str
    search_results: str



llm = ChatOpenAI(model="gpt-4-turbo", api_key=API_KEYS["openai"])    

def router(state: State):
    user_message = state["messages"][-1]["content"]
    prompt = f"""
    Analyze the user request below and decompose it into a structured workflow of interdependent tasks in JSON format:

    Request: "{user_message}"

    Task Types:
    - "file_io": For reading or writing files based on user input
    - "code_writer": For generating code or providing code-based solutions
    - "search_web": For retrieving information available online

    Return a JSON array of tasks where each task has:
    1. A descriptive task_type (one of the above)
    2. A unique integer ID (0, 1, 2)
    3. Dependencies (IDs of tasks that must complete before this one)
    4. Arguments specific to each task type:
    - file_io: "operation" ("read"/"write"), "file_path", "content" (for writes)
    - code_writer: "language", "requirements", "context"
    - search_web: "query", "focus_area"

    Example format:
    [
    {{
        "task_type": "file_io",
        "id": 1,
        "dep": [],
        "args": {{
        "operation": "read",
        "file_path": "input.csv"
        }}
    }},
    {{
        "task_type": "code_writer",
        "id": 2,
        "dep": [1],
        "args": {{
        "language": "python",
        "requirements": "Parse CSV and calculate averages"
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





# Example test inputs
test_inputs = [
    "Find the latest research on quantum computing and summarize it.",
    "Write a Python script to scrape weather data and save it to a file.",
    "Fetch stock prices for Tesla and plot a graph.",
    "Read the contents of 'report.txt' and summarize the key points.",
    "Write a function in Python that takes a list of numbers and returns the sum."
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