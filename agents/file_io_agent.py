import os


from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages
from typing import TypedDict, Union, Literal, Annotated

from workflow.state import State, FileIOArgs, CodeWriterArgs, SearchWebArgs





import pandas as pd

def read_file_as_dataframe(file_path):
    """
    Reads a file and returns its content as a Pandas DataFrame.
    Supports CSV, Excel, and JSON files.
    """
    if not os.path.exists(file_path):
        return {"file_results": "File not found."}

    try:
        # Determine the file type based on the extension
        if file_path.endswith(".csv"):
            df = pd.read_csv(file_path)
        elif file_path.endswith(".xlsx"):
            df = pd.read_excel(file_path)
        elif file_path.endswith(".json"):
            df = pd.read_json(file_path)
        else:
            return {"file_results": "Unsupported file format."}

        return df

    except Exception as e:
        return {"file_results": f"Error reading file: {str(e)}"}



def file_io(state: State):
    """
    Handles file operations (reading/writing).
    """
    #file_path = "/Users/mihokoda/Desktop/CityLLM/code files/test data/data.txt"
    file_path = state["task_args"]["file_path"]


    # Read file
    if state["task_args"]["operation"] == "read":
        result = read_file_as_dataframe(file_path)
        return result
  

    # Write file THIS NEEDS UPDATING. IT SHOULD HAVE INPUT OF A DF, RETURN AS CSV/XSLX? 
    elif state["task_args"]["operation"] == "write":
        user_request = state["messages"][-1]["content"]
        with open(file_path, "w") as file:
            file.write(user_request)
        return {"file_results": "File updated successfully."}

    return {"file_results": "Invalid file operation."}



'''
#TEST SCRIPT
import os
import json

# Ensure the directory exists for file operations
file_path = "/Users/mihokoda/Desktop/CityLLM/code files/test data/data.txt"
os.makedirs(os.path.dirname(file_path), exist_ok=True)

# Sample test inputs
test_cases = [
    {"content": "write Hello, this is a test!", "expected": "File updated successfully."},
    {"content": "read", "expected": "Hello, this is a test!"},  # This should return the content we wrote
    {"content": "write Another test message", "expected": "File updated successfully."},
    {"content": "read", "expected": "Another test message"},  # This should return the latest content
    {"content": "delete", "expected": "Invalid file operation."},  # Invalid operation test
]

# Function to run tests
def test_file_io():
    for i, test in enumerate(test_cases, 1):
        print(f"\n📂 **Test Case {i}:** {test['content']}")

        # Mock state format expected by file_io function
        state = {"messages": [{"role": "user", "content": test["content"]}]}

        # Call the file_io function
        result = file_io(state)

        # Pretty-print JSON output
        print("📄 **File I/O Result:**")
        print(json.dumps(result, indent=4))

        # Check if result matches expected output
        if result["file_results"] == test["expected"]:
            print("✅ Test Passed!")
        else:
            print(f"❌ Test Failed! Expected: {test['expected']}")

# Run the test
test_file_io()
'''