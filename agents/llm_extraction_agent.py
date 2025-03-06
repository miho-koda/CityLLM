from workflow.state import State, FileIOArgs, CodeWriterArgs, SearchWebArgs

import os
import sys
from langchain.chat_models import ChatOpenAI
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from config import API_KEYS


from typing import Dict, Any, List
from langchain.schema import HumanMessage


llm = ChatOpenAI(model="gpt-4-turbo", api_key=API_KEYS["openai"])

def llm_extraction(state: State):
    """
    Process LLM extraction task based on the provided state.
    
    Args:
        state: The current application state containing task arguments and search results
        
    Returns:
        The direct response from the LLM
    """    
    # Get the instructions and input from task_args
    instructions = state.get("task_args", {}).get("instructions", "")
    input_data = state['dep_results']
    
    # Prepare the prompt
    prompt = f"{instructions}\n\nData:\n{input_data}"
    
    # Call the LLM
    response = llm([HumanMessage(content=prompt)])
    
    # Return the raw response content
    return response.content