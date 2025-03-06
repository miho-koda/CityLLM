import json
import sys
from collections import defaultdict, deque


from agents.file_io_agent import file_io
from agents.code_agent import code_writer
from agents.search_agent import search_web
from agents.llm_extraction_agent import llm_extraction

from workflow.state import State, FileIOArgs, CodeWriterArgs, SearchWebArgs




TASK_FUNCTIONS = {
    "file_io": file_io,
    "code_writer": code_writer,
    "search_web": search_web,
    "llm_extraction": llm_extraction
}


def build_dependency_graph(tasks):
    """
    Build an adjacency list and in-degree dictionary for tasks based on dependencies.
    Returns a tuple (graph, in_degree, task_map) where:
        - graph: {task_id: [dependent_task_ids]}
        - in_degree: {task_id: number_of_dependencies}
        - task_map: {task_id: task_data}
    """
    print(tasks)
    graph = defaultdict(list)
    in_degree = {}
    task_map = {}
    
    for task in tasks:
        task_id = task["id"]
        task_map[task_id] = task
        in_degree[task_id] = len(task["dep"])
        for dep in task["dep"]:
            graph[dep].append(task_id)
            
    return graph, in_degree, task_map

def topological_sort(graph, in_degree):
    """
    Perform a topological sort on the dependency graph.
    Returns a list of task IDs in execution order.
    """
    queue = deque([task_id for task_id, deg in in_degree.items() if deg == 0])
    sorted_tasks = []
    
    while queue:
        current = queue.popleft()
        sorted_tasks.append(current)
        for neighbor in graph[current]:
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                queue.append(neighbor)
                
    if len(sorted_tasks) != len(in_degree):
        raise Exception("Cycle detected or missing dependency in tasks!")
    
    return sorted_tasks

from typing import TypedDict, Union, Literal, Annotated, Dict, Any, List

def execute_tasks(sorted_task_ids, task_map):
    """
    Execute tasks in the order provided, using a shared state that passes results between tasks.
    Each task receives its own 'args' plus outputs from its dependencies.
    """
    # Initialize shared state that will be updated after each task
    state: State = {
    "messages": [],
    "task_type": None,  # Default task type (can be changed later)
    "task_args": {},  # Empty dictionary (will be populated based on the task type)
    "dep_results": []  # Empty list for storing dependency results
    }

    
    
    results: Dict[int, Any] = {}  # Store all task results by task_id    

    for task_id in sorted_task_ids:
        task = task_map[task_id]
        task_type = task["task_type"]
        
        # update state task_type 
        state["task_type"] = task_type
        
        # update state messages
        messages = task.get("messages", [])
        state["messages"] = messages
        
        # update state task_args
        if task_type == "file_io":
            task_args = {
                "operation": task.get("args", {}).get("operation"),
                "file_path": task.get("args", {}).get("file_path"),
            }
            
            # Append "content" to messages if provided (for write operations)
            if task.get("args", {}).get("operation") == "write":
                content = task.get("args", {}).get("content", "")
                state["messages"].append({"role": "user", "content": content})
                
        elif task_type == "code_writer":
            task_args = {
                "language": task.get("args", {}).get("language"),
                "requirements": task.get("args", {}).get("requirements"),
            }
            
            # Append "context" to messages if provided
            context = task.get("args", {}).get("context", "")
            if context:
                state["messages"].append({"role": "user", "content": context})
                
        elif task_type == "search_web":
            task_args = {
                "focus_area": task.get("args", {}).get("focus_area"), 
                "instructions": task.get("args", {}).get('instructions')
            }
            
            # Append "query" to messages if provided
            query = task.get("args", {}).get("query", "")
            if query:
                state["messages"].append(query)
            
        
        elif task_type == "llm_extraction":
            task_args = {
                "instructions": task.get("args", {}).get("instructions", "")
            }

            
        # Update task_args in state
        state["task_args"] = task_args
        
        # update dep-results in state
        for index in task["dep"]:        
            state["dep_results"].append(results[index])
            

        
        print(f"\nExecuting Task {task_id} ({task_type}) with state: {state}")
        
        # Get the function for the task type
        func = TASK_FUNCTIONS.get(task_type)
        if not func:
            print(f"Unknown task type: {task_type}")
            results[task_id] = None
            continue
            
        try:
            result = func(state)
            results[task_id] = result
            print(f"Task {task_id} result: {result}")
                      
        except Exception as e:
            print(f"Error executing task {task_id}: {str(e)}")
            results[task_id] = None
    
    return results

def main(tasks):
    graph, in_degree, task_map = build_dependency_graph(tasks['tasks'])
    sorted_task_ids = topological_sort(graph, in_degree)
    
    print("Execution Order:", sorted_task_ids)
    results = execute_tasks(sorted_task_ids, task_map)
    print("\nFinal Results:")
    for task_id, result in results.items():
        print(f"Task {task_id}: {result}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python run_tasks.py tasks.json")
        sys.exit(1)
    
    json_file = sys.argv[1]
    main(json_file)





