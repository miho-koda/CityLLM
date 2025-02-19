#COME BACK TO THIS
'''
import argparse
from workflow.graph import build_workflow
from agents.file_io_agent import load_dataset
from agents.search_agent import search_geospatial_info
from agents.code_exec_agent import generate_geospatial_code
from storage.output_results import save_results

def main(task_request):
    """
    Entry point for running the geospatial analysis workflow.
    """
    print(f"🛠️ Running task: {task_request}")

    # Step 1: Initialize the workflow graph
    workflow = build_workflow()

    # Step 2: Load dataset if available
    dataset = load_dataset("storage/input_data/nybb.geojson")
    
    if dataset is None:
        print("⚠️ Dataset not found. Searching for alternatives...")
        search_results = search_geospatial_info("New York Boroughs dataset")
        print(f"🔍 Found data sources: {search_results}")
    
    # Step 3: Generate and execute geospatial analysis code
    code = generate_geospatial_code(task_request)
    print("🖥️ Generated Code:\n", code)

    # Step 4: Run workflow execution
    workflow.run(task_request)

    # Step 5: Store results
    save_results(task_request)

    print("✅ Task completed successfully.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a geospatial analysis task.")
    parser.add_argument("task_request", type=str, help="Describe the geospatial analysis you want to perform.")
    args = parser.parse_args()

    main(args.task_request)

    


'''