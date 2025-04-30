# import pandas as pd
# import numpy as np
# from shapely.geometry import Point, Polygon
# from shapely.wkt import loads as load_wkt
# from shapely.ops import unary_union
# from concurrent.futures import ProcessPoolExecutor, as_completed
# import multiprocessing
# import glob
# import os
# import time
# import functools
# import argparse
# from zone_agent import ZoneAgent

# from site_selection.demand import demand
# from site_selection.parking import parking
# from site_selection.competition import competition
# from site_selection.competition import filter_zones_by_average_rating
# from site_selection.accessibility import accessibility


# from shapely.geometry import Point, Polygon, MultiPoint
# from scipy.spatial import ConvexHull
# from sklearn.cluster import MiniBatchKMeans


# from shapely.wkt import loads as load_wkt
# from shapely.geometry import Point



# import pandas as pd
# import geopandas as gpd
# from shapely.geometry import Point
# import gzip

# # Get the directory where this script is located
# base_dir = os.path.dirname(os.path.abspath(__file__))

# # === Run each benchmark scenario ===
# if __name__ == "__main__":
#     # Parse command line arguments
#     parser = argparse.ArgumentParser(description='Zone Analysis Framework')
#     parser.add_argument('--mode', type=str, default='zero_shot',
#                       choices=['zero_shot', 'chain_of_thought', 'react'],
#                       help='Mode of operation')
#     parser.add_argument('--model', type=str, default='gpt-3.5-turbo',
#                       help='Language model to use')
#     parser.add_argument('--max_steps', type=int, default=30,
#                       help='Maximum number of steps')
#     parser.add_argument('--max_retries', type=int, default=3,
#                       help='Maximum number of retries per action')
#     args = parser.parse_args()

#     # Initialize the agent
#     agent = ZoneAgent(
#         mode=args.mode,
#         max_steps=args.max_steps,
#         max_retries=args.max_retries,
#         model_name=args.model
#     )

#     # Example queries
#     example_queries = [
#         # Simple queries
#         "I want to build a POI in a zone where there are at least 1 parking lot with 10 parking spaces.",
#     ]

#     # Run analysis for each query
#     for i, query in enumerate(example_queries, 1):
#         print(f"\nAnalyzing Query {i}:")
#         print(f"Query: {query}")
#         print("-" * 80)
        
#         answer, scratchpad = agent.run(query)
        
#         print("\nAnalysis Results:")
#         print(answer)
#         print("\nReasoning Process:")
#         print(scratchpad)
#         print("=" * 80)


import pandas as pd
import numpy as np
from shapely.geometry import Point, Polygon
from shapely.wkt import loads as load_wkt
from shapely.ops import unary_union
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
import glob
import os
import time
import functools
import argparse
from zone_agent import ZoneAgent

from site_selection.demand import demand
from site_selection.parking import parking
from site_selection.competition import competition
from site_selection.competition import filter_zones_by_average_rating
from site_selection.accessibility import accessibility


from shapely.geometry import Point, Polygon, MultiPoint
from scipy.spatial import ConvexHull
from sklearn.cluster import MiniBatchKMeans


from shapely.wkt import loads as load_wkt
from shapely.geometry import Point



import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
import gzip

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Zone Analysis Framework')
    parser.add_argument('--mode', type=str, default='zero_shot',
                      choices=['zero_shot', 'chain_of_thought', 'react'],
                      help='Mode of operation')
    parser.add_argument('--model', type=str, default='gpt-4-turbo',
                      help='Language model to use')
    parser.add_argument('--max_steps', type=int, default=30,
                      help='Maximum number of steps')
    parser.add_argument('--max_retries', type=int, default=3,
                      help='Maximum number of retries per action')
    parser.add_argument('--output_path', type=str, default='output',
                      help='Path to save results')
    return parser.parse_args()

def run_single_prompt(prompt_text, output_path, llm_name, test_case_idx, args):
    """
    Run a single prompt through the ZoneAgent, save zone ids, compare with objective, and update logistics.
    """
    print(f"\n=== Processing Test Case {test_case_idx} ===")
    print(f"Prompt: {prompt_text}")

    # Create specific output directory for this test case
    case_output_path = os.path.join(output_path, str(test_case_idx))
    os.makedirs(case_output_path, exist_ok=True)

    # === Step 1: Initialize agent ===
    agent = ZoneAgent(
        mode=args.mode,
        model_name=args.model,
        max_steps=args.max_steps,
        max_retries=args.max_retries
    )

    test_case = f"test_case_{test_case_idx}"

    # === Step 2: Run the agent ===
    print(f"\nAnalyzing {test_case}: {prompt_text}")
    print("-" * 80)
    
    start_time = time.time()
    answer, scratchpad = agent.run(prompt_text)
    generation_time = time.time() - start_time

    print("\nAnalysis Results:")
    print(answer)
    print("\nReasoning Process:")
    print(scratchpad)
    print("=" * 80)

    # === Step 3: Parse survived zone_ids from answer ===
    try:
        if "Survived Zones" in answer:
            zone_ids_text = answer.split(":")[1].strip()
            zone_ids_list = eval(zone_ids_text)  # careful: assume trusted output
            zone_ids_result = pd.Series(zone_ids_list, name="zone_id")

            # === Step 4: Save survived zone ids ===
            save_path = os.path.join(case_output_path, "react_result.csv")
            zone_ids_result.to_csv(save_path, index=False)
            print(f"✅ Saved survived zone ids to {save_path}")

            # === Step 5: Load objective zone ids ===
            objective_path = os.path.join(case_output_path, "objective.csv")
            if not os.path.exists(objective_path):
                print(f"❌ Objective file {objective_path} does not exist. Skipping comparison.")
                result_status = "objective_missing"
            else:
                objective_df = pd.read_csv(objective_path)
                zone_ids_objective = objective_df["zone_id"].dropna().astype(str).reset_index(drop=True)
                zone_ids_objective = zone_ids_objective.sort_values(ignore_index=True)

                zone_ids_result = zone_ids_result.dropna().astype(str).reset_index(drop=True)
                zone_ids_result = zone_ids_result.sort_values(ignore_index=True)

                # === Step 6: Compare ===
                if zone_ids_result.equals(zone_ids_objective):
                    result_status = "same"
                    print("✅ zone_id columns match exactly.")
                else:
                    result_status = "different"
                    print("❌ zone_id columns differ.")
                    only_in_result = zone_ids_result[~zone_ids_result.isin(zone_ids_objective)]
                    only_in_objective = zone_ids_objective[~zone_ids_objective.isin(zone_ids_result)]

                    print("\n🔍 zone_ids only in result:")
                    print(only_in_result.head(10))

                    print("\n🔍 zone_ids only in objective:")
                    print(only_in_objective.head(10))
        else:
            print("❌ No survived zones found in answer")
            result_status = "no_survived_zones"

    except Exception as e:
        print(f"❌ Error during parsing/comparison: {e}")
        result_status = "comparison_error"

    # === Step 7: Update logistics ===
    logistics_path = os.path.join(output_path, "logistics.csv")

    new_entry = pd.DataFrame([{
        "test_case": test_case,
        "llm": llm_name,
        "generation_time": round(generation_time, 2),
        "execution_time": "-",
        "comparison": result_status,
        "prompt": prompt_text
    }])

    if os.path.exists(logistics_path):
        logistics_df = pd.read_csv(logistics_path)
        logistics_df = pd.concat([logistics_df, new_entry], ignore_index=True)
    else:
        logistics_df = new_entry

    logistics_df.to_csv(logistics_path, index=False)
    print(f"✅ Logged {test_case} | {llm_name}: {result_status} | Gen: {round(generation_time, 2)}s")

def run_all_prompts(prompts, output_path, llm_name, args):
    """
    Run multiple prompts sequentially.
    """
    for idx, prompt_text in enumerate(prompts):
        run_single_prompt(prompt_text, output_path, llm_name, idx, args)

def main():
    # Parse command line arguments
    args = parse_arguments()

    # Set base output path
    base_output_path = "/Users/mihokoda/Desktop/CityLLM/test_results/sim/1"
    os.makedirs(base_output_path, exist_ok=True)

    # Run all prompts
    run_all_prompts(simple_1, base_output_path, args.model, args)


simple_1 = [
    "I want to build a POI in a zone where there are at least 10 parking spaces.",
    # "I want to open a new restaurant, but I need a location with at least 50 parking spots nearby.",
    # "Looking for a spot to build a shopping mall—must have at least 200 parking spaces.",
    # "I’m planning to construct a medical clinic. Are there zones with at a parking lot with least 30 parking spots available?",
    # "Where could I put a new grocery store? It needs a parking lot with at least 80 parking spaces.",
    # "I need to find a location for a movie theater, ideally a parking lot with 150+ parking spots.",
]   

if __name__ == "__main__":
    main()