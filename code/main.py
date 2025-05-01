import os
import time
import geopandas as gpd
import gzip


import pandas as pd
import numpy as np
from shapely.geometry import Point, Polygon
from shapely.wkt import loads as load_wkt
from shapely.ops import unary_union
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
import glob
import functools
import argparse
from zone_agent import ZoneAgent

from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
import os
import time
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

def run_single_prompt(prompt_text, llm_name, test_case_idx, args, test_case_folder):
    """
    Run a single prompt through the ZoneAgent, save zone ids inside specific test case folder,
    compare with objective.csv inside the same folder, update a central logistics file.
    """
    print(f"\n=== Processing Test Case {test_case_idx} ===")
    print(f"Prompt: {prompt_text}")

    agent = ZoneAgent(
        mode=args.mode,
        model_name=args.model,
        max_steps=args.max_steps,
        max_retries=args.max_retries
    )

    test_case = f"test_case_{test_case_idx}"

    print(f"\nAnalyzing {test_case}: {prompt_text}")
    print("-" * 80)

    start_time = time.time()
    
    answer, scratchpad = agent.run(prompt_text)
    if answer: 
        result_status = "success"
        execution_successful = True
    else:
        result_status = "execution_error"
        execution_successful = False
    generation_time = time.time() - start_time

    print("\nAnalysis Results:")
    print(answer)
    print("\nReasoning Process:")
    print(scratchpad)
    print("=" * 80)

    # Save scratchpad to file - now this will run even if agent.run() fails
    os.makedirs(test_case_folder, exist_ok=True)
    scratchpad_path = os.path.join(test_case_folder, f"{test_case}_scratchpad.txt")
    with open(scratchpad_path, 'w', encoding='utf-8') as f:
        f.write(scratchpad)
    print(f"✅ Saved scratchpad to {scratchpad_path}")

    import re

    try:
        # Only process and save zone IDs if execution was successful
        if execution_successful:
            # Check if answer is a string that looks like a list
            if isinstance(answer, str):
                answer_clean = answer.strip('[]{}')

                # Try extracting numbers if any
                numbers_found = re.findall(r'\d+', answer_clean)
                
                if numbers_found:
                    zone_ids_list = [int(x) for x in numbers_found]
                    zone_ids_result = pd.Series(zone_ids_list, name="zone_id")
                else:
                    # If no numbers found, just an empty list
                    zone_ids_result = pd.Series([], name="zone_id")

                # Save result zone ids
                os.makedirs(test_case_folder, exist_ok=True)
                save_path = os.path.join(test_case_folder, f"{test_case}_zone_ids.csv")
                zone_ids_result.to_csv(save_path, index=False)
                print(f"✅ Saved survived zone ids to {save_path}")

                # Load objective.csv
                objective_path = os.path.join(test_case_folder, "objective.csv")
                if not os.path.exists(objective_path):
                    print(f"❌ Objective file {objective_path} does not exist. Skipping comparison.")
                    result_status = "objective_missing"
                else:
                    objective_df = pd.read_csv(objective_path)
                    zone_ids_objective = objective_df["zone_id"].dropna().astype(str).reset_index(drop=True)
                    zone_ids_objective = zone_ids_objective.sort_values(ignore_index=True)

                    zone_ids_result = zone_ids_result.dropna().astype(str).reset_index(drop=True)
                    zone_ids_result = zone_ids_result.sort_values(ignore_index=True)

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
                print(f"❌ Answer is not a string, it's a {type(answer)}")
                result_status = "invalid_answer_type"

    except Exception as e:
        print(f"❌ Error during parsing/comparison: {e}")
        result_status = "comparison_error"

    # === Update central logistics file ===
    logistics_path = "/Users/mihokoda/Desktop/CityLLM/test_results/logistics.csv"

    new_entry = pd.DataFrame([{
        "test_case": test_case,
        "llm": llm_name,
        "generation_time": round(generation_time, 2),
        "execution_time": "-",
        "comparison": result_status,
        "prompt": prompt_text,
        "test_case_folder": test_case_folder
    }])

    if os.path.exists(logistics_path):
        logistics_df = pd.read_csv(logistics_path)
        logistics_df = pd.concat([logistics_df, new_entry], ignore_index=True)
    else:
        logistics_df = new_entry

    logistics_df.to_csv(logistics_path, index=False)
    print(f"✅ Logged {test_case} | {llm_name}: {result_status} | Gen: {round(generation_time, 2)}s")

def run_all_prompts(prompts, llm_name, args, test_case_folders):
    """
    Run multiple prompts sequentially, each directed to its own test case folder.
    """
    for idx, prompt_text in enumerate(prompts):

        run_single_prompt(
                prompt_text,
                llm_name,
                idx,
                args,
                test_case_folders[idx]
        )
def prep(prompts, difficulty, i):
    args = parse_arguments()

    base_folder = "/Users/mihokoda/Desktop/CityLLM/test_results"

    # Only run inside sim/1
    difficulty_path = os.path.join(base_folder, difficulty, str(i))  # <--- only sim/1

    if not os.path.isdir(difficulty_path):
        print(f"❌ Folder does not exist: {difficulty_path}")
        return

    # Build the test case folders based on the index of simple_1
    test_case_folders = []
    for idx in range(len(prompts)):
        folder_name = f"tc_{difficulty}_{i}_{idx}"
        full_test_case_path = os.path.join(difficulty_path, folder_name)
        if os.path.isdir(full_test_case_path):
            test_case_folders.append(full_test_case_path)
        else:
            print(f"⚠️ Warning: Folder does not exist: {full_test_case_path}")

    print(f"▶️ Running prompts for {difficulty_path}")
    run_all_prompts(prompts, args.model, args, test_case_folders)

# Example prompts

simple_13 = [
    "Show me zones where at least 3 types of transportation are available in the zone.",
    "I want to open a logistics hub — find zones with at least 4 distinct transportation types nearby.",
    "Highlight areas where 3 or more transportation types exist within the zone.",
    "Filter for zones with access to at least 2 types of transportation options like subway, taxi, and bus.",
    "Identify zones where at least 3 transportation types are available for easy commuter access.",
    "Find me locations where the zone supports 4 or more different types of transportation.",
            
]

simple_14 = [
    "Find zones where at least 40% of POIs are in sub category {Beauty Salons}.",
    "I'm looking for areas where 35% or more of all POIs are in top category {Restaurants and Other Eating Places}.",
    "Show me zones where over 50% of POIs fall under sub category {Snack and Nonalcoholic Beverage Bars}.",
    "Can you locate zones where at least 60%+ of POIs belong to top category {Offices of Physicians}?",
    "Looking for zones where at least 45% of POIs are in sub category {Educational Support Services}.",
    "I need zones where 30% or more of POIs are under top category {Personal Care Services}.",
        

]

simple_15 = [
    "Show me zones where sub category {Beauty Salons} is the singular most common POI type.",
    "I'm looking for areas where sub category {Full-Service Restaurants} is the dominant sub category in terms of POI count.",
    "Find zones where the singular most common POI type is sub category {Snack and Nonalcoholic Beverage Bars}.",
    "Can you highlight areas where sub category {Offices of Dentists} outnumber all other sub categories?",
    "I'm trying to find zones where the leading POI sub category is {Gasoline Stations with Convenience Stores}.",
    "Identify zones where sub category {Art Dealers} appears more than any other.",
        
]

simple_16 = [
    "I want to open a medical clinic where over 50% of the total spending in 2023 comes from top category {Offices of Physicians}.",
    "Find zones where at least 40% of the transaction volume in 2022 is from sub category {Couriers and Express Delivery Services}.",
    "Show me areas where 60% of the total dollars spent in 2023 went to top category {Legal Services}.",
    "I’m targeting zones where over 35% of the total sales in 2022 come from sub category {Used Car Dealers}.",
    "Looking for regions where 45% of all customers in 2023 interacted with sub category {Offices of Lawyers}.",
    "Where in the city does top category {Other Amusement and Recreation Industries} contribute over 55% of the spending in 2022?",

]

simple_17 = [
    "Show me areas that contain at least 25 POIs — I'm planning a community café there.",
    "I need a zone with a minimum of 20 places of interest for my coworking space idea.",
    "Which zones have at least 15 establishments? I'm considering setting up a fitness studio.",
    "I want to find areas where there are at least 60 POIs — good for foot traffic and visibility.",
    "I'm launching a pet grooming service and need a zone with 20 or more active businesses.",
    "Highlight zones that have 5+ POIs total — I want a lively place for a music lounge.",
            
]

simple_18 = [
    "I want areas where top category {Offices of Physicians} doesn’t dominate more than 15% of POIs.",
    "Show me zones where the sub category {Offices of Lawyers} is no more than 20%.",
    "Avoid zones where top category {Offices of Physicians} takes up more than 9% of businesses.",
    "Give me zones where no sub category like {Fitness and Recreational Sports Centers} accounts for more than 25%.",
    "Looking for diverse areas—no single top category such as {Offices of Other Health Practitioners} should go above 20%.",
    "I want to skip zones where the sub category {Investment Advice} represents more than 60% of POIs.",
            
]
if __name__ == "__main__":
    for i in range(17, 19):            
        prep(eval(f"simple_{i}"), "sim", i)
    
if __name__ == "__main__":
    main()