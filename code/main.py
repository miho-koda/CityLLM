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

from site_selection.analysis import get_spendparam_years, get_num_parking, get_largest_parking_lot_area, get_largest_parking_capacity, get_distance_km
from site_selection.filter import filter_df_based_on_zone, filter_pois_by_top_category, filter_pois_by_sub_category, get_transport_pois_in_zone
from site_selection.loader import get_parking_dataset, get_poi_spend_dataset
from site_selection.zone import assign_poi_zones, create_zone, assign_parking_zones, get_neighbor_zones, get_zone_center

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
        if idx == 3 or idx == 4 or idx == 5:   
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

if __name__ == "__main__":
    for i in range(1, 19):            
        prep(eval(f"med{i}"), "med", i)
    

medium_1 = [
    "I’m looking for zones where raw total spend from 2019 to 2021 was more than $22 million.",
    "Identify areas where over 400,000 transactions took place between 2020 and 2022.",
    "Show me zones with at least 150,000 unique customers across the years 2021 to 2023.",
    "Analyze zones where raw total spend from 2020 to 2022 exceeded $18 million.",
    "Can you find districts that had more than 500,000 transactions from 2019 through 2021?",
    "I want to look at zones where the number of customers from 2021 to 2024 went beyond 300,000.",
]

medium_2 = [
    "Looking for zones where the average median spend per transaction from 2020 to 2023 was above $45 — aiming for a mid-range retail spot.",
    "Show me areas where the average median spend per customer from 2021 to 2024 was over $300.",
    "Find zones that experienced an average yearly spend decline of less than -5% between 2019 and 2021 — might be a good place to introduce a discount brand.",
    "I want to analyze parts where the average year-over-year spend change from 2020 to 2022 stayed consistently over 20%.",
    "Identify areas where the average median spend per transaction from 2019 to 2021 was more than $70 — targeting upscale shoppers.",
    "Looking at rural zones where the average spend per customer from 2020 to 2023 was at least $225.",
            
]

medium_3 = [
    "I want to open a brunch spot where median spend per customer was ≤ $22 and yearly transactions ≥ 80,000 in 2023.",
    "Where can I put a BBQ joint? Need areas with 90,000+ yearly customers and ≥ 10% annual spending growth in 2021.",
    "Scouting locations for a vegan cafe - want zones with median spend per transaction ≤ $18 and 5%+ year-over-year spending growth in 2024.",
    "Need to open a convenience store where total transactions ≥ 300,000 and customer spend grew ≥ 7% year-over-year in 2023.",
    "Looking for a location for a food truck park - show me areas with ≥ 15,000 monthly customers and ≤ $9 median spend in 2022.",
    "Planning a bakery - find areas with 10%+ year-over-year spending growth and ≥ 20,000 transactions in 2021.",
            
]

medium_4 = [
    "I’m looking to develop a business plaza—need a zone with at least 5 parking lots and one lot over 9,000 square meters.",
    "Planning to open a medical center, and I want an area that either has at least 6 parking lots or one lot bigger than 12,000 square meters.",
    "I want to build a sports training facility where the zone includes at least 4 parking lots and one of them must exceed 10,000 square meters.",
    "Looking to set up a distribution center—open to zones that either have 3 or more parking lots or one very large lot of at least 15,000 square meters.",
    "Thinking about launching a garden center, but I need at least 2 parking lots and one of them must be larger than 5,000 square meters.",
    "Considering a big-box retail location—must have at least 7 parking lots or one that’s bigger than 13,000 square meters.",
            
]

medium_5 = [
    "I'm looking to open a family entertainment center, and I need a parking lot with at least 100 parking spaces that is also larger than 2,000 square meters.",
    "Thinking about launching a big-box retail store—open to any zone that either has a parking lot with 300+ parking spaces or at least one lot over 5,000 square meters.",
    "I want to develop a new sports complex, but only if the site has a parking lot with at least 250 parking spots and a single parking lot bigger than 10,000 square meters.",
    "I'm planning to open a convention center, and I need a parking lot with at least 400 parking spaces that is also larger than 12,000 square meters.",
    "Looking for a site to build a new hospital—must have a parking lot with more than 200 spaces and an area greater than 8,000 square meters.",
    "I want to develop a luxury outlet mall, but only in zones where there’s a parking lot with at least 500 parking spots and over 15,000 square meters in size.",
            
]

medium_6 = [
    "I’m looking to build a lifestyle center, and I need a zone with at least 4 parking lots, one of which has at least 300 parking spaces.",
    "Planning a sports arena—only considering zones with at least 6 parking lots and one with over 500 parking spaces.",
    "Thinking of opening a regional conference center—must have at least 5 parking lots, with one offering no fewer than 400 spaces.",
    "I want to open a premium outlet village, and I need a zone that includes at least 7 parking lots, with at least one lot providing 600 parking spots.",
    "Looking to develop a modern civic center—I’m targeting areas with a minimum of 3 parking lots and at least one that holds 250 cars.",
    "Scouting locations for a university extension campus—must include 4 or more parking lots, and one must have at least 350 parking spaces.",
            
]

medium_7 = [
    "I want to open a clothing store with top category {Other Miscellaneous Store Retailers} and sub category {Art Dealers}. Show me zones with fewer than 4 competitors in the same sub-category and a combined population of at least 15,000 across the zone and its 3 closest neighbors.",
    "Thinking about launching a boutique. Top category: {Personal Care Services}, sub-category: {Beauty Salons}. I'm looking for zones with less than 5 competitors in the same category or where the zone plus 2 nearest neighbors have over 20,000 people.",
    "I want to open a Korean BBQ restaurant where the total population of my zone and 2 closest zones is at least 18,000, and the number of existing competitors in the same category is fewer than 3.",
    "I want to open a clothing store with top category {Offices of Real Estate Agents and Brokers} and sub category {Offices of Real Estate Agents and Brokers}. Show me zones with fewer than 3 competitors in the same sub-category and no fewer than 12,000 people across the zone and its 2 closest neighbors.",
    "I'm planning to launch a Mediterranean restaurant. I need a zone where there are fewer than 5 similar restaurants in the same category or where the combined population of the zone and 3 nearby zones is at least 25,000.",
    "Looking to open a clothing store with top category {Advertising, Public Relations, and Related Services} and sub category {Advertising Agencies}. Show me zones that either have fewer than 2 competitors in the same category or a combined population at least 8,000 with 2 adjacent zones.",
            
]

medium_8 = [
    "I want to open a ramen restaurant. Show me zones with at least 5 POIs in the top category {Restaurants and Other Eating Places} and at least 3 subway entrances nearby.",
    "Planning a gastropub—I'd like zones with at least 4 POIs in the top category {Drinking Places (Alcoholic Beverages)} and a minimum of 6 nearby bus stops.",
    "I'm scouting areas for a brunch café. I need zones that either have 3+ POIs in the sub category {Full-Service Restaurants} or at least 5 subway entrances within walking distance.",
    "Looking to launch a wine bar. Find me zones that include at least 4 POIs in the top category {Beer, Wine, and Liquor Stores} and also have 4 or more taxi stands nearby.",
    "I'm interested in opening a coffee shop. Show me zones with at least 6 POIs in the top category {Snack and Nonalcoholic Beverage Bars} or a minimum of 5 bus stops in the area.",
    "I want to open a sushi restaurant. Show me zones with at least 5 POIs in the top category {Full-Service Restaurants} and a minimum of 4 subway entrances nearby.",
    
]

medium_9 = [
    "I'm looking to open a tea shop where there are at least 5 bus stops nearby and the closest bus stop is less than 180 meters from the zone centroid.",
    "Planning a boutique hotel—I'd like zones with at least 4 subway entrances or a subway entrance located within 150 meters of the zone centroid.",
    "I want to open a bookstore where the zone has at least 6 taxi stands and the nearest one is no more than 200 meters from the zone centroid.",
    "I'm looking to open a food truck hub where there are at least 6 bus stops nearby and the nearest one is within 200 meters of the zone centroid.",
    "Planning to launch a small hotel—I'm targeting zones that either have at least 3 subway entrances or one located within 120 meters of the zone centroid.",
    "I want to open a coworking space where the area includes at least 4 taxi stands and the closest one is no more than 250 meters from the zone centroid.",
            
]

medium_10 = [
    "I'm planning to open a community café where at least 65% of POIs are within 400 meters of a bus stop and the zone has at least 5 bus stops nearby.",
    "I want to launcht a coworking lounge—preferably in zones with 60% or more POIs within 300 meters of a subway entrance or areas with at least 4 subway entrances nearby.",
    "Looking to open a bookstore café. Show me zones where at least 70% of POIs are within 500 meters of a station and there are 3 or more stations nearby.",
    "I'm scouting locations for a craft brewery—interested in zones with 75% of are POIs within 350 meters of a bus stop or zones that include at least 6 bus stops.",
    "I want to set up a vegan restaurant in a zone where 80% of POIs are within 400 meters of a subway entrance and the area includes no fewer than 4 subway entrances.",
    "I'm looking to open a community theater where at least 70% of POIs are within 400 meters of a subway entrance and there are at least 3 subway entrances nearby.",
    
]

medium_11 = [
    "I want to open a clothing store with top category {Personal Care Services} and sub category {Beauty Salons}. Show me zones with fewer than 3 competitors in the same category and at least 2 parking lots.",
    "Thinking about launching a boutique. Top category: {Other Amusement and Recreation Industries}, sub-category: {Fitness and Recreational Sports Centers}. I need a zone with fewer than 4 competitors or at least 3 parking lots in the area.",
    "Looking to open a clothing store with top category {Offices of Other Health Practitioners} and sub category {Offices of All Other Miscellaneous Health Practitioners}. Find me a zone with fewer than 2 competitors and at least 2 parking lots.", 
    "I want to launch a fashion outlet. Top category: {Other Miscellaneous Store Retailers}, sub-category: {Art Dealers}. Show me zones with fewer than 5 competitors or zones with 4 or more parking lots.",
    "Planning to set up a sustainable clothing store. I want zones with fewer than 3 competitors in the same sub-category and a minimum of 3 parking lots nearby.",
    "I'm interested in opening a thrift store with top category {Educational Support Services} and sub category {Educational Support Services}. Show me zones with fewer than 4 competitors or areas that offer at least 2 parking lots.",
                
]

medium_12 = [
    "I want to open a tapas restaurant where my zone and 2 closest neighbors have a total population of at least 15,000 and the area includes at least 5 POIs in the top category {Full-Service Restaurants}.",
    "Thinking of launching a speakeasy bar—I'm looking for zones that either have 3+ POIs in the sub category {Drinking Places (Alcoholic Beverages)} or where the population of the zone plus 2 nearby zones exceeds 12,000.",
    "Looking to open a ramen shop. Show me areas with a combined population of at least 20,000 from my zone and 3 neighbors and at least 6 POIs in the category {Restaurants and Other Eating Places}.",
    "I'm planning to open a vegan café—targeting zones that have 4 or more POIs in the sub category {Snack and Nonalcoholic Beverage Bars} or where the total population across the zone and 2 nearby ones is at least 10,000.",
    "I want to start a sushi restaurant where my zone plus 3 closest neighbors have a combined population of 18,000 and at least 5 POIs in the sub category {Full-Service Restaurants}.",
    "Thinking of opening a dessert bar—I'm looking for zones with at least 3 sub category {Snack and Nonalcoholic Beverage Bars} or 2 nearest neighbors plus my zone having at least 9,000 residents total.",
            
]

medium_13 = [
    "I'm scouting locations for a high-traffic salon and retail space. Show me zones with at least 10 POIs in the sub-categories {Beauty Salons} or {Women's Clothing Stores}, and where the nearest bus stop is under 200 meters from the zone centroid.",
    "Looking to build a late-night food plaza—find me zones with 12+ POIs in sub category {Drinking Places (Alcoholic Beverages)} and {Snack and Nonalcoholic Beverage Bars}, and where the nearest subway entrance is less than 150 meters away.",
    "I want to open a hybrid tattoo parlor and juice bar. I need zones with at least 8 POIs in the categories {Beauty Salons} and {Snack and Nonalcoholic Beverage Bars}, and a taxi stop within 180 meters from the zone centroid.",
    "Thinking of launching a wellness and café combo—show me zones with 10+ POIs in {Beauty Salons} or {Snack and Nonalcoholic Beverage Bars}, and where the closest bus stop is less than 200 meters away.",
    "I want to start a nightlife venue—looking for areas with at least 14 POIs in {Drinking Places (Alcoholic Beverages)} and {Snack and Nonalcoholic Beverage Bars}, and the nearest station under 250 meters from the centroid.",
    "Opening a boutique gym and smoothie shop—show me zones with 9 or more POIs in {Fitness and Recreational Sports Centers} and {Snack and Nonalcoholic Beverage Bars}, and the closest subway entrance within 200 meters of the centroid.",
            
]

medium_14 = [
    "Find zones where at least 35% of total raw total spend in 2020 comes from top category {Restaurants and Other Eating Places}.",
    "Show me zones where 40% or more of raw num transactions in 2021 come from top category {Gasoline Stations}.",
    "I’m looking for areas where at least 30% of total raw num customers in 2022 are from top category {Personal Care Services}.",
    "Identify zones where 50% or more of the total raw total spend in 2023 is attributed to top category {Offices of Physicians}.",
    "Filter zones where at least 45% of total raw num transactions in 2024 come from top category {Beer, Wine, and Liquor Stores}.",
    "Find me zones where at least 60% of the raw num customers in 2022 are tied to top category {Offices of Other Health Practitioners}.",
            
]

medium_15 = [
    "Show me places where top category {Management of Companies and Enterprises} doesn't exceed 30% and there are at least 13 POIs.",
    "Show me places where top category {Software Publishers} doesn't exceed 30% and there are at least 5 POIs.",
    "I need zones that have at least 22 POIs and less than 20% in top category {Drinking Places (Alcoholic Beverages)}.",
    "Find zones with at least 8 POIs and no more than 20% from top category {Management of Companies and Enterprises}.",
    "I'm targeting zones where at least 24 POIs exist and fewer than 35% are from {Scheduled Passenger Air Transportation}.",
    "Help me locate zones with over 30 POIs, but top category {Medical and Diagnostic Laboratories} should be below 40%.",
            
]

medium_16 = [
    "Looking to build a spa — find me areas where sub category {Advertising Agencies} dominates at least 40% of 2024 spend or has 2+ parking spots.",
    "Want to open a coffee lounge in a spot where sub category {Full-Service Restaurants} is strong — 50%+ of spend in 2022 — or somewhere with 4 parking spaces.",
    "I'm opening a family clinic and want zones where at least 60% of 2022 spending comes from sub category {Offices of Dentists} AND there's space for at least 2 parking lots.",
    'Looking to build a spa — find me areas where sub category {Advertising Agencies} dominates at least 60% of 2024 spend or has 3+ parking spots.',
    "I'm exploring locations for a tutoring center — it should either have 30%+ of 2019's spend from top category {Legal Services} or decent parking: at least 2 lots.",
    "I'm opening a family clinic and want zones where at least 30% of 2020 spending comes from sub category {Offices of Dentists} AND there's space for at least 2 parking lots.",
    
]
