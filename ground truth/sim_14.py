import sys
import os
import pandas as pd

# Add the absolute path to the code directory
code_dir = "/Users/mihokoda/Desktop/CityLLM/code"
if code_dir not in sys.path:
    sys.path.insert(0, code_dir)

# Import directly from site_selection
from site_selection.loader import get_poi_spend_dataset
from site_selection.zone import create_zone

def simple_14(percent_threshold, top_category=None, sub_category=None):
    assert top_category or sub_category, "At least one of top_category or sub_category must be specified."

    poi_df = get_poi_spend_dataset()
    zone_df = create_zone(poi_df)

    if top_category:
        match_mask = poi_df["TOP_CATEGORY"] == top_category
    else:
        match_mask = poi_df["SUB_CATEGORY"] == sub_category

    poi_df["is_match"] = match_mask

    counts = poi_df.groupby("zone_id").agg(
        total_pois=("PLACEKEY", "count"),
        matching_pois=("is_match", "sum")
    ).reset_index()

    counts["match_percent"] = (counts["matching_pois"] / counts["total_pois"]) * 100

    qualifying_zone_ids = counts[counts["match_percent"] >= percent_threshold]["zone_id"]

    return zone_df[zone_df["zone_id"].isin(qualifying_zone_ids)]

simple_14_test_cases = [
    (40, "Beauty Salons"),        
    (35, "Restaurants and Other Eating Places"),
    (50, "Snack and Nonalcoholic Beverage Bars"),           
    (60, "Offices of Physicians"),  
    (45, "Educational Support Services"),         
    (30, "Personal Care Services"),  
]

all_valid_zones = set()
for i, (percent_threshold, top_category) in enumerate(simple_14_test_cases):
    obj_path = f'/Users/mihokoda/Desktop/CityLLM/test_results/sim/14/tc_sim_14_{i}/objective.csv'
    obj = pd.read_csv(obj_path)
    obj_zones = set(obj['zone_id'].unique())    
    
    valid_zones_df = simple_14(percent_threshold, top_category)
    result_zones = set(valid_zones_df['zone_id'].unique())
    
    all_valid_zones.update(result_zones)
    print(f"Test case {i} (params: {percent_threshold}, {top_category}):")
    print(f"  Found zones: {len(result_zones)}")
    print(f"  Objective zones: {len(obj_zones)}")