import sys
import os
import pandas as pd

code_dir = "/Users/mihokoda/Desktop/CityLLM/code"
if code_dir not in sys.path:
    sys.path.insert(0, code_dir)

from site_selection.loader import get_poi_spend_dataset
from site_selection.zone import create_zone

def simple_15(top_category=None, sub_category=None):
    assert top_category or sub_category, "You must provide either a top_category or a sub_category."

    poi_df = get_poi_spend_dataset()
    zone_df = create_zone(poi_df)

    category_column = "TOP_CATEGORY" if top_category else "SUB_CATEGORY"
    target_value = top_category if top_category else sub_category

    grouped = poi_df.groupby(["zone_id", category_column]).size().reset_index(name="count")

    dominant = grouped.loc[grouped.groupby("zone_id")["count"].idxmax()]

    matching_zone_ids = dominant[dominant[category_column] == target_value]["zone_id"]

    return zone_df[zone_df["zone_id"].isin(matching_zone_ids)]

simple_15_test_cases = [
    ("Beauty Salons"),        
    ("Full-Service Restaurants"),
    ("Snack and Nonalcoholic Beverage Bars"),           
    ("Offices of Dentists"),  
    ("Gasoline Stations with Convenience Stores"),         
    ("Art Dealers"),  
]

all_valid_zones = set()
for i, (category) in enumerate(simple_15_test_cases):
    obj_path = f'/Users/mihokoda/Desktop/CityLLM/test_results/sim/15/tc_sim_15_{i}/objective.csv'
    obj = pd.read_csv(obj_path)
    obj_zones = set(obj['zone_id'].unique())    
    
    valid_zones_df = simple_15(sub_category=category)
    result_zones = set(valid_zones_df['zone_id'].unique())
    
    all_valid_zones.update(result_zones)
    print(f"Test case {i} (params: {category}):")
    print(f"  Found zones: {len(result_zones)}")
    print(f"  Objective zones: {len(obj_zones)}")