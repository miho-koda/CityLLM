import sys
import os
import pandas as pd

code_dir = "/Users/mihokoda/Desktop/CityLLM/code"
if code_dir not in sys.path:
    sys.path.insert(0, code_dir)

from site_selection.loader import get_poi_spend_dataset
from site_selection.zone import create_zone
from site_selection.filter import filter_df_based_on_zone, filter_pois_by_top_category, filter_pois_by_sub_category

def simple_7(num, top_category=None, sub_category=None):
    poi_df = get_poi_spend_dataset()
    zone_df = create_zone(poi_df)

    survived_zones = []
    zone_ids = zone_df['zone_id'].unique() if 'zone_id' in zone_df.columns else zone_df.index

    for zone_id in zone_ids:
        filtered_poi_df = filter_df_based_on_zone(poi_df, zone_id)
        if top_category:
            filtered_poi_df = filter_pois_by_top_category(filtered_poi_df, top_category)
        if sub_category:
            filtered_poi_df = filter_pois_by_sub_category(filtered_poi_df, sub_category)

        if len(filtered_poi_df) >= num:
            survived_zones.append(zone_id)

    filtered_zone_df = zone_df[zone_df['zone_id'].isin(survived_zones)]
    return filtered_zone_df


simple_7_test_cases = [
    (4, "Drinking Places (Alcoholic Beverages)", None),
    (3, "Beer, Wine, and Liquor Stores", None),
    (5, "Drinking Places (Alcoholic Beverages)", None),
    (4, "Restaurants and Other Eating Places", None),
    (3, "Drinking Places (Alcoholic Beverages)", None),
    (3, "Home Furnishings Stores", None),
]

all_valid_zones = set()


for i, (num, top_category, sub_category) in enumerate(simple_7_test_cases):
    obj_path = f'/Users/mihokoda/Desktop/CityLLM/test_results/sim/7/tc_sim_7_{i}/objective.csv'
    obj = pd.read_csv(obj_path)
    obj_zones = set(obj['zone_id'].unique())
    
    valid_zones_df = simple_7(num, top_category, sub_category)
    result_zones = set(valid_zones_df['zone_id'].unique())
    
    all_valid_zones.update(result_zones)
    print(f"Test case {i} (params: {num}, {top_category}, {sub_category}):")
    print(f"  Found zones: {len(result_zones)}")
    print(f"  Objective zones: {len(obj_zones)}")
    print(f"  Match: {result_zones == obj_zones}")
