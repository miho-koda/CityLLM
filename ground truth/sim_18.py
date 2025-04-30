import sys
import os
import pandas as pd

code_dir = "/Users/mihokoda/Desktop/CityLLM/code"
if code_dir not in sys.path:
    sys.path.insert(0, code_dir)

from site_selection.loader import get_poi_spend_dataset
from site_selection.zone import create_zone
from site_selection.filter import filter_df_based_on_zone, filter_pois_by_top_category, filter_pois_by_sub_category
from site_selection.analysis import get_spendparam_years

def simple_18(max_percent, top_category=None, sub_category=None):
    poi_df = get_poi_spend_dataset()
    zone_df = create_zone(poi_df)

    survived_zones = []

    # Determine filter method and column name
    filter_fn = None
    category_value = None
    category_col = None

    if top_category:
        filter_fn = filter_pois_by_top_category
        category_value = top_category
        category_col = "TOP_CATEGORY"
    elif sub_category:
        filter_fn = filter_pois_by_sub_category
        category_value = sub_category
        category_col = "SUB_CATEGORY"
    else:
        raise ValueError("Either top_category or sub_category must be provided.")

    for zone_id in zone_df["zone_id"]:

        zone_pois = filter_df_based_on_zone(poi_df, zone_id)
        if len(zone_pois) == 0:
            continue

        # Filtered POIs that match the specified category
        filtered = filter_fn(zone_pois, category_value)

        share = (len(filtered) / len(zone_pois)) * 100
        if share <= max_percent:
            survived_zones.append(zone_id)

    return zone_df[zone_df["zone_id"].isin(survived_zones)]

simple_18_test_cases = [
    (15, "Offices of Physicians", None),
    (20, None, "Offices of Lawyers"),
    (9, "Offices of Physicians", None),
    (25, None, "Fitness and Recreational Sports Centers"),
    (20, "Offices of Other Health Practitioners", None),
    (60, None, "Investment Advice"),
]

all_valid_zones = set()
for i, (max_percent, top_category, sub_category) in enumerate(simple_18_test_cases):
    obj_path = f'/Users/mihokoda/Desktop/CityLLM/test_results/sim/18/tc_sim_18_{i}/objective.csv'
    obj = pd.read_csv(obj_path)
    obj_zones = set(obj['zone_id'].unique())    
    
    valid_zones_df = simple_18(max_percent, top_category, sub_category)
    
    result_zones = set(valid_zones_df['zone_id'].unique())
    
    all_valid_zones.update(result_zones)
    print(f"Test case {i} (params: {max_percent}, {top_category}, {sub_category}):")
    print(f"  Found zones: {len(result_zones)}")
    print(f"  Objective zones: {len(obj_zones)}")