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

def simple_16(percent_threshold, spend_param, year, top_category=None, sub_category=None):
    poi_df = get_poi_spend_dataset()
    zone_df = create_zone(poi_df)

    survived_zones = []

    for zone_id in zone_df['zone_id']:
        zone_pois = filter_df_based_on_zone(poi_df, zone_id)
        if len(zone_pois) == 0:
            continue

        total_spend = get_spendparam_years(zone_pois, spend_param, year)
        if total_spend == 0:
            continue

        # Filter by category
        if top_category:
            category_pois = filter_pois_by_top_category(zone_pois, top_category)
        elif sub_category:
            category_pois = filter_pois_by_sub_category(zone_pois, sub_category)
        else:
            continue

        category_spend = get_spendparam_years(category_pois, spend_param, year)
        share = (category_spend / total_spend) * 100

        if share >= percent_threshold:
            survived_zones.append(zone_id)

    return zone_df[zone_df['zone_id'].isin(survived_zones)]


simple_16_test_cases = [
    (50, "RAW_TOTAL_SPEND", 2023, "Offices of Physicians", None),
    (40, "RAW_NUM_TRANSACTIONS", 2022, None, "Couriers and Express Delivery Services"),
    (60, "RAW_TOTAL_SPEND", 2023, "Legal Services", None),
    (35, "RAW_TOTAL_SPEND", 2022, None, "Used Car Dealers"),
    (45, "RAW_NUM_CUSTOMERS", 2023, None, "Offices of Lawyers"),
    (55, "RAW_TOTAL_SPEND", 2022, "Other Amusement and Recreation Industries", None),
]

all_valid_zones = set()
for i, (percent_threshold, spend_param, year, top_category, sub_category) in enumerate(simple_16_test_cases):
    obj_path = f'/Users/mihokoda/Desktop/CityLLM/test_results/sim/16/tc_sim_16_{i}/objective.csv'
    obj = pd.read_csv(obj_path)
    obj_zones = set(obj['zone_id'].unique())    
    
    valid_zones_df = simple_16(percent_threshold, spend_param, year, top_category, sub_category)
    
    result_zones = set(valid_zones_df['zone_id'].unique())
    
    all_valid_zones.update(result_zones)
    print(f"Test case {i} (params: {percent_threshold}, {spend_param}, {year}, {top_category}, {sub_category}):")
    print(f"  Found zones: {len(result_zones)}")
    print(f"  Objective zones: {len(obj_zones)}")