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

def simple_17(min_poi_count):
    poi_df = get_poi_spend_dataset()
    zone_df = create_zone(poi_df)

    survived_zones = []

    for zone_id in zone_df['zone_id']:
        poi_count = len(filter_df_based_on_zone(poi_df, zone_id))
        if poi_count >= min_poi_count:
            survived_zones.append(zone_id)

    return zone_df[zone_df['zone_id'].isin(survived_zones)]

simple_17_test_cases = [
    (25),
    (20),
    (15),
    (60),
    (20),
    (5),
]

all_valid_zones = set()
for i, min_poi_count in enumerate(simple_17_test_cases):
    obj_path = f'/Users/mihokoda/Desktop/CityLLM/test_results/sim/17/tc_sim_17_{i}/objective.csv'
    obj = pd.read_csv(obj_path)
    obj_zones = set(obj['zone_id'].unique())    
    
    valid_zones_df = simple_17(min_poi_count)
    
    result_zones = set(valid_zones_df['zone_id'].unique())
    
    all_valid_zones.update(result_zones)
    print(f"Test case {i} (params: {min_poi_count}):")
    print(f"  Found zones: {len(result_zones)}")
    print(f"  Objective zones: {len(obj_zones)}")