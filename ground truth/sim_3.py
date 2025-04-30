import sys
import os
import pandas as pd

code_dir = "/Users/mihokoda/Desktop/CityLLM/code"
if code_dir not in sys.path:
    sys.path.insert(0, code_dir)

from site_selection.loader import get_poi_spend_dataset, get_parking_dataset
from site_selection.zone import create_zone, assign_parking_zones
from site_selection.analysis import get_num_parking
from site_selection.filter import filter_df_based_on_zone


def simple_3(num):
    poi_spend_df = get_poi_spend_dataset()
    parking_df = get_parking_dataset()
    zone_df = create_zone(poi_spend_df)
    parking_df_with_zones = assign_parking_zones(parking_df, zone_df)

    valid_zones = []
    for zone_id in zone_df['zone_id'].unique():
        filtered_parking_df = filter_df_based_on_zone(parking_df_with_zones, zone_id)
        total_space = get_num_parking(filtered_parking_df)
        if total_space >= num:
            valid_zones.append(zone_id)
    
    return zone_df[zone_df['zone_id'].isin(valid_zones)]


simple_3_test_cases = [
   3, 4, 1, 4, 3, 2
]
all_valid_zones = set()

for i, num in enumerate(simple_3_test_cases):
    obj_path = f'/Users/mihokoda/Desktop/CityLLM/test_results/sim/3/tc_sim_3_{i}/objective.csv'
    obj = pd.read_csv(obj_path)
    obj_zones = set(obj['zone_id'].unique())
    
    valid_zones_df = simple_3(num)
    result_zones = set(valid_zones_df['zone_id'].unique())
    
    all_valid_zones.update(result_zones)
    print(f"Test case {i} (threshold {num}):")
    print(f"  Found zones: {len(result_zones)}")
    print(f"  Objective zones: {len(obj_zones)}")
