import sys
import os
import pandas as pd

code_dir = "/Users/mihokoda/Desktop/CityLLM/code"
if code_dir not in sys.path:
    sys.path.insert(0, code_dir)

from site_selection.loader import get_poi_spend_dataset, get_parking_dataset
from site_selection.zone import create_zone, assign_parking_zones
from site_selection.analysis import get_largest_parking_capacity
from site_selection.filter import filter_df_based_on_zone


def simple_1(num):
    poi_spend_df = get_poi_spend_dataset()
    parking_df = get_parking_dataset()
    zone_df = create_zone(poi_spend_df)
    parking_df_with_zones = assign_parking_zones(parking_df, zone_df)

    valid_zones = []
    for zone_id in zone_df['zone_id'].unique():
        filtered_parking_df = filter_df_based_on_zone(parking_df_with_zones, zone_id)
        total_spaces = get_largest_parking_capacity(filtered_parking_df)
        if total_spaces >= num:
            valid_zones.append(zone_id)
    
    return zone_df[zone_df['zone_id'].isin(valid_zones)]


simple_1_test_cases = [
    10, 50, 200, 30, 80, 150
]

# Initialize a set to store all valid zones across all test cases
all_valid_zones = set()

# Run through each test case
for i, num in enumerate(simple_1_test_cases):
    # Update the objective path for this specific test case
    obj_path = f'/Users/mihokoda/Desktop/CityLLM/test_results/sim/1/tc_sim_1_{i}/objective.csv'
    obj = pd.read_csv(obj_path)
    obj_zones = set(obj['zone_id'].unique())
    
    # Run simple_1 for this test case
    valid_zones_df = simple_1(num)
    result_zones = set(valid_zones_df['zone_id'].unique())
    
    # Add to our running total
    all_valid_zones.update(result_zones)
    print(f"Test case {i} (threshold {num}):")
    print(f"  Found zones: {len(result_zones)}")
    print(f"  Objective zones: {len(obj_zones)}")
