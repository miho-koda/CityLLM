
import sys
import os
import pandas as pd

code_dir = "/Users/mihokoda/Desktop/CityLLM/code"
if code_dir not in sys.path:
    sys.path.insert(0, code_dir)

from site_selection.loader import get_poi_spend_dataset
from site_selection.zone import create_zone
from site_selection.filter import get_neighbor_zones, get_population

def simple_6(num, num_2):
    poi_df = get_poi_spend_dataset()
    zone_df = create_zone(poi_df)

    survived_zones = []
    for zone_id in zone_df["zone_id"]:  
        neighbor_zone_lis = get_neighbor_zones(zone_df, zone_id, num)
        total_population = get_population(zone_id, zone_df)
        for neighbor_zone_id in neighbor_zone_lis:
            total_population += get_population(neighbor_zone_id, zone_df)
        
        if total_population >= num_2:
            survived_zones.append(zone_id)

    filtered_zone_df = zone_df[zone_df['zone_id'].isin(survived_zones)]
    return filtered_zone_df


simple_6_test_cases = [
    (2, 10000),  
    (2, 12000),  
    (3, 15000),  
    (1, 8000),  
    (2, 18000),  
    (4, 25000),  
]
all_valid_zones = set()

for i, (num, num_2) in enumerate(simple_6_test_cases):
    obj_path = f'/Users/mihokoda/Desktop/CityLLM/test_results/sim/6/tc_sim_6_{i}/objective.csv'
    obj = pd.read_csv(obj_path)
    obj_zones = set(obj['zone_id'].unique())
    
    valid_zones_df = simple_6(num, num_2)
    result_zones = set(valid_zones_df['zone_id'].unique())
    
    all_valid_zones.update(result_zones)
    print(f"Test case {i} (params: {num}, {num_2}):")
    print(f"  Found zones: {len(result_zones)}")
    print(f"  Objective zones: {len(obj_zones)}")
    print(f"  Match: {result_zones == obj_zones}")
