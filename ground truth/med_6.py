import sys
import os
import pandas as pd
import operator
import numpy as np

code_dir = "/Users/mihokoda/Desktop/CityLLM/code"
if code_dir not in sys.path:
    sys.path.insert(0, code_dir)

from site_selection.loader import get_poi_spend_dataset, get_parking_dataset
from site_selection.zone import create_zone, assign_parking_zones
from site_selection.analysis import get_largest_parking_capacity, get_num_parking
from site_selection.filter import filter_df_based_on_zone

def medium_6(min_capacity, min_lots, logic="AND"):
    poi_df = get_poi_spend_dataset()
    zone_df = create_zone(poi_df)
    parking_df = get_parking_dataset()
    parking_df = assign_parking_zones(parking_df, zone_df)

    survived_zones = []

    for zone_id in zone_df['zone_id']:
        filtered_parking_df = filter_df_based_on_zone(parking_df, zone_id)
        capacity = get_largest_parking_capacity(filtered_parking_df)
        lot_count = get_num_parking(filtered_parking_df)

        if logic == "AND":
            if capacity >= min_capacity and lot_count >= min_lots:
                survived_zones.append(zone_id)
        elif logic == "OR":
            if capacity >= min_capacity or lot_count >= min_lots:
                survived_zones.append(zone_id)

    return zone_df[zone_df['zone_id'].isin(survived_zones)]

medium_6_test_cases = [
    (300, 4, "AND"),
    (500, 6, "AND"),
    (400, 5, "AND"),
    (600, 7, "AND"),
    (250, 3, "AND"),
    (350, 4, "AND"),
]

all_valid_zones = set()
for i, (min_capacity, min_lots, logic) in enumerate(medium_6_test_cases):
    obj_path = f'/Users/mihokoda/Desktop/CityLLM/test_results/med/6/tc_med_6_{i}/objective.csv'
    obj = pd.read_csv(obj_path)
    obj_zones = set(obj['zone_id'].unique())    
    
    valid_zones_df = medium_6(min_capacity, min_lots, logic)
    result_zones = set(valid_zones_df['zone_id'].unique())
    
    all_valid_zones.update(result_zones)
    print(f"Test case {i} (params: {min_capacity}, {min_lots}, {logic}):")
    print(f"  Found zones: {len(result_zones)}")
    print(f"  Objective zones: {len(obj_zones)}")
    print(f"  Match: {len(result_zones) == len(obj_zones)}")