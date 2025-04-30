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
from site_selection.analysis import get_num_parking, get_largest_parking_lot_area
from site_selection.filter import filter_df_based_on_zone

def medium_4(min_lots, min_lot_area, logic):

    poi_df = get_poi_spend_dataset()
    zone_df = create_zone(poi_df)
    parking_df = get_parking_dataset()
    parking_df = assign_parking_zones(parking_df, zone_df)

    survived_zones = []

    for zone_id in zone_df["zone_id"]:
        filtered_parking_df = filter_df_based_on_zone(parking_df, zone_id)
        count = get_num_parking(filtered_parking_df)
        largest = get_largest_parking_lot_area(filtered_parking_df)

        passes_count = count >= min_lots
        passes_size = largest >= min_lot_area

        if (logic == "AND" and passes_count and passes_size) or \
           (logic == "OR" and (passes_count or passes_size)):
            survived_zones.append(zone_id)

    return zone_df[zone_df["zone_id"].isin(survived_zones)]

medium_4_test_cases = [
    (5, 9000, "AND"),
    (6, 12000, "OR"),
    (4, 10000, "AND"),
    (3, 15000, "OR"),
    (2, 5000, "AND"),
    (7, 13000, "OR"),
]

all_valid_zones = set()
for i, (min_lots, min_lot_area, logic) in enumerate(medium_4_test_cases):
    obj_path = f'/Users/mihokoda/Desktop/CityLLM/test_results/med/4/tc_med_4_{i}/objective.csv'
    obj = pd.read_csv(obj_path)
    obj_zones = set(obj['zone_id'].unique())    
    
    valid_zones_df = medium_4(min_lots, min_lot_area, logic)
    result_zones = set(valid_zones_df['zone_id'].unique())
    
    all_valid_zones.update(result_zones)
    print(f"Test case {i} (params: {min_lots}, {min_lot_area}, {logic}):")
    print(f"  Found zones: {len(result_zones)}")
    print(f"  Objective zones: {len(obj_zones)}")
    print(f"  Match: {len(result_zones) == len(obj_zones)}")