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
from site_selection.analysis import get_largest_parking_capacity, get_largest_parking_lot_area
from site_selection.filter import filter_df_based_on_zone

def medium_5(capacity_val, capacity_op, area_val, area_op, logic="AND"): 
    poi_df = get_poi_spend_dataset()     
    zone_df = create_zone(poi_df)     
    parking_df = get_parking_dataset()     
    parking_df = assign_parking_zones(parking_df, zone_df)          
    # Define comparison operations     
    comparison_ops = {         
        ">=": lambda x, y: x >= y,         
        "<=": lambda x, y: x <= y,         
        ">": lambda x, y: x > y,         
        "<": lambda x, y: x < y     
        }         
    # Validate comparison operators     
    if capacity_op not in comparison_ops or area_op not in comparison_ops:         
        raise ValueError(f"Invalid comparison operator. Must be one of: {list(comparison_ops.keys())}")          
  
    capacity_compare = comparison_ops[capacity_op]     
    area_compare = comparison_ops[area_op]    
          
    survived_zones = []          

    for zone_id in zone_df['zone_id']:    

        filtered_parking_df = filter_df_based_on_zone(parking_df, zone_id) 

        max_capacity = get_largest_parking_capacity(filtered_parking_df)         
        max_area = get_largest_parking_lot_area(filtered_parking_df)

        satisfies_capacity = capacity_compare(max_capacity, capacity_val)         
        satisfies_area = area_compare(max_area, area_val)     

        if (logic == "AND" and satisfies_capacity and satisfies_area) or (logic == "OR" and (satisfies_capacity or satisfies_area)):             
            survived_zones.append(zone_id)          
    return zone_df[zone_df['zone_id'].ijnhsin(survived_zones)]

medium_5_test_cases = [
    (100, ">=", 2000, ">", "AND"),
    (300, ">=", 5000, ">", "OR"),
    (250, ">=", 10000, ">", "AND"),
    (400, ">=", 12000, ">", "OR"),
    (200, ">=", 8000, ">", "AND"),
    (500, ">=", 15000, ">", "OR"),
]

all_valid_zones = set()
for i, (capacity_val, capacity_op, area_val, area_op, logic) in enumerate(medium_5_test_cases):
    obj_path = f'/Users/mihokoda/Desktop/CityLLM/test_results/med/5/tc_med_5_{i}/objective.csv'
    obj = pd.read_csv(obj_path)
    obj_zones = set(obj['zone_id'].unique())    
    
    valid_zones_df = medium_5(capacity_val, capacity_op, area_val, area_op, logic)
    result_zones = set(valid_zones_df['zone_id'].unique())
    
    all_valid_zones.update(result_zones)
    print(f"Test case {i} (params: {capacity_val}, {capacity_op}, {area_val}, {area_op}, {logic}):")
    print(f"  Found zones: {len(result_zones)}")
    print(f"  Objective zones: {len(obj_zones)}")
    print(f"  Match: {len(result_zones) == len(obj_zones)}")