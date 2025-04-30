import sys
import os
import pandas as pd

code_dir = "/Users/mihokoda/Desktop/CityLLM/code"
if code_dir not in sys.path:
    sys.path.insert(0, code_dir)

from site_selection.loader import get_poi_spend_dataset
from site_selection.zone import create_zone 
from site_selection.analysis import get_spendparam_years
from site_selection.filter import filter_df_based_on_zone

def medium_1(spend_param, start_year, end_year, threshold):
    poi_spend_df = get_poi_spend_dataset()

    zone_df = create_zone(poi_spend_df)

    filtered_zone_ids = []
    
    for zone_id in zone_df['zone_id'].unique():
        zone_pois = filter_df_based_on_zone(poi_spend_df, zone_id)
        
        total_value = 0
        for year in range(start_year, end_year + 1):
            total_value += get_spendparam_years(zone_pois, spend_param, year)
        
        if total_value > threshold:
            filtered_zone_ids.append(zone_id)

    filtered_zone_df = zone_df[zone_df['zone_id'].isin(filtered_zone_ids)]

    return filtered_zone_df[['zone_id', 'geometry', 'center_lat', 'center_lng', 'num_pois']]

medium_1_test_cases = [
    ('RAW_TOTAL_SPEND', 2019, 2021, 22000000),       
    ('RAW_NUM_TRANSACTIONS', 2020, 2022, 400000),    
    ('RAW_NUM_CUSTOMERS', 2021, 2023, 150000),      
    ('RAW_TOTAL_SPEND', 2020, 2022, 18000000),      
    ('RAW_NUM_TRANSACTIONS', 2019, 2021, 500000),    
    ('RAW_NUM_CUSTOMERS', 2021, 2024, 300000),      
]

all_valid_zones = set()
for i, (spend_param, start_year, end_year, threshold) in enumerate(medium_1_test_cases):
    obj_path = f'/Users/mihokoda/Desktop/CityLLM/test_results/med/1/tc_med_1_{i}/objective.csv'
    obj = pd.read_csv(obj_path)
    obj_zones = set(obj['zone_id'].unique())    
    
    valid_zones_df = medium_1(spend_param, start_year, end_year, threshold)
    result_zones = set(valid_zones_df['zone_id'].unique())
    
    all_valid_zones.update(result_zones)
    print(f"Test case {i} (params: {spend_param}, {start_year}-{end_year}, threshold: {threshold}):")
    print(f"  Found zones: {len(result_zones)}")
    print(f"  Objective zones: {len(obj_zones)}")
    print(f"  Match: {len(result_zones) == len(obj_zones)}")