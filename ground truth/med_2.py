import sys
import os
import pandas as pd
import operator
import numpy as np

code_dir = "/Users/mihokoda/Desktop/CityLLM/code"
if code_dir not in sys.path:
    sys.path.insert(0, code_dir)

from site_selection.loader import get_poi_spend_dataset
from site_selection.zone import create_zone 
from site_selection.analysis import get_spendparam_years
from site_selection.filter import filter_df_based_on_zone
from comparison.compare import compare

def medium_2(spend_param, year_start, year_end, threshold, logic):
    poi_df = get_poi_spend_dataset()
    zone_df = create_zone(poi_df)

    survived_zones = []
    logic_fn = compare(logic)

    for zone_id in zone_df['zone_id']:
        values = []
        for year in range(int(year_start), int(year_end) + 1):
            filtered_poi_df = filter_df_based_on_zone(poi_df, zone_id)
            yearly_value = get_spendparam_years(filtered_poi_df, spend_param, str(year))

            if yearly_value is not None and yearly_value > 0:
                values.append(yearly_value)

        if not values:
            continue  

        average_value = sum(values) / len(values)

        if logic_fn(average_value, threshold):
            survived_zones.append(zone_id)

    return zone_df[zone_df['zone_id'].isin(survived_zones)]


medium_2_test_cases = [
    ("MEDIAN_SPEND_PER_TRANSACTION", 2020, 2023, 45, ">"),
    ('MEDIAN_SPEND_PER_CUSTOMER', 2021, 2024, 300, ">"),    
    ('SPEND_PCT_CHANGE_VS_PREV_YEAR', 2019, 2021, -5, "<"), 
    ('SPEND_PCT_CHANGE_VS_PREV_YEAR', 2020, 2022, 20, ">"),     
    ('MEDIAN_SPEND_PER_TRANSACTION', 2019, 2021, 70, ">"),      
    ('MEDIAN_SPEND_PER_CUSTOMER', 2020, 2023, 225, ">"),        
]

all_valid_zones = set()
for i, (spend_param, start_year, end_year, threshold, logic) in enumerate(medium_2_test_cases):
    obj_path = f'/Users/mihokoda/Desktop/CityLLM/test_results/med/2/tc_med_2_{i}/objective.csv'
    obj = pd.read_csv(obj_path)
    obj_zones = set(obj['zone_id'].unique())    
    
    valid_zones_df = medium_2(spend_param, start_year, end_year, threshold, logic)
    result_zones = set(valid_zones_df['zone_id'].unique())
    
    all_valid_zones.update(result_zones)
    print(f"Test case {i} (params: {spend_param}, {start_year}-{end_year}, threshold: {threshold}):")
    print(f"  Found zones: {len(result_zones)}")
    print(f"  Objective zones: {len(obj_zones)}")
    print(f"  Match: {len(result_zones) == len(obj_zones)}")