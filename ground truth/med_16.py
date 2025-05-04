import sys
import os
import pandas as pd
import operator
import numpy as np

code_dir = "/Users/mihokoda/Desktop/CityLLM/code"
if code_dir not in sys.path:
    sys.path.insert(0, code_dir)

from site_selection.loader import get_poi_spend_dataset, get_parking_dataset
from site_selection.zone import create_zone, get_zone_center, assign_parking_zones
from site_selection.analysis import get_spendparam_years, get_num_parking
from site_selection.filter import filter_df_based_on_zone, filter_pois_by_top_category, filter_pois_by_sub_category, get_transport_pois_in_zone
from site_selection.population import get_population
from comparison.compare import compare

def medium_16(percent_threshold, spend_param, year, min_parking_lots, logic='AND', top_category=None, sub_category=None):
    poi_df = get_poi_spend_dataset()
    zone_df = create_zone(poi_df)

    # Load and assign parking zones
    parking_df = get_parking_dataset()
    parking_df = assign_parking_zones(parking_df, zone_df)

    survived_zones = []

    for zone_id in zone_df['zone_id']:
        zone_pois = filter_df_based_on_zone(poi_df, zone_id)
        if len(zone_pois) == 0:
            continue

        total_spend = get_spendparam_years(zone_pois, spend_param, year)
        if total_spend == 0:
            continue

        # Filter POIs by category
        if sub_category:
            category_pois = filter_pois_by_sub_category(zone_pois, sub_category)
        elif top_category:
            category_pois = filter_pois_by_top_category(zone_pois, top_category)
        else:
            continue

        category_spend = get_spendparam_years(category_pois, spend_param, year)
        spend_share_condition = (category_spend / total_spend) * 100 >= percent_threshold

        # Check parking lot count
        filtered_parking_df = filter_df_based_on_zone(parking_df, zone_id)
        parking_lot_condition = get_num_parking(filtered_parking_df) >= min_parking_lots

        # Combine based on logic
        if logic == 'AND' and spend_share_condition and parking_lot_condition:
            survived_zones.append(zone_id)
        elif logic == 'OR' and (spend_share_condition or parking_lot_condition):
            survived_zones.append(zone_id)

    return zone_df[zone_df['zone_id'].isin(survived_zones)]

medium_16_test_cases = [
  (40, 'RAW_TOTAL_SPEND', 2024, 2, 'OR', None, 'Advertising Agencies'),
  (50, 'RAW_TOTAL_SPEND', 2022, 4, 'OR', None, 'Full-Service Restaurants'),
  (60, 'RAW_TOTAL_SPEND', 2022, 2, 'AND', None, 'Offices of Dentists'),
  (60, 'RAW_TOTAL_SPEND', 2024, 3, 'OR', None, 'Advertising Agencies'),
  (30, 'RAW_TOTAL_SPEND', 2019, 2, 'OR', 'Legal Services', None),
  (30, 'RAW_TOTAL_SPEND', 2020, 2, 'AND', None, 'Offices of Dentists'),
]

all_valid_zones = set()
for i, (percent_threshold, spend_param, year, min_parking_lots, logic, top_category, sub_category) in enumerate(medium_16_test_cases):
    obj_path = f'/Users/mihokoda/Desktop/CityLLM/test_results/med/16/tc_med_16_{i}/objective.csv'
    obj = pd.read_csv(obj_path)
    obj_zones = set(obj['zone_id'].unique())    
    
    valid_zones_df = medium_16(percent_threshold, spend_param, year, min_parking_lots, logic, top_category, sub_category)
    result_zones = set(valid_zones_df['zone_id'].unique())
    
    all_valid_zones.update(result_zones)
    print(f"Test case {i} (params: {percent_threshold}, {spend_param}, {year}, {min_parking_lots}, {logic}, {top_category}, {sub_category}):")
    print(f"  Found zones: {len(result_zones)}")
    print(f"  Objective zones: {len(obj_zones)}")
    print(f"  Match: {len(result_zones) == len(obj_zones)}")