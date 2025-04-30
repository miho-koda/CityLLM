import sys
import os
import pandas as pd

code_dir = "/Users/mihokoda/Desktop/CityLLM/code"
if code_dir not in sys.path:
    sys.path.insert(0, code_dir)

from site_selection.loader import get_poi_spend_dataset
from site_selection.zone import create_zone
from site_selection.filter import filter_pois_by_top_category, filter_pois_by_sub_category


def simple_4(top_category, sub_category, num):
    poi_df = get_poi_spend_dataset()
    zone_df = create_zone(poi_df)
    
    filtered_poi_df = poi_df.copy()
    if top_category:
        filtered_poi_df = filter_pois_by_top_category(filtered_poi_df, top_category)
    if sub_category:
        filtered_poi_df = filter_pois_by_sub_category(filtered_poi_df, sub_category)
    
    zone_counts = filtered_poi_df['zone_id'].value_counts()
    zone_competitor_map = zone_counts.to_dict()
    
    zone_df['num_competitors'] = zone_df['zone_id'].map(zone_competitor_map).fillna(0)
    
    filtered_zone_df = zone_df[zone_df['num_competitors'] < num]

    filtered_zone_df = filtered_zone_df.drop(columns=['num_competitors'])
    
    return filtered_zone_df

simple_4_test_cases = [
    ("Other Schools and Instruction", "Exam Preparation and Tutoring", 3),
    ("Other Amusement and Recreation Industries", "Fitness and Recreational Sports Centers", 3),
    ("Offices of Physicians", "Offices of Physicians, Mental Health Specialists", 2),
    ("Residential Building Construction", "Residential Remodelers", 4),
    ("Offices of Physicians", None, 3),
    ("Offices of Real Estate Agents and Brokers", None, 2)
]
all_valid_zones = set()

for i, (top_category, sub_category, num) in enumerate(simple_4_test_cases):
    obj_path = f'/Users/mihokoda/Desktop/CityLLM/test_results/sim/4/tc_sim_4_{i}/objective.csv'
    obj = pd.read_csv(obj_path)
    obj_zones = set(obj['zone_id'].unique())
    
    # Pass all three parameters to simple_4
    valid_zones_df = simple_4(top_category, sub_category, num)
    result_zones = set(valid_zones_df['zone_id'].unique())
    
    all_valid_zones.update(result_zones)
    print(f"Test case {i} (params: {top_category}, {sub_category}, {num}):")
    print(f"  Found zones: {len(result_zones)}")
    print(f"  Objective zones: {len(obj_zones)}")
    print(f"  Match: {result_zones == obj_zones}")
