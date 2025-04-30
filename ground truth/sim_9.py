import sys
import os
import pandas as pd

code_dir = "/Users/mihokoda/Desktop/CityLLM/code"
if code_dir not in sys.path:
    sys.path.insert(0, code_dir)

from site_selection.loader import get_poi_spend_dataset
from site_selection.zone import create_zone
from site_selection.filter import get_transport_pois_in_zone

def simple_9(num, transportation_type):
    poi_spend_df = get_poi_spend_dataset()
    
   
    zone_df = create_zone(poi_spend_df)
    
    suitable_zones = []
    for zone_id in zone_df['zone_id'].unique():
        transport_pois = get_transport_pois_in_zone(zone_df, transportation_type)
        if zone_id in transport_pois and len(transport_pois[zone_id]) >= num:
            suitable_zones.append(zone_id)
    
    filtered_zone_df = zone_df[zone_df['zone_id'].isin(suitable_zones)]

    return filtered_zone_df

simple_9_test_cases = [
    (4, "subway_entrance"),
    (6, "bus_stop"),
    (3, "station"),
    (7, "taxi"),
    (2, "subway_entrance"),
    (4, "taxi")
]

all_valid_zones = set()


for i, (num, transportation_type) in enumerate(simple_9_test_cases):
    obj_path = f'/Users/mihokoda/Desktop/CityLLM/test_results/sim/9/tc_sim_9_{i}/objective.csv'
    obj = pd.read_csv(obj_path)
    obj_zones = set(obj['zone_id'].unique())
    
    valid_zones_df = simple_9(num, transportation_type)
    result_zones = set(valid_zones_df['zone_id'].unique())
    
    all_valid_zones.update(result_zones)
    print(f"Test case {i} (params: {num}, {transportation_type}):")
    print(f"  Found zones: {len(result_zones)}")
    print(f"  Objective zones: {len(obj_zones)}")
    print(f"  Match: {result_zones == obj_zones}")
