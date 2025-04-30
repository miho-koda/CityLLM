import sys
import os
import pandas as pd

code_dir = "/Users/mihokoda/Desktop/CityLLM/code"
if code_dir not in sys.path:
    sys.path.insert(0, code_dir)

from site_selection.loader import get_poi_spend_dataset
from site_selection.zone import create_zone
from site_selection.filter import get_transport_pois_in_zone

def simple_13(min_types_required):
    transport_types = ["bus_stop", "subway_entrance", "taxi", "station", "aerodrome"]

    poi_df = get_poi_spend_dataset()
    zone_df = create_zone(poi_df)

    transport_presence = {zone_id: set() for zone_id in zone_df["zone_id"]}

    for transport_type in transport_types:
        transport_map = get_transport_pois_in_zone(zone_df, transport_type)
        for zone_id, locations in transport_map.items():
            if locations:  
                transport_presence[zone_id].add(transport_type)

    qualified_zone_ids = [
        zone_id for zone_id, types in transport_presence.items()
        if len(types) >= min_types_required
    ]

    return zone_df[zone_df["zone_id"].isin(qualified_zone_ids)]


simple_13_test_cases = [
    (3),        
    (4),
    (3),           
    (2),  
    (3),
    (4),         
]

all_valid_zones = set()
for i, (min_types_required) in enumerate(simple_13_test_cases):
    obj_path = f'/Users/mihokoda/Desktop/CityLLM/test_results/sim/13/tc_sim_13_{i}/objective.csv'
    obj = pd.read_csv(obj_path)
    obj_zones = set(obj['zone_id'].unique())
    
    valid_zones_df = simple_13(min_types_required)
    result_zones = set(valid_zones_df['zone_id'].unique())
    
    all_valid_zones.update(result_zones)
    print(f"Test case {i} (params: {min_types_required}):")
    print(f"  Found zones: {len(result_zones)}")
    print(f"  Objective zones: {len(obj_zones)}")