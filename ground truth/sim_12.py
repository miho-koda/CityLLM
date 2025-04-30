import sys
import os

code_dir = "/Users/mihokoda/Desktop/CityLLM/code"
if code_dir not in sys.path:
    sys.path.insert(0, code_dir)

from site_selection.loader import get_poi_spend_dataset
from site_selection.zone import create_zone
from site_selection.analysis import get_distance_km
from site_selection.filter import get_transport_pois_in_zone
import pandas as pd

def simple_12(min_count, transportation_type, dist_threshold_m):
    poi_df = get_poi_spend_dataset()
    zone_df = create_zone(poi_df)

    transport_dict = get_transport_pois_in_zone(zone_df, transportation_type)

    survived_zones = []
    for zone_id in zone_df["zone_id"]:
        center_lat, center_lng = get_zone_center(zone_df, zone_id)
        transport_pois = transport_dict.get(zone_id, [])
        
        count_within = 0
        for poi_lat, poi_lng in transport_pois:
            distance_m = get_distance_km(center_lat, center_lng, poi_lat, poi_lng) * 1000
            if distance_m <= dist_threshold_m:
                count_within += 1
                if count_within >= min_count:
                    survived_zones.append(zone_id)
                    break  

    return zone_df[zone_df["zone_id"].isin(survived_zones)]


simple_12_test_cases = [
    (6, "station", 400),        
    (3, "bus_stop", 200),
    (5, "subway_entrance", 300),           
    (4, "taxi", 250),  
    (6, "station", 400),         
    (3, "aerodrome", 500)  #aerdrome is a very optimal means of transportation  
]

all_valid_zones = set()
for i, (min_count, transportation_type, dist_threshold_m) in enumerate(simple_12_test_cases):
    obj_path = f'/Users/mihokoda/Desktop/CityLLM/test_results/sim/12/tc_sim_12_{i}/objective.csv'
    obj = pd.read_csv(obj_path)
    obj_zones = set(obj['zone_id'].unique())
    
    valid_zones_df = simple_12(min_count, transportation_type, dist_threshold_m)
    result_zones = set(valid_zones_df['zone_id'].unique())
    
    all_valid_zones.update(result_zones)
    print(f"Test case {i} (params: {min_count}, {transportation_type}, {dist_threshold_m}):")
    print(f"  Found zones: {len(result_zones)}")
    print(f"  Objective zones: {len(obj_zones)}")