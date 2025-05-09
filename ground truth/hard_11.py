import os
import sys
import pandas as pd

code_dir = "/Users/mihokoda/Desktop/CityLLM/code"
if code_dir not in sys.path:
    sys.path.insert(0, code_dir)

from site_selection.loader import get_poi_spend_dataset, get_parking_dataset
from site_selection.zone import create_zone, assign_parking_zones, get_zone_center, get_neighbor_zones
from site_selection.analysis import get_spendparam_years, get_num_parking, get_largest_parking_lot_area, get_largest_parking_capacity, get_distance_km
from site_selection.filter import filter_df_based_on_zone, filter_pois_by_top_category, filter_pois_by_sub_category, get_transport_pois_in_zone
from site_selection.population import get_population


def hard_11(population, num_zones, num_transport, transport_type, max_distance):
    # Get datasets and create zones
    poi_df = get_poi_spend_dataset()
    zone_df = create_zone(poi_df)

    # Get transport POIs by zone once
    transport_dict = get_transport_pois_in_zone(zone_df, transport_type)
    
    # Calculate population for all zones once
    zone_populations = {}
    for zone_id in zone_df['zone_id']:
        zone_populations[zone_id] = get_population(zone_id, zone_df)

    survived_zones = []
    for zone_id in zone_df['zone_id']:
        # Calculate population with neighbors (Simple 6)
        neighbor_zone_list = get_neighbor_zones(zone_df, zone_id, num_zones)
        total_population = zone_populations[zone_id]
        for neighbor_id in neighbor_zone_list:
            total_population += zone_populations[neighbor_id]
            
        A = total_population >= population  # population condition
        
        # Check transport count (Simple 9)
        transport_pois = transport_dict.get(zone_id, [])
        B = len(transport_pois) >= num_transport  # transport count condition
        
        # Check distance to nearest transport (Simple 10)
        C = False  # Default to False
        if transport_pois:
            zone_center = get_zone_center(zone_df, zone_id)
            for transport_poi in transport_pois:
                dist = get_distance_km(zone_center[0], zone_center[1], 
                                    transport_poi[0], transport_poi[1])
                if dist <= max_distance/1000:  # Convert meters to kilometers
                    C = True
                    break  # Exit loop as soon as we find one within range
        
        if (A or B) and C:
            survived_zones.append(zone_id)
    
    filtered_zone_df = zone_df[zone_df['zone_id'].isin(survived_zones)]
    return filtered_zone_df

hard_10_test_cases = [
    (14000, 2, 4, "subway_entrance", 300),
    (13000, 3, 5, "bus_stop", 250),
    (12000, 2, 5, "subway_entrance", 300),
    (14000, 2, 5, "station", 300),
    (10000, 2, 4, "subway_entrance", 250),
    (15000, 2, 5, "bus_stop", 200),
]

all_valid_zones = set()

for i, (population, num_zones, num_transport, transport_type, max_distance) in enumerate(hard_10_test_cases):
    obj_path = f"/Users/mihokoda/Desktop/CityLLM/test_results/hard/10/tc_hard_10_{i}/objective.csv"
    obj = pd.read_csv(obj_path)
    obj_zones = set(obj["zone_id"].unique())

    result_df = hard_11(
        population=population,
        num_zones=num_zones,
        num_transport=num_transport,
        transport_type=transport_type,
        max_distance=max_distance,
    )
    result_zones = set(result_df["zone_id"].unique())
    all_valid_zones.update(result_zones)

    print(f"Test case {i}:")
    print(f"  Objective zones: {len(obj_zones)}")
    print(f"  Found zones:     {len(result_zones)}")
    print(f"  Match:           {obj_zones == result_zones}")
    print()
