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


def hard_13(num_transport_types, population, num_neighbors, max_competitors, max_pois, sub_category):
    # Get datasets and create zones
    poi_df = get_poi_spend_dataset()
    zone_df = create_zone(poi_df)
    
    # Pre-calculate all transport types by zone
    transport_types_by_zone = {}
    transport_types = ['bus_stop', 'subway_entrance', 'taxi', 'station']
    for zone_id in zone_df['zone_id']:
        types_in_zone = []
        for transport_type in transport_types:
            transport_pois = get_transport_pois_in_zone(zone_df, transport_type).get(zone_id, [])
            if len(transport_pois) > 0:
                types_in_zone.append(transport_type)
        transport_types_by_zone[zone_id] = types_in_zone
    
    # Pre-calculate populations for all zones
    zone_populations = {}
    for zone_id in zone_df['zone_id']:
        zone_populations[zone_id] = get_population(zone_id, zone_df)
    transport_dict_by_type = {
        t: get_transport_pois_in_zone(zone_df, t)
        for t in transport_types
        }
    survived_zones = []
    for zone_id in zone_df['zone_id']:
        types_in_zone = [
            t for t in transport_types
            if len(transport_dict_by_type[t].get(zone_id, [])) > 0
        ]
        num_types = sum(
            1 for t in transport_types
            if len(transport_dict_by_type[t].get(zone_id, [])) > 0
        )
        A = num_types >= num_transport_types
        # Calculate population with neighbors
        neighbor_zone_list = get_neighbor_zones(zone_df, zone_id, num_neighbors)
        total_population = zone_populations[zone_id]
        for neighbor_id in neighbor_zone_list:
            total_population += zone_populations.get(neighbor_id, 0)
        B = total_population >= population
        
        # Filter POIs by sub-category
        filtered_poi_df = filter_df_based_on_zone(poi_df, zone_id)
        sub_category_pois = filter_pois_by_sub_category(filtered_poi_df, sub_category)
        
        # Check competitor count (same as POI count for this sub-category)
        num_competitors = len(sub_category_pois)
        C = num_competitors < max_competitors
        
        # Check if POI count exceeds maximum
        D = num_competitors <= max_pois
        
        # Apply logic: (A and B) or (C and D)
        if (A and B) or (C and D):
            survived_zones.append(zone_id)
    
    filtered_zone_df = zone_df[zone_df['zone_id'].isin(survived_zones)]
    return filtered_zone_df


# Define the test cases based on your descriptions
hard_13_test_cases = [
    (4, 14000, 2, 2, 3, "Commercial Banking"),
    (3, 12000, 2, 2, 2, "Offices of Lawyers"),
    (3, 13000, 1, 3, 3, "Offices of Real Estate Agents and Brokers"),
    (3, 14000, 1, 2, 4, "Offices of Physicians, Mental Health Specialists"),
    (3, 12000, 2, 3, 2, "Advertising Agencies"),
    (4, 13000, 2, 2, 3, "Insurance Agencies and Brokerages"),
]

# Test loop with comprehensive debugging
all_matched = True
for i, (num_transport_types, population, num_neighbors, max_competitors, max_pois, sub_category) in enumerate(hard_13_test_cases):
    obj_path = f"/Users/mihokoda/Desktop/CityLLM/test_results/hard/13/tc_hard_13_{i}/objective.csv"
    obj = pd.read_csv(obj_path)
    obj_zones = set(obj["zone_id"].unique())

    result_df = hard_13(
        num_transport_types=num_transport_types,
        population=population,
        num_neighbors=num_neighbors,
        max_competitors=max_competitors,
        max_pois=max_pois,
        sub_category=sub_category,
    )
    result_zones = set(result_df["zone_id"].unique())
    
    match = obj_zones == result_zones
    all_matched = all_matched and match
    
    print(f"Test case {i}:")
    print(f"  Parameters: {num_transport_types}+ transport types AND {population}+ population with {num_neighbors} neighbors")
    print(f"  OR fewer than {max_competitors} competitors AND NOT more than {max_pois} POIs in '{sub_category}'")
    print(f"  Objective zones: {len(obj_zones)}")
    print(f"  Found zones:     {len(result_zones)}")
    print(f"  Match:           {match}")
    
           

print(f"All test cases matched: {all_matched}")