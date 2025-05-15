import os
import sys
import pandas as pd
import operator
import math

code_dir = "/Users/mihokoda/Desktop/CityLLM/code"
if code_dir not in sys.path:
    sys.path.insert(0, code_dir)

from site_selection.loader import get_poi_spend_dataset, get_parking_dataset
from site_selection.zone import create_zone, assign_parking_zones, get_zone_center, get_neighbor_zones
from site_selection.analysis import get_spendparam_years, get_num_parking, get_largest_parking_lot_area, get_largest_parking_capacity, get_distance_km
from site_selection.filter import filter_df_based_on_zone, filter_pois_by_top_category, filter_pois_by_sub_category, get_transport_pois_in_zone
from site_selection.population import get_population


def hard_11(population, pop_op, num_zones, num_transport, transport_op, transport_type, max_distance, distance_op, logic_expr):
    # Operator mapping
    ops = {
        "<": operator.lt,
        "<=": operator.le,
        ">": operator.gt,
        ">=": operator.ge,
        "==": operator.eq,
        "!=": operator.ne,
    }
    
    # Validate operators
    if pop_op not in ops or transport_op not in ops or distance_op not in ops:
        raise ValueError("Invalid comparison operator provided.")
    
    # Get datasets and create zones
    poi_df = get_poi_spend_dataset()
    zone_df = create_zone(poi_df)

    # Get transport POIs by zone once
    transport_dict = get_transport_pois_in_zone(zone_df, transport_type)
    print("Transport POIs:", transport_dict.get(320, []))
    # Calculate population for all zones once
    zone_populations = {}
    for zone_id in zone_df['zone_id']:
        zone_populations[zone_id] = get_population(zone_id, zone_df)
    
    survived_zones = []
    for zone_id in zone_df['zone_id']:
        # Calculate population with neighbors
        neighbor_zone_list = get_neighbor_zones(zone_df, zone_id, num_zones)
        total_population = zone_populations[zone_id]
        for neighbor_id in neighbor_zone_list:
            total_population += zone_populations.get(neighbor_id, 0)
            
        A = ops[pop_op](total_population, population)  # Population condition
        
        # Check transport count
        transport_pois = transport_dict.get(zone_id, [])
        B = ops[transport_op](len(transport_pois), num_transport)  # Transport count
        
        # Check distance to nearest transport
        if transport_pois:
            zone_center = get_zone_center(zone_df, zone_id)
            closest_distance_km = min(
                get_distance_km(zone_center[0], zone_center[1], poi[0], poi[1])
                for poi in transport_pois
            )
            C = ops[distance_op](closest_distance_km, max_distance / 1000)
        else:
            C = False
        
        # Evaluate the logical expression
        local_vars = {'A': A, 'B': B, 'C': C}
        try:
            if eval(logic_expr, {"__builtins__": {}}, local_vars):
                survived_zones.append(zone_id)
        except Exception as e:
            print(f"Error evaluating logic expression for zone {zone_id}: {e}")
    
    filtered_zone_df = zone_df[zone_df['zone_id'].isin(survived_zones)]
    return filtered_zone_df


hard_11_test_cases = [
    (14000, ">=", 2, 4, ">=", "subway_entrance", 300, "<=", "(A or B) and C"),
    (13000, ">=", 3, 5, ">=", "bus_stop", 250, "<=", "(A or B) and C"),
    (12000, ">=", 2, 5, ">=", "subway_entrance", 300, "<=", "(A or B) and C"),
    (14000, ">=", 2, 5, ">=", "subway_entrance", 300, "<=", "(A or B) and C"),
    (10000, ">=", 2, 4, ">=", "subway_entrance", 250, "<=", "(A or B) and C"),
    (15000, ">=", 2, 5, ">=", "bus_stop", 200, "<", "A and B and C"),
]

all_matched = True
for i, (population, pop_op, num_zones, num_transport, transport_op, transport_type, max_distance, distance_op, logic_expr) in enumerate(hard_11_test_cases):
    obj_path = f"/Users/mihokoda/Desktop/CityLLM/test_results/hard/11/tc_hard_11_{i}/objective.csv"
    obj = pd.read_csv(obj_path)
    obj_zones = set(obj["zone_id"].unique())

    result_df = hard_11(
        population=population,
        pop_op=pop_op,
        num_zones=num_zones,
        num_transport=num_transport,
        transport_op=transport_op,
        transport_type=transport_type,
        max_distance=max_distance,
        distance_op=distance_op,
        logic_expr=logic_expr,
    )
    result_zones = set(result_df["zone_id"].unique())
    
    match = obj_zones == result_zones
    all_matched = all_matched and match
    
    print(f"Test case {i}:")
    print(f"  Parameters: Population {pop_op} {population}, Neighbors: {num_zones}, {transport_type} {transport_op} {num_transport}, Distance {distance_op} {max_distance}m")
    print(f"  Logic: {logic_expr}")
    print(f"  Objective zones: {len(obj_zones)}")
    print(f"  Found zones:     {len(result_zones)}")
    print(f"  Match:           {match}")
    
    

print(f"All test cases matched: {all_matched}")