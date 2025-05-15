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


def hard_12(num_parking_lots, num_pois, category_type, category_name, max_population, num_neighbors):
    """
    Function for hard_12 criteria:
    - Either has enough parking lots OR enough POIs in given category
    - BUT NOT if population across zone and neighbors exceeds maximum

    Parameters:
    num_parking_lots: Minimum number of parking lots required
    num_pois: Minimum number of POIs in the specified category
    category_type: Either 'sub_category' or 'top_category'
    category_name: Name of the category to filter by
    max_population: Maximum population across zone and neighbors
    num_neighbors: Number of neighbors to consider for population
    """
    # Get datasets and create zones
    poi_df = get_poi_spend_dataset()
    zone_df = create_zone(poi_df)
    
    # Get parking dataset and assign to zones
    parking_df = get_parking_dataset()
    parking_df = assign_parking_zones(parking_df, zone_df)  # Note: assign_parking_zones returns the updated dataframe
    
    # Pre-calculate populations for all zones
    zone_populations = {}
    for zone_id in zone_df['zone_id']:
        zone_populations[zone_id] = get_population(zone_id, zone_df)
    
    survived_zones = []
    for zone_id in zone_df['zone_id']:
        # Check number of parking lots (Simple 2)
        filtered_parking_df = filter_df_based_on_zone(parking_df, zone_id)
        num_parking = get_num_parking(filtered_parking_df)  # No zone_id and zone_df parameters
        A = num_parking >= num_parking_lots
        
        # Check POIs in specific category (Simple 4/7)
        filtered_poi_df = filter_df_based_on_zone(poi_df, zone_id)
        if category_type == 'sub_category':
            category_filtered = filter_pois_by_sub_category(filtered_poi_df, category_name)
        else:  # top_category
            category_filtered = filter_pois_by_top_category(filtered_poi_df, category_name)
        
        num_category_pois = len(category_filtered)
        B = num_category_pois >= num_pois
        
        # Check population with neighbors (Simple 6)
        neighbor_zone_list = get_neighbor_zones(zone_df, zone_id, num_neighbors)
        total_population = zone_populations[zone_id]
        for neighbor_id in neighbor_zone_list:
            total_population += zone_populations.get(neighbor_id, 0)
        
        C = total_population <= max_population  # C is "NOT exceeding max population"
        
        # Apply logic: (A or B) and C
        if (A or B) and C:
            survived_zones.append(zone_id)
    
    filtered_zone_df = zone_df[zone_df['zone_id'].isin(survived_zones)]
    return filtered_zone_df


def debug_zone_hard12(zone_id, num_parking_lots, num_pois, category_type, category_name, max_population, num_neighbors):
    """Detailed debugging for a specific zone in hard_12"""
    # Get datasets
    poi_df = get_poi_spend_dataset()
    zone_df = create_zone(poi_df)
    parking_df = get_parking_dataset()
    parking_df = assign_parking_zones(parking_df, zone_df)
    
    # Check parking lots
    filtered_parking_df = filter_df_based_on_zone(parking_df, zone_id)
    num_parking = get_num_parking(filtered_parking_df)
    A = num_parking >= num_parking_lots
    
    # Check POIs in specific category
    filtered_poi_df = filter_df_based_on_zone(poi_df, zone_id)
    if category_type == 'sub_category':
        category_filtered = filter_pois_by_sub_category(filtered_poi_df, category_name)
    else:  # top_category
        category_filtered = filter_pois_by_top_category(filtered_poi_df, category_name)
    
    num_category_pois = len(category_filtered)
    B = num_category_pois >= num_pois
    
    # Check population with neighbors
    zone_population = get_population(zone_id, zone_df)
    neighbor_zone_list = get_neighbor_zones(zone_df, zone_id, num_neighbors)
    
    total_population = zone_population
    neighbor_populations = {}
    
    for neighbor_id in neighbor_zone_list:
        neighbor_pop = get_population(neighbor_id, zone_df)
        neighbor_populations[neighbor_id] = neighbor_pop
        total_population += neighbor_pop
    
    C = total_population <= max_population
    
    # Print detailed debug info
    print(f"Debug for zone {zone_id}:")
    print(f"  Parking condition (A):")
    print(f"    Number of parking lots: {num_parking}")
    print(f"    Threshold: {num_parking_lots}")
    print(f"    A = {A}")
    
    print(f"  POI category condition (B):")
    print(f"    Category type: {category_type}")
    print(f"    Category name: {category_name}")
    print(f"    Number of POIs in category: {num_category_pois}")
    print(f"    Threshold: {num_pois}")
    print(f"    B = {B}")
    
    print(f"  Population condition (C):")
    print(f"    Zone population: {zone_population}")
    print(f"    Number of neighbors: {num_neighbors}")
    print(f"    Neighbor zones: {neighbor_zone_list}")
    print(f"    Neighbor populations: {neighbor_populations}")
    print(f"    Total population: {total_population}")
    print(f"    Max population: {max_population}")
    print(f"    C = {C}")
    
    print(f"  Final logic: (A or B) and C = {(A or B) and C}")
    return (A, B, C)


# Define the test cases based on your descriptions
hard_12_test_cases = [
    (4, 6, "sub_category", "Full-Service Restaurants", 12000, 2),
    (5, 5, "sub_category", "Snack and Nonalcoholic Beverage Bars", 14000, 4),
    (3, 4, "sub_category", "Automotive Parts, Accessories, and Tire Stores", 15000, 0),
    (4, 5, "sub_category", "Snack and Nonalcoholic Beverage Bars", 12000, 2),
    (5, 6, "sub_category", "Full-Service Restaurants", 14000, 0),
    (3, 5, "top_category", "Other Miscellaneous Store Retailers", 25000, 5),
]

# Test loop with comprehensive debugging
all_matched = True
for i, (num_parking_lots, num_pois, category_type, category_name, max_population, num_neighbors) in enumerate(hard_12_test_cases):
    obj_path = f"/Users/mihokoda/Desktop/CityLLM/test_results/hard/12/tc_hard_12_{i}/objective.csv"
    obj = pd.read_csv(obj_path)
    obj_zones = set(obj["zone_id"].unique())

    result_df = hard_12(
        num_parking_lots=num_parking_lots,
        num_pois=num_pois,
        category_type=category_type,
        category_name=category_name,
        max_population=max_population,
        num_neighbors=num_neighbors,
    )
    result_zones = set(result_df["zone_id"].unique())
    
    match = obj_zones == result_zones
    all_matched = all_matched and match
    
    print(f"Test case {i}:")
    print(f"  Parameters: Parking lots >= {num_parking_lots} OR {category_type} '{category_name}' POIs >= {num_pois}")
    print(f"  NOT population > {max_population} across {num_neighbors} neighbors")
    print(f"  Objective zones: {len(obj_zones)}")
    print(f"  Found zones:     {len(result_zones)}")
    print(f"  Match:           {match}")
    
    if not match:
        # Find zones in objective but not in result
        missing_zones = obj_zones - result_zones
        if missing_zones:
            print(f"  Missing zones: {missing_zones}")
            # Debug up to 3 missing zones
            for zone_id in list(missing_zones)[:3]:
                debug_zone_hard12(zone_id, num_parking_lots, num_pois, category_type, 
                                 category_name, max_population, num_neighbors)
        
        # Find zones in result but not in objective
        extra_zones = result_zones - obj_zones
        if extra_zones:
            print(f"  Extra zones: {extra_zones}")
            # Debug up to 3 extra zones
            for zone_id in list(extra_zones)[:3]:
                debug_zone_hard12(zone_id, num_parking_lots, num_pois, category_type, 
                                 category_name, max_population, num_neighbors)
            
    print()

print(f"All test cases matched: {all_matched}")