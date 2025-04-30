import sys
import os
import pandas as pd

code_dir = "/Users/mihokoda/Desktop/CityLLM/code"
if code_dir not in sys.path:
    sys.path.insert(0, code_dir)

from site_selection.loader import get_poi_spend_dataset
from site_selection.zone import create_zone
from site_selection.filter import filter_df_based_on_zone, filter_pois_by_top_category, filter_pois_by_sub_category
import pandas as pd

def simple_8(num, top_categories, sub_categories):
    poi_df = get_poi_spend_dataset()
    zone_df = create_zone(poi_df)
    
    survived_zones = []
    
    for zone_id in zone_df['zone_id']:
        base_filtered_poi_df = filter_df_based_on_zone(poi_df, zone_id)
        
        combined_filtered_pois = pd.DataFrame()
        
        if sub_categories:
            for sub_category in sub_categories:
                category_filtered = filter_pois_by_sub_category(base_filtered_poi_df, sub_category)
                combined_filtered_pois = pd.concat([combined_filtered_pois, category_filtered])
        
        elif top_categories:
            for top_category in top_categories:
                category_filtered = filter_pois_by_top_category(base_filtered_poi_df, top_category)
                combined_filtered_pois = pd.concat([combined_filtered_pois, category_filtered])
        else:
            combined_filtered_pois = base_filtered_poi_df
        
        combined_filtered_pois = combined_filtered_pois.drop_duplicates()
        
        if len(combined_filtered_pois) >= num:
            survived_zones.append(zone_id)
            
    filtered_zone_df = zone_df[zone_df['zone_id'].isin(survived_zones)]
    return filtered_zone_df

simple_8_test_cases = [
    (8, None, ["Beauty Salons", "Women's Clothing Stores"]),
    (12, None, ["Drinking Places (Alcoholic Beverages)", "Snack and Nonalcoholic Beverage Bars"]),
    (6, None, ["Lessors of Residential Buildings and Dwellings", "Elementary and Secondary Schools"]),
    (15, None, ["Supermarkets and Other Grocery (except Convenience) Stores", "Snack and Nonalcoholic Beverage Bars"]),
    (10, None, ["Barber Shops", "Jewelry Stores", "Other Personal Care Services"]),
    (7, None, ["Women's Clothing Stores", "Jewelry Stores"])
]

all_valid_zones = set()


for i, (num, top_categories, sub_categories) in enumerate(simple_8_test_cases):
    obj_path = f'/Users/mihokoda/Desktop/CityLLM/test_results/sim/8/tc_sim_8_{i}/objective.csv'
    obj = pd.read_csv(obj_path)
    obj_zones = set(obj['zone_id'].unique())
    
    valid_zones_df = simple_8(num, top_categories, sub_categories)
    result_zones = set(valid_zones_df['zone_id'].unique())
    
    all_valid_zones.update(result_zones)
    print(f"Test case {i} (params: {num}, {top_categories}, {sub_categories}):")
    print(f"  Found zones: {len(result_zones)}")
    print(f"  Objective zones: {len(obj_zones)}")
    print(f"  Match: {result_zones == obj_zones}")
