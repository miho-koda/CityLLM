import operator
import pandas as pd
import sys


code_dir = "/Users/mihokoda/Desktop/CityLLM/code"
if code_dir not in sys.path:
    sys.path.insert(0, code_dir)

from site_selection.loader import get_poi_spend_dataset
from site_selection.zone import create_zone
from site_selection.filter import filter_df_based_on_zone, filter_pois_by_sub_category, filter_pois_by_top_category
from site_selection.analysis import get_spendparam_years

# Supported operators
OPERATOR_FN = {
    "<": operator.lt,
    "<=": operator.le,
    ">": operator.gt,
    ">=": operator.ge,
    "==": operator.eq
}

import operator

OPERATOR_FN = {
    "<": operator.lt,
    "<=": operator.le,
    ">": operator.gt,
    ">=": operator.ge,
    "==": operator.eq
}

def hard_3(top_category: str, sub_category: str, max_competitors: int, 
           filter: list[tuple[str, str, float, str]], year_start: int, year_end: int) -> pd.DataFrame:
    poi_df = get_poi_spend_dataset()
    zone_df = create_zone(poi_df)
    survived_zones = []

    for zone_id in zone_df['zone_id']:
        zone_poi_df = filter_df_based_on_zone(poi_df, zone_id)

        # Competitor check
        if sub_category:
            competitor_df = filter_pois_by_sub_category(zone_poi_df, sub_category)
        elif top_category:
            competitor_df = filter_pois_by_top_category(zone_poi_df, top_category)
        else:
            continue  # If neither specified, skip this zone

        if len(competitor_df) > max_competitors:
            continue  # Too many competitors

        # Spend checks
        all_filters_passed = True
        for spend_param, op_str, threshold, mode in filter:
            yearly_values = [
                get_spendparam_years(zone_poi_df, spend_param, str(year))
                for year in range(year_start, year_end + 1)
            ]
            agg_value = sum(yearly_values) if mode == "sum" else (
                sum(yearly_values) / len(yearly_values) if yearly_values else 0
            )
            if not OPERATOR_FN[op_str](agg_value, threshold):
                all_filters_passed = False
                break

        if all_filters_passed:
            survived_zones.append(zone_id)

    return zone_df[zone_df['zone_id'].isin(survived_zones)]

hard_3_test_cases = [
   (
        "Restaurants and Other Eating Places", "Full-Service Restaurants", 3,
        [("RAW_NUM_TRANSACTIONS", ">", 300_000, "sum")], 2022, 2024
    ),
    (
        "Offices of Other Health Practitioners", "Offices of All Other Miscellaneous Health Practitioners", 2,
        [("RAW_NUM_CUSTOMERS", ">", 120_000, "sum")], 2022, 2024
    ),
    (
        "Beer, Wine, and Liquor Stores", "Beer, Wine, and Liquor Stores", 4,
        [("RAW_TOTAL_SPEND", ">", 40_000_000, "sum")], 2022, 2024
    ),
    (
        "Gasoline Stations", "Gasoline Stations with Convenience Stores", 3,
        [("RAW_NUM_TRANSACTIONS", ">", 500_000, "sum")], 2022, 2024
    ),
    (
        "Restaurants and Other Eating Places", "Full-Service Restaurants", 4,
        [("RAW_TOTAL_SPEND", ">", 60_000_000, "sum"),
         ("RAW_NUM_TRANSACTIONS", ">", 400_000, "sum")], 2022, 2024
    ),
    (
        "Personal Care Services", "Beauty Salons", 3,
        [("RAW_NUM_CUSTOMERS", ">", 200_000, "sum"),
         ("RAW_TOTAL_SPEND", ">", 35_000_000, "sum")], 2022, 2024
    ),
]

all_valid_zones = set() 

for i, (top_category, sub_category, max_competitors, filters, year_start, year_end) in enumerate(hard_3_test_cases):
    obj_path = f'/Users/mihokoda/Desktop/CityLLM/test_results/hard/3/tc_hard_3_{i}/objective.csv'
    obj = pd.read_csv(obj_path)
    obj_zones = set(obj['zone_id'].unique())    
    
    # Directly call the function with filters list
    valid_zones_df = hard_3(top_category, sub_category, max_competitors, filters, year_start, year_end)

    result_zones = set(valid_zones_df['zone_id'].unique())
    
    all_valid_zones.update(result_zones)
    print(f"Test case {i} (params: {top_category}, {sub_category}, {max_competitors}, {filters}, {year_start}, {year_end}):")
    print(f"  Found zones: {len(result_zones)}")
    print(f"  Objective zones: {len(obj_zones)}")
    print(f"  Match: {result_zones == obj_zones}")
