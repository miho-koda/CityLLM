import operator
import pandas as pd
import sys


code_dir = "/Users/mihokoda/Desktop/CityLLM/code"
if code_dir not in sys.path:
    sys.path.insert(0, code_dir)

from site_selection.loader import get_poi_spend_dataset
from site_selection.zone import create_zone
from site_selection.filter import filter_df_based_on_zone
from site_selection.analysis import get_spendparam_years

# Supported operators
OPERATOR_FN = {
    "<": operator.lt,
    "<=": operator.le,
    ">": operator.gt,
    ">=": operator.ge,
    "==": operator.eq
}

def hard_2(start_year, end_year, filters):

    poi_df = get_poi_spend_dataset()
    zone_df = create_zone(poi_df)

    survived_zones = []

    for zone_id in zone_df["zone_id"]:
        zone_pois = filter_df_based_on_zone(poi_df, zone_id)
        passed_all = True

        for spend_param, op_str, threshold, mode in filters:
            if op_str not in OPERATOR_FN:
                print(f"❌ Unsupported operator: {op_str}")
                passed_all = False
                break

            values = [
                get_spendparam_years(zone_pois, spend_param, year)
                for year in range(start_year, end_year + 1)
            ]

            if not values or all(v == 0 for v in values):
                passed_all = False
                break

            agg = sum(values)
            result = agg if mode == "sum" else agg / len(values)

            if not OPERATOR_FN[op_str](result, threshold):
                passed_all = False
                break

        if passed_all:
            survived_zones.append(zone_id)

    return zone_df[zone_df["zone_id"].isin(survived_zones)]

hard_2_test_cases = [
    (2020, 2023, [
        ("MEDIAN_SPEND_PER_CUSTOMER", "<", 40, "avg"),
        ("RAW_TOTAL_SPEND", ">", 55_000_000, "sum"),
        ("RAW_NUM_CUSTOMERS", ">", 120_000, "sum")
    ]),

    # 2. Coffee/bookshop
    (2019, 2022, [
        ("SPEND_PCT_CHANGE_VS_PREV_YEAR", ">=", 0.04, "avg"),
        ("SPEND_PCT_CHANGE_VS_PREV_YEAR", ">", 0.10, "avg"),
        ("MEDIAN_SPEND_PER_TRANSACTION", "<", 25, "avg"),
        ("RAW_NUM_TRANSACTIONS", ">", 180_000, "sum")
    ]),

    # 3. Family-owned pizzeria
    (2021, 2023, [
        ("RAW_TOTAL_SPEND", ">", 70_000_000, "sum"),
        ("MEDIAN_SPEND_PER_CUSTOMER", ">", 200, "avg"),
        ("RAW_NUM_CUSTOMERS", ">", 250_000, "sum"),
        ("SPEND_PCT_CHANGE_VS_PREV_YEAR", ">", 0.08, "avg")
    ]),

    # 4. Community gym
    (2020, 2023, [
        ("MEDIAN_SPEND_PER_TRANSACTION", "<", 20, "avg"),
        ("RAW_TOTAL_SPEND", ">=", 60_000_000, "sum"),
        ("RAW_NUM_TRANSACTIONS", ">", 160_000, "sum"),
        ("RAW_NUM_CUSTOMERS", ">", 130_000, "sum"),
        ("SPEND_PCT_CHANGE_VS_PREV_YEAR", ">", 0, "avg")
    ]),

    # 5. Farmers' co-op
    (2021, 2024, [
        ("RAW_NUM_TRANSACTIONS", ">", 100_000, "sum"),
        ("RAW_TOTAL_SPEND", ">", 45_000_000, "sum"),
        ("MEDIAN_SPEND_PER_CUSTOMER", "<=", 35, "avg"),
        ("RAW_NUM_CUSTOMERS", ">", 150_000, "sum"),
        ("SPEND_PCT_CHANGE_VS_PREV_YEAR", ">", 0, "avg")
    ]),

    # 6. Boba tea shop
    (2021, 2023, [
        ("RAW_TOTAL_SPEND", ">", 40_000_000, "sum"),
        ("MEDIAN_SPEND_PER_TRANSACTION", "<", 18, "avg")
    ]),
]

all_valid_zones = set() 

for i, (start_year, end_year, filters) in enumerate(hard_2_test_cases):
    obj_path = f'/Users/mihokoda/Desktop/CityLLM/test_results/hard/2/tc_hard_2_{i}/objective.csv'
    obj = pd.read_csv(obj_path)
    obj_zones = set(obj['zone_id'].unique())    
    
    # Directly call the function with filters list
    valid_zones_df = hard_2(start_year, end_year, filters)

    result_zones = set(valid_zones_df['zone_id'].unique())
    
    all_valid_zones.update(result_zones)
    print(f"Test case {i} (params: {start_year}, {end_year}, {filters}):")
    print(f"  Found zones: {len(result_zones)}")
    print(f"  Objective zones: {len(obj_zones)}")
    print(f"  Match: {result_zones == obj_zones}")
