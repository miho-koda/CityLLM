import operator
import ast
import pandas as pd
import sys


code_dir = "/Users/mihokoda/Desktop/CityLLM/code"
if code_dir not in sys.path:
    sys.path.insert(0, code_dir)

from site_selection.loader import get_poi_spend_dataset, get_parking_dataset
from site_selection.zone import create_zone, assign_parking_zones
from site_selection.filter import filter_df_based_on_zone, filter_pois_by_sub_category, filter_pois_by_top_category
from site_selection.analysis import get_spendparam_years

def hard_4(
    num_competitors: int,
    num_parking: int,
    num_pois: int,
    logic_expr: str, 
    sub_category: str = None,
    top_category: str = None,
):
    poi_df = get_poi_spend_dataset()
    parking_df = get_parking_dataset()
    zone_df = create_zone(poi_df)
    parking_df = assign_parking_zones(parking_df, zone_df)

    survived_zones = []

    for zone_id in zone_df["zone_id"]:
        zone_pois = filter_df_based_on_zone(poi_df, zone_id)
        zone_parking = parking_df[parking_df["zone_id"] == zone_id]

        # Apply filters
        if sub_category:
            relevant_pois = filter_pois_by_sub_category(zone_pois, sub_category)
        elif top_category:
            relevant_pois = filter_pois_by_top_category(zone_pois, top_category)
        else:
            relevant_pois = zone_pois

        # A: number of competitors
        A = len(relevant_pois) < num_competitors
        # B: number of parking lots
        B = len(zone_parking) >= num_parking
        # C: number of POIs
        C = len(relevant_pois) >= num_pois

        # Safe evaluation
        expr = logic_expr.replace("AND", "and").replace("OR", "or").replace("NOT", "not")
        try:
            if eval(expr, {"A": A, "B": B, "C": C}):
                survived_zones.append(zone_id)
        except Exception as e:
            print(f"Invalid logic expression '{logic_expr}' for zone {zone_id}: {e}")
            continue

    return zone_df[zone_df["zone_id"].isin(survived_zones)]

hard_4_test_cases = [
    # 1. Skincare bar
    (3, 4, 6, "(A AND B AND C)", "Beauty Salons", None),

    # 2. Minimalist clothing store
    (4, 3, 5, "(A AND B AND C)", None, "Other Miscellaneous Store Retailers"),

    # 3. Health-food café
    (3, 4, 6, "(A AND B AND C)", "Snack and Nonalcoholic Beverage Bars", None),

    # 4. Juice + fitness hybrid
    (2, 3, 5, "(A AND B AND C)", None, "Fitness and Recreational Sports Centers"),

    # 5. Modern bistro
    (5, 4, 7, "(A AND B AND C)", "Full-Service Restaurants", None),

    # 6. Wine lounge
    (3, 3, 6, "(A AND B AND C)", None, "Drinking Places (Alcoholic Beverages)"),
]
all_valid_zones = set()

for i, (num_competitors, num_parking, num_pois, logic_expr, sub_cat, top_cat) in enumerate(hard_4_test_cases):
    obj_path = f"/Users/mihokoda/Desktop/CityLLM/test_results/hard/4/tc_hard_4_{i}/objective.csv"
    obj = pd.read_csv(obj_path)
    obj_zones = set(obj["zone_id"].unique())

    result_df = hard_4(
        num_competitors=num_competitors,
        num_parking=num_parking,
        num_pois=num_pois,
        logic_expr=logic_expr,
        sub_category=sub_cat,
        top_category=top_cat,
    )
    result_zones = set(result_df["zone_id"].unique())
    all_valid_zones.update(result_zones)

    print(f"Test case {i}:")
    print(f"  Objective zones: {len(obj_zones)}")
    print(f"  Found zones:     {len(result_zones)}")
    print(f"  Match:           {obj_zones == result_zones}")
    print()
    