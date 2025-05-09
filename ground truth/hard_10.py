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


def hard_10(
    num_pois, poi_op,
    num_transport, transport_op, transport_type,
    num_competitors, competitor_op,
    logic_expr,
    sub_category1=None, sub_category2=None,
    top_category1=None, top_category2=None
):
    import operator

    # Operator mapping
    ops = {
        "<": operator.lt,
        "<=": operator.le,
        ">": operator.gt,
        ">=": operator.ge,
        "==": operator.eq,
        "!=": operator.ne,
    }

    # Get datasets and create zones
    poi_df = get_poi_spend_dataset()
    zone_df = create_zone(poi_df)

    # Get transport POIs by zone once
    transport_dict = get_transport_pois_in_zone(zone_df, transport_type)
    
    survived_zones = []

    for zone_id in zone_df['zone_id']:
        # Get filtered POIs for current zone
        filtered_poi_df = filter_df_based_on_zone(poi_df, zone_id)

        # Get POI count for first category (Simple 7)
        if sub_category1:
            cat_filtered_1 = filter_pois_by_sub_category(filtered_poi_df, sub_category1)
        elif top_category1:
            cat_filtered_1 = filter_pois_by_top_category(filtered_poi_df, top_category1)
        else:
            continue

        # Get transport count (Simple 9)
        transport_count = len(transport_dict.get(zone_id, []))

        # Get competitor count for second category (Simple 4)
        if sub_category2:
            cat_filtered_2 = filter_pois_by_sub_category(filtered_poi_df, sub_category2)
        elif top_category2:
            cat_filtered_2 = filter_pois_by_top_category(filtered_poi_df, top_category2)
        else:
            cat_filtered_2 = cat_filtered_1  # Fallback to same category

        # Evaluate conditions using provided operators
        A = ops[poi_op](len(cat_filtered_1), num_pois)
        B = ops[transport_op](transport_count, num_transport)
        C = ops[competitor_op](len(cat_filtered_2), num_competitors)

        # Evaluate expression like: A and B or not C
        try:
            if eval(logic_expr):
                survived_zones.append(zone_id)
        except Exception as e:
            print(f"Error evaluating logic expression: {e}")

    return zone_df[zone_df['zone_id'].isin(survived_zones)]


hard_10_test_cases = [
    (5, ">=", 4, ">=", "subway_entrance", 3, "<=", "A or B and not C", "Snack and Nonalcoholic Beverage Bars", "Snack and Nonalcoholic Beverage Bars", None, None),
    (6, ">=", 5, ">=", "bus_stop", 4, "<", "A or B and not C", "Full-Service Restaurants", "Full-Service Restaurants", None, None),
    (5, ">=", 4, ">=", "taxi", 3, "<=", "A or B and not C", "Beauty Salons", "Beauty Salons", None, None),
    (6, ">=", 5, ">=", "subway_entrance", 3, "<=", "A or B and not C", "Snack and Nonalcoholic Beverage Bars", "Snack and Nonalcoholic Beverage Bars", None, None),
    (4, ">=", 4, ">=", "bus_stop", 3, "<", "A or B and not C", "Drinking Places", "Drinking Places", None, None),
    (5, ">=", 4, ">=", "bus_stop", 3, "<=", "A or B and not C", "Snack and Nonalcoholic Beverage Bars", "Snack and Nonalcoholic Beverage Bars", None, None),
]
#FIXXX
all_valid_zones = set()

for i, (num_pois, poi_op, num_transport, transport_op, transport_type, num_competitors, competitor_op, logic_expr, sub_category1, sub_category2, top_category1, top_category2) in enumerate(hard_10_test_cases):
    obj_path = f"/Users/mihokoda/Desktop/CityLLM/test_results/hard/10/tc_hard_10_{i}/objective.csv"
    obj = pd.read_csv(obj_path)
    obj_zones = set(obj["zone_id"].unique())

    result_df = hard_10(
        num_pois=num_pois,
        poi_op=poi_op,
        num_transport=num_transport,
        transport_op=transport_op,
        transport_type=transport_type,
        num_competitors=num_competitors,
        competitor_op=competitor_op,
        logic_expr=logic_expr,
        sub_category1=sub_category1,
        sub_category2=sub_category2,
        top_category1=top_category1,
        top_category2=top_category2,
    )
    result_zones = set(result_df["zone_id"].unique())
    all_valid_zones.update(result_zones)

    print(f"Test case {i}:")
    print(f"  Objective zones: {len(obj_zones)}")
    print(f"  Found zones:     {len(result_zones)}")
    print(f"  Match:           {obj_zones == result_zones}")
    print()