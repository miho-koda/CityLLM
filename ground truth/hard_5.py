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


def hard_5(
    num_competitors: int,
    num_transport: int,
    transport_type: str,
    population: int,
    num_zones: int,
    logic_expr: str,
    sub_category: str = None,
    top_category: str = None,
):
    poi_df = get_poi_spend_dataset()
    zone_df = create_zone(poi_df)

    survived_zones = []
    transport_dict = get_transport_pois_in_zone(zone_df, transport_type)
    for zone_id in zone_df["zone_id"]:
        zone_pois = filter_df_based_on_zone(poi_df, zone_id)

        if sub_category:
            relevant_pois = filter_pois_by_sub_category(zone_pois, sub_category)
        elif top_category:
            relevant_pois = filter_pois_by_top_category(zone_pois, top_category)
        else:
            relevant_pois = zone_pois

        # A: number of competitors
        A = len(relevant_pois) < num_competitors
        B = len(transport_dict.get(zone_id, [])) >= num_transport

        neighbors = get_neighbor_zones(zone_df, zone_id, num_zones)
        neighbor_ids = [zone_id] + neighbors
        pop = sum(get_population(zid, zone_df) for zid in neighbor_ids)
        C = pop >= population

        # Apply logic expression
        expr = logic_expr.replace("AND", "and").replace("OR", "or").replace("NOT", "not")
        try:
            if eval(expr, {"A": A, "B": B, "C": C}):
                survived_zones.append(zone_id)
        except Exception as e:
            print(f"Invalid logic for zone {zone_id}: {e}")
            continue

    return zone_df[zone_df["zone_id"].isin(survived_zones)]

hard_5_test_cases = [
    # 1. Wellness café
    (3, 5, "subway_entrance", 15000, 2, "(A AND B AND C)", "Snack and Nonalcoholic Beverage Bars", None),

    # 2. Boutique legal office
    (4, 4, "bus_stop", 18000, 3, "(A AND B AND C)", None, "Legal Services"),

    # 3. Tutoring center
    (2, 3, "station", 12000, 2, "(A AND B AND C)", "Exam Preparation and Tutoring", None),

    # 4. Restaurant incubator
    (5, 6, "bus_stop", 20000, 2, "(A AND B AND C)", None, "Restaurants and Other Eating Places"),

    # 5. Co-working hub
    (3, 4, "subway_entrance", 14000, 2, "(A AND B AND C)", None, "Offices of Real Estate Agents and Brokers"),

    # 6. Day spa
    (4, 5, "taxi", 16000, 3, "(A AND B AND C)", "Beauty Salons", None),
]

all_valid_zones = set()

for i, (num_competitors, num_transport, transport_type, population, num_zones, logic_expr, sub_cat, top_cat) in enumerate(hard_5_test_cases):
    obj_path = f"/Users/mihokoda/Desktop/CityLLM/test_results/hard/5/tc_hard_5_{i}/objective.csv"
    obj = pd.read_csv(obj_path)
    obj_zones = set(obj["zone_id"].unique())

    result_df = hard_5(
        num_competitors=num_competitors,
        num_transport=num_transport,
        transport_type=transport_type,
        population=population,
        num_zones=num_zones,
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
