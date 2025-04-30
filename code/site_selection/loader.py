import pandas as pd
import os
import glob
import gzip
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
import time
import functools
import traceback
def get_poi_spend_dataset():
    """
    Load the POI spend dataset for all years.
    """
    base_path = "/Users/mihokoda/Desktop/CityLLM/data/safegraph_dataset/Boston_POI_Spend_with_zones.csv" # use in local
    poi_spend_df = pd.read_csv(base_path)
    return poi_spend_df



def get_parking_dataset():
    """
    Load parking dataset for a specific state and optionally filter by city.
    
   
    Returns:
        pandas.DataFrame: Parking dataset, filtered by city if specified
    """
    base_path = "/Users/mihokoda/Desktop/CityLLM/data/safegraph_dataset/Massachusetts_Parking.csv" # use in local
    parking_df = pd.read_csv(base_path)
    return parking_df

