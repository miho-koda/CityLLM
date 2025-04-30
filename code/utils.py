import re
import string
import pandas as pd
import geopandas as gpd
from typing import Dict, Any, Tuple, List

def normalize_string(s: str) -> str:
    """Normalize a string by removing articles and punctuation"""
    def remove_articles(text: str) -> str:
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text: str) -> str:
        return " ".join(text.split())

    def remove_punc(text: str) -> str:
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text: str) -> str:
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))

def parse_args_string(args_str: str) -> Dict[str, Any]:
    """Parse a string of arguments into a dictionary"""
    args = {}
    if not args_str:
        return args
        
    parts = args_str.split(',')
    for part in parts:
        if '=' in part:
            key, value = part.split('=', 1)
            args[key.strip()] = value.strip()
            
    return args

def validate_coordinates(lat: float, lng: float) -> bool:
    """Validate latitude and longitude coordinates"""
    try:
        lat = float(lat)
        lng = float(lng)
        return -90 <= lat <= 90 and -180 <= lng <= 180
    except:
        return False

def calculate_distance(lat1: float, lng1: float, lat2: float, lng2: float) -> float:
    """Calculate distance between two points in kilometers"""
    from math import sin, cos, sqrt, atan2, radians
    
    R = 6371.0  # Earth's radius in kilometers

    lat1, lng1 = radians(float(lat1)), radians(float(lng1))
    lat2, lng2 = radians(float(lat2)), radians(float(lng2))

    dlng = lng2 - lng1
    dlat = lat2 - lat1

    a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlng / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))

    return R * c

def validate_year(year: int) -> bool:
    """Validate that a year is within the valid range (2019-2024)"""
    try:
        year = int(year)
        return 2019 <= year <= 2024
    except:
        return False

def validate_category(category: str, valid_categories: List[str]) -> bool:
    """Validate that a category exists in the valid categories list"""
    return category in valid_categories

def format_currency(amount: float) -> str:
    """Format a number as currency"""
    return f"${amount:,.2f}"

def format_percentage(value: float) -> str:
    """Format a number as percentage"""
    return f"{value:.1f}%"

def calculate_growth_rate(start_value: float, end_value: float) -> float:
    """Calculate growth rate between two values"""
    if start_value == 0:
        return float('inf')
    return ((end_value - start_value) / start_value) * 100

def extract_zone_metrics(zone_data: Dict[str, Any]) -> Dict[str, Any]:
    """Extract key metrics from zone data"""
    metrics = {
        'population': zone_data.get('population', 0),
        'num_pois': zone_data.get('num_pois', 0),
        'num_parking': zone_data.get('num_parking', 0),
        'largest_parking_area': zone_data.get('largest_parking_area', 0),
        'transport_pois': zone_data.get('transport_pois', {}),
        'spending_metrics': zone_data.get('spending_metrics', {})
    }
    return metrics

def format_zone_analysis(
    zone_id: int,
    metrics: Dict[str, Any],
    recommendations: List[str],
    risks: List[str]
) -> str:
    """Format zone analysis results"""
    analysis = f"""
Zone {zone_id} Analysis:

Key Metrics:
- Population: {metrics['population']:,}
- Number of POIs: {metrics['num_pois']}
- Parking Lots: {metrics['num_parking']}
- Largest Parking Area: {metrics['largest_parking_area']:,.0f} sq meters

Transport Access:
"""
    for poi_type, count in metrics['transport_pois'].items():
        analysis += f"- {poi_type}: {count}\n"
    
    analysis += "\nSpending Metrics:\n"
    for metric, value in metrics['spending_metrics'].items():
        analysis += f"- {metric}: {value}\n"
    
    analysis += "\nRecommendations:\n"
    for i, rec in enumerate(recommendations, 1):
        analysis += f"{i}. {rec}\n"
    
    analysis += "\nRisk Factors:\n"
    for i, risk in enumerate(risks, 1):
        analysis += f"{i}. {risk}\n"
        
    return analysis

def load_dataframe(file_path: str, file_type: str = 'csv') -> pd.DataFrame:
    """Load a dataframe from file"""
    if file_type == 'csv':
        return pd.read_csv(file_path)
    elif file_type == 'geojson':
        return gpd.read_file(file_path)
    else:
        raise ValueError(f"Unsupported file type: {file_type}")

def save_dataframe(df: pd.DataFrame, file_path: str, file_type: str = 'csv') -> None:
    """Save a dataframe to file"""
    if file_type == 'csv':
        df.to_csv(file_path, index=False)
    elif file_type == 'geojson':
        if isinstance(df, gpd.GeoDataFrame):
            df.to_file(file_path, driver='GeoJSON')
        else:
            raise ValueError("DataFrame must be GeoDataFrame to save as GeoJSON")
    else:
        raise ValueError(f"Unsupported file type: {file_type}") 