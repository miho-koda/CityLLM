# benchmark_functions.py
import geopandas
from geodatasets import get_path

def q1_benchmark():
    """
    Load the New York Boroughs dataset (nybb) and return the first 5 rows.
    """
    path_to_data = get_path("nybb")
    gdf = geopandas.read_file(path_to_data)
    # Instead of printing, return the head() so we can compare later.
    return gdf.head()

def q2_benchmark():
    """
    Calculate the area of each borough in the New York Boroughs dataset (nybb)
    """
    path_to_data = get_path("nybb")
    gdf = geopandas.read_file(path_to_data)
    # Calculate area (using the geometry's area; note: real area calculations may need proper CRS)
    gdf["area"] = gdf.geometry.area
    return gdf["area"]

def q3_benchmark():
    """
    Create an interactive map of the New York Boroughs dataset (nybb), color-coded by area.
    """
    path_to_data = get_path("nybb")
    gdf = geopandas.read_file(path_to_data)
    gdf["area"] = gdf.area  
    map_obj = gdf.explore("area", legend=False)
    return map_obj

def q4_benchmark():
    """
    Find and plot the convex hull of each borough in the New York Boroughs dataset (nybb).
    """
    path_to_data = get_path("nybb")
    gdf = geopandas.read_file(path_to_data)

    gdf["convex_hull"] = gdf.geometry.convex_hull
    ax = gdf["convex_hull"].plot(edgecolor="black", facecolor="lightblue", alpha=0.5)
    gdf.plot(ax=ax, edgecolor="black", facecolor="none", linewidth=1)
    return ax


def q5_benchmark():
    """
    "Dissolve the New York Boroughs dataset (nybb) by a common attribute and plot the result."
    """

    path_to_data = get_path("nybb")
    gdf = geopandas.read_file(path_to_data)

    dissolved_gdf = gdf.dissolve(by="BoroCode")
    ax = dissolved_gdf.plot(edgecolor="black", facecolor="lightgreen", alpha=0.6)
    return ax

def q6_benchmark():
    """
    # 6. "Create a plot of the New York Boroughs dataset (nybb)'s geometries and overlay the centroids for each feature. Color-code the centroids based on the area of their corresponding geometries."
    """

    path_to_data = get_path("nybb")
    gdf = geopandas.read_file(path_to_data)

    gdf["centroid"] = gdf.centroid
    gdf["area"] = gdf.area
    ax = gdf.plot(column="area", legend=True)
    gdf.set_geometry("centroid").plot(ax=ax, color="black")

    return ax

def q7_benchmark():
    """
    # 7. Load the New York Boroughs dataset (nybb).Compute the convex hull of all boroughs. Extract the centroid of the convex hull. And plot the original boroughs and overlay the convex hull and centroid.
    """
    path_to_data = get_path("nybb")
    gdf = geopandas.read_file(path_to_data)    

    convex_hull = gdf.unary_union.convex_hull
    centroid = convex_hull.centroid
    
    convex_hull_gdf = geopandas.GeoDataFrame(geometry=[convex_hull], crs=gdf.crs)
    centroid_gdf = geopandas.GeoDataFrame(geometry=[centroid], crs=gdf.crs)
    
    ax = gdf.plot(edgecolor="black", alpha=0.3)
    convex_hull_gdf.plot(ax=ax, color="blue", alpha=0.5, edgecolor="black")
    centroid_gdf.plot(ax=ax, color="red", markersize=100)
    
    return ax

def q8_benchmark():
    """
    #8. Load the New York Boroughs dataset (nybb). Compute the area of each borough. Identify the borough with the largest and smallest area. Return their names as a tuple: (largest_borough, smallest_borough).
    """
    path_to_data = get_path("nybb")
    gdf = geopandas.read_file(path_to_data)   

    gdf["area"] = gdf.geometry.area
    largest_borough = gdf.loc[gdf["area"].idxmax(), "BoroName"]
    smallest_borough = gdf.loc[gdf["area"].idxmin(), "BoroName"]
    return (largest_borough, smallest_borough)

def q9_benchmark():
    """
    #9. Compute the average distance between New York Boroughs dataset (nybb)'s borough centroids by calculating the average pairwise distance between all centroids. Return the computed average distance.
    """
    from itertools import combinations
    import geopandas as gpd
    from geodatasets import get_path

    path_to_data = get_path("nybb")
    gdf = geopandas.read_file(path_to_data)

    # Compute centroids
    gdf["centroid"] = gdf.geometry.centroid

    # Compute pairwise distances
    distances = []
    for (i, j) in combinations(gdf.index, 2):  # Generate borough index pairs
        centroid_i = gdf.loc[i, "centroid"]  # Get centroid of borough i
        centroid_j = gdf.loc[j, "centroid"]  # Get centroid of borough j
        dist = centroid_i.distance(centroid_j)  # Compute distance between centroids
        distances.append(dist)  # Store the computed distance


    # Compute average distance
    average_distance = sum(distances) / len(distances)

    # Return the result
    return average_distance

def q10_benchmark():
    """
    #10. Find the two boroughs of the New York Boroughs dataset (nybb) that are closest to each other by compute the minimum distance between each pair of boroughs. Return their names as a tuple.
    """
    from itertools import combinations
    import geopandas as gpd
    from geodatasets import get_path

    path_to_data = get_path("nybb")
    gdf = geopandas.read_file(path_to_data)

    # Compute distances between all pairs
    min_distance = float("inf")
    closest_pair = None

    for i, j in combinations(gdf.index, 2):
        dist = gdf.loc[i, "geometry"].distance(gdf.loc[j, "geometry"])
        if dist < min_distance:
            min_distance = dist
            closest_pair = (gdf.loc[i, "BoroName"], gdf.loc[j, "BoroName"])

    # Return the closest pair
    return closest_pair

def q11_benchmark():
    """
    Plot the convex hulls of the New York Boroughs dataset (nybb) geometries, 
    then apply a 10,000-foot buffer to both the boroughs and their centroids. 
    Display the buffered regions with 50% transparency, overlay the centroids in red with the same transparenc,  and overlay the original boundaries for reference."
    """

    import geopandas as gpd
    from geodatasets import get_path

    path_to_data = get_path("nybb")
    gdf = geopandas.read_file(path_to_data)

    gdf["centroid"] = gdf.centroid
    gdf["convex_hull"] = gdf.convex_hull
    gdf["buffered"] = gdf.buffer(10000)
    gdf["buffered_centroid"] = gdf["centroid"].buffer(10000)
    gdf["boundary"] = gdf.boundary


    ax = gdf["buffered"].plot(alpha=0.5)  # Plot buffered boroughs with transparency
    gdf["buffered_centroid"].plot(ax=ax, color="red", alpha=0.5)  # Overlay buffered centroids in red
    gdf["boundary"].plot(ax=ax, color="white", linewidth=0.5)  # Overlay original boundaries in white

    return ax

def q12_benchmark():
    """
    #12. Find the Pair of Boroughs That Have the Most Nearly Identical Shapes (Geometrically Similar) in the New York Boroughs dataset (nybb). Return a tuple: (borough1, borough2, hausdorff_distance)
    """
    import itertools
    import geopandas as gpd
    from geodatasets import get_path

    path_to_data = get_path("nybb")
    gdf = geopandas.read_file(path_to_data)


    # Compute pairwise Hausdorff distances
    min_distance = float("inf")
    most_similar_pair = None

    for i, j in itertools.combinations(gdf.index, 2):
        dist = gdf.loc[i, "geometry"].hausdorff_distance(gdf.loc[j, "geometry"])
        if dist < min_distance:
            min_distance = dist
            most_similar_pair = (gdf.loc[i, "BoroName"], gdf.loc[j, "BoroName"], dist)

    return most_similar_pair

def q13_benchmark():
    """
    #13. Compute the Borough with the Most Complex Boundary in the New York Boroughs dataset (nybb).  Return a tuple: (borough_name, num_boundary_points).
    """

    import geopandas as gpd
    from geodatasets import get_path

    # Load dataset
    gdf = gpd.read_file(get_path("nybb"))

    # Compute boundary geometries
    gdf["boundary"] = gdf.boundary

    # Extract unique boundary points
    gdf["unique_boundary_points"] = gdf.boundary.extract_unique_points().count_coordinates()

    # Find borough with the most complex boundary
    most_complex_borough = gdf.loc[gdf["unique_boundary_points"].idxmax(), "BoroName"]
    max_boundary_points = gdf["unique_boundary_points"].max()

    # Return result
    return (most_complex_borough, max_boundary_points)


def q14_benchmark():
    """
    14. We load the New York Boroughs dataset, compute the Hausdorff distance between each pair of boroughs to determine which pair of boroughs has the most similar shapes (i.e. the smallest Hausdorff distance), return a tuple with the borough names and the distance, and plot the geometries for visual confirmation.
    """

    import geopandas as gpd
    import matplotlib.pyplot as plt
    import itertools

    # Load the New York Boroughs dataset (nybb) from the GeoPandas sample data
    gdf = gpd.read_file(get_path("nybb"))

    # Initialize variables to store the minimum distance and the corresponding pair of boroughs
    min_distance = float("inf")
    most_similar_pair = None

    # Compute pairwise Hausdorff distances between borough geometries
    for i, j in itertools.combinations(gdf.index, 2):
        geom_i = gdf.loc[i, "geometry"]
        geom_j = gdf.loc[j, "geometry"]
        distance = geom_i.hausdorff_distance(geom_j)
        if distance < min_distance:
            min_distance = distance
            most_similar_pair = (gdf.loc[i, "BoroName"], gdf.loc[j, "BoroName"], distance)

    # Print the result
    print("Most similar pair of boroughs (by Hausdorff distance):")
    print(most_similar_pair)

    # Extract the two boroughs for plotting
    borough1_name, borough2_name, _ = most_similar_pair
    borough1 = gdf[gdf["BoroName"] == borough1_name]
    borough2 = gdf[gdf["BoroName"] == borough2_name]

    # Plot the two geometries for visual confirmation
    fig, ax = plt.subplots(figsize=(8, 8))
    borough1.boundary.plot(ax=ax, edgecolor="red", linewidth=2, label=borough1_name)
    borough2.boundary.plot(ax=ax, edgecolor="blue", linewidth=2, label=borough2_name)
    ax.set_title("Pair of Boroughs with the Most Similar Shapes")
    ax.legend()
    
    return ax

def q15_benchmark():
    """
    For each borough, count the number of exterior coordinates in its geometry, simplify the geometry with a specified tolerance, and count the simplified coordinates. Then compute the percentage reduction for each borough and return a DataFrame containing the borough name, original and simplified coordinate counts, and the reduction percentage; also, identify and plot the borough that shows the maximum reduction.
    """

    import geopandas as gpd
    import pandas as pd
    import matplotlib.pyplot as plt

    def count_coordinates(geom):
        """
        Count the number of exterior coordinates.
        Handles both Polygon and MultiPolygon geometries.
        """
        if geom.geom_type == 'Polygon':
            return len(geom.exterior.coords)
        elif geom.geom_type == 'MultiPolygon':
            count = 0
            for poly in geom.geoms:
                count += len(poly.exterior.coords)
            return count
        else:
            return 0

    # Load the dataset
    nybb = gpd.read_file(get_path("nybb"))

    borough_names = []
    original_counts = []
    simplified_counts = []
    percentage_reductions = []
    tol = 0.001  # Simplification tolerance

    for idx, row in nybb.iterrows():
        geom = row.geometry
        orig_count = count_coordinates(geom)
        simplified_geom = geom.simplify(tol, preserve_topology=True)
        simp_count = count_coordinates(simplified_geom)
        reduction = ((orig_count - simp_count) / orig_count * 100) if orig_count > 0 else 0
        
        borough_names.append(row.BoroName)
        original_counts.append(orig_count)
        simplified_counts.append(simp_count)
        percentage_reductions.append(reduction)

    df_results = pd.DataFrame({
        "Borough Name": borough_names,
        "Original Coordinate Count": original_counts,
        "Simplified Coordinate Count": simplified_counts,
        "Percentage Reduction": percentage_reductions
    })

    print("Coordinate Reduction Analysis:")
    print(df_results)

    # Identify the borough with maximum reduction percentage
    max_reduction_idx = df_results["Percentage Reduction"].idxmax()
    max_reduction_boro = df_results.loc[max_reduction_idx, "Borough Name"]
    print(f"\nBorough with maximum reduction: {max_reduction_boro}")

    # Plot the original vs. simplified geometry for the borough with maximum reduction
    best_row = nybb[nybb["BoroName"] == max_reduction_boro].iloc[0]
    orig_geom = best_row.geometry
    simplified_geom = orig_geom.simplify(tol, preserve_topology=True)

    fig, ax = plt.subplots(figsize=(8, 8))
    gpd.GeoSeries(orig_geom).boundary.plot(ax=ax, color='black', linewidth=2, label='Original Geometry')
    gpd.GeoSeries(simplified_geom).boundary.plot(ax=ax, color='red', linestyle='--', linewidth=2, label='Simplified Geometry')
    ax.set_title(f"{max_reduction_boro}: Original vs. Simplified Geometry")
    ax.legend()
    plt.show()
    return ax

