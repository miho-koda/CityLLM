import os
import sys
from langchain_openai import ChatOpenAI  # OpenAI model
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from config import API_KEYS

from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages

from workflow.state import State, FileIOArgs, CodeWriterArgs, SearchWebArgs


# GeoPandas Documentation Reference
geo_docs = """
GeoPandas Documentation:

1. **Reading and Writing Files:**
- `geopandas.read_file(filepath)`: Reads a file and returns a GeoDataFrame.
- `GeoDataFrame.to_file(filename, driver="ESRI Shapefile")`: Writes a GeoDataFrame to a file.

2. **Data Structures:**
- `GeoDataFrame`: A tabular data structure that contains a collection of geometries and associated data.
- `GeoSeries`: A Series object designed to store shapely geometry objects.

3. **Geometric Operations:**
- `GeoSeries.buffer(distance)`: Returns a GeoSeries with buffered geometries.
- `GeoSeries.centroid`: Returns a GeoSeries of centroids for each geometry.
- `GeoSeries.convex_hull`: Returns the convex hull of each geometry.
- `GeoSeries.plot()`: Plots the GeoSeries geometries.

4. **Spatial Joins:**
- `geopandas.sjoin(left_df, right_df, how="inner", op="intersects")`: Spatial join between two GeoDataFrames.

5. **Coordinate Reference Systems (CRS):**
- `GeoDataFrame.set_crs(crs, allow_override=False)`: Sets the CRS for the GeoDataFrame.
- `GeoDataFrame.to_crs(crs)`: Transforms geometries to a new CRS.

6. **Aggregation and Dissolve:**
- `GeoDataFrame.dissolve(by=None, aggfunc="first")`: Aggregates geometries by a specified column.

7. **Plotting:**
- `GeoDataFrame.plot(column=None, cmap=None, legend=False)`: Plots the GeoDataFrame.

8. **Miscellaneous:**
- `geopandas.overlay(df1, df2, how="intersection")`: Performs spatial overlay between two GeoDataFrames.
- `geopandas.clip(gdf, mask, keep_geom_type=False)`: Clips points, lines, or polygon geometries to the mask extent.
"""

# Initialize LLM
llm = ChatOpenAI(model="gpt-4-turbo", api_key=API_KEYS["openai"])

def code_writer(state: State):
    user_message = state["messages"][-1]["content"]
    language = state["task_args"]["language"]
    requirements = state["task_args"]["requirements"]
    dep = state["dep_results"]
    # Prompt Formatting
    prompt = f"""
    You are a Python function generator specialized in geospatial data processing.
    Return only valid and executable Python code with no explanations.
    
    - Use **GeoPandas version 1.0.1** and **Geodatasets version 2024.8.0**.
    - Always use `'geodatasets.get_path()'` to access datasets.
    - Use the provided documentation below to generate responses.
    
    {geo_docs}
    
    Never generate responses outside this documentation.
    Create a Python function that **returns the final result instead of printing it**.

    Analyze the user request below and provide function code back:

    **Request:** "{user_message}" using {language} with the requirement {requirements} given the information {dep}
    """

    # Get Response from LLM
    response = llm.invoke(prompt).content

    # Remove code block formatting if present
    if response.startswith("```python"):
        response = response[9:]
    if response.endswith("```"):
        response = response[:-3]

    return {"code_results": response.strip()}  # Return as dictionary

'''
#TEST SCRIPT
import json

# ✅ Sample test cases
test_cases = [
    {"content": "Write a function to read a shapefile and return its centroid.",},
    {"content": "Create a function to buffer geometries in a GeoDataFrame by 10 units.",},
    {"content": "Write a function that performs a spatial join between two GeoDataFrames.",},
    {"content": "Create a function to plot a GeoDataFrame with a legend.",},
    {"content": "Write a function that converts a GeoDataFrame to a different coordinate reference system (CRS=4326).",},
]

# ✅ Function to run the test cases
def test_code_writer():
    for i, test in enumerate(test_cases, 1):
        print(f"\n📝 **Test Case {i}:** {test['content']}")

        # Mock state format expected by code_writer function
        state = {"messages": [{"role": "user", "content": test["content"]}]}

        # Call the code_writer function
        result = code_writer(state)

        # Extract generated code
        generated_code = result["code_results"]

        # Pretty-print the generated code
        print("🖥️ **Generated Code:**\n", generated_code)

        

# ✅ Run the test function
test_code_writer()
'''