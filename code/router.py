from langchain_openai import ChatOpenAI  # OpenAI model
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from config import API_KEYS
import json

llm = ChatOpenAI(model="gpt-4-turbo", api_key=API_KEYS["openai"])   
from langchain.schema import HumanMessage

import prompt
dataframe_documentation = prompt.dataframe_documentation
in_house_functions_documentation = prompt.in_house_functions_documentation

def router(user_message):
    prompt = f"""
    You are a Python programmer with access to the following functions as tools. Each tool has a specific format, description, and example. You can use the provided functions to generate code.

    Here are the available functions:
    {in_house_functions_documentation}

    Here is the documentation for the DataFrames you will be working with:
    {dataframe_documentation}

    Write Python code to answer the following query:
    {user_message}

    Only use the provided functions. Output only valid Python code, not markdown.
    The final function you generate should always end with a return statement that returns a filtered zone_df. This zone_df must contain only the zones that satisfy the user's request. All relevant zone-level information should be included in this returned DataFrame.
"""

    try:
        response = llm([HumanMessage(content=prompt)])
        code = response.content
        # Clean up code block markers if present
        if code.startswith("```python"):
            code = code.split("```python")[1]
        if code.startswith("```"):
            code = code.split("```")[1]
        if code.endswith("```"):
            code = code[:-3]
        return code.strip()
    except Exception as e:
        print(f"Error occurred: {str(e)}")
        return None

print(router("I want to look at zones in new york city where the raw total spend at year 2022 is ≥ 40000"))


# 
# def router(user_message):
#     prompt = f"""    
#     You are an advanced task planner responsible for breaking down a high-level user request into a structured, interdependent workflow using a minimal number of AI agents while ensuring **maximal efficiency** Do not output anything else. Only return the JSON file.

#     ### **User Request**
#     "{user_message}"

#     ### **Your Task**
#     Analyze the request and decompose it into a **logical sequence of interdependent tasks**, ensuring that:
#     - **Dependencies are correctly defined** (a task may depend on the output of multiple tasks).
#     - **The fewest number of AI agents are used** while maintaining **optimal performance**.
#     - **Tasks are ordered efficiently** to avoid redundant computation.
#     - **Minimize multiple calls to each tasks by writing a single, comprehensive script that can handle multiple operations at once.**

#     ### **Available Task Types**
#     Your decomposition should use the following structured task types:

#     #### **1. filter_by_city**
#     - Used to filter data by city name.
#     - **Arguments:**
#     - `city`: String representing the full city name (e.g., "New York", "Houston", "Honolulu").

#     #### **2. filter_by_region**
#     - Used to filter data by state abbreviation.
#     - **Arguments:**
#     - `region`: String representing the state abbreviation (e.g., "MA", "TX", "NC").

#     #### **3. filter_by_demand**
#     - Used to filter zones by demand percentile.
#     - **Arguments:**
#     - `top_demand_percent`: Integer representing the top percentile of demand to keep (e.g., 20 to show only zones with top 20% demand of the city).

#     #### **4. filter_by_rating**
#     - Used to filter zones by their average rating.
#     - **Arguments:**
#     - `min_avg_rating`: Float value (1-5) representing the minimum rating threshold (e.g., 4.3 to show only zones with average rating above 4.3).

#     #### **5. filter_by_competition**
#     - Used to filter zones by competition level. User must atleast specifiy one of top_category or sub_category, or both. 
#     - **Arguments:**
#     - `max_competitors`: Integer representing the maximum number of allowed competitors of the same category per zone.
#     - `top_category` = None: String representing the broad category (Optional)
#     - `sub_category` = None: String representing the smaller category (Optional)

#     #### **6. filter_by_transit_distance**
#     - Used to filter locations by maximum distance to transit.
#     - **Arguments:**
#     - `place`: String representing the full city or region name (e.g., "New York", "Houston", "Honolulu", "MA", "NC").
#     - `max_transit_distance`: Integer representing the maximum allowed distance in meters from the location to either a bus stop or a station.

#     #### **7. filter_by_road_distance**
#     - Used to filter locations by maximum distance to major roads.
#     - **Arguments:**
#     - `place`: String representing the full city or region name (e.g., "New York", "Houston", "Honolulu", "MA", "NC").
#     - `max_road_distance`: Integer representing the maximum allowed distance in meters from the location to a major road.

#     #### **8. filter_by_parking_spaces**
#     - Used to filter locations by minimum number of available parking spaces.
#     - **Arguments:**
#     - `min_parking_spaces`: Integer representing the minimum number of parking spaces required.

#     #### **9. filter_by_parking_area**
#     - Used to filter locations by minimum parking area.
#     - **Arguments:**
#     - `min_parking_area_m2`: Integer representing the minimum parking area in square meters.

#     #### **10. code_writer**
#     - Used to compute derived values or custom metrics when no direct filter exists.
#     - Example use cases: calculating density, spend-per-customer ratios, competition per capita, average visit duration, etc.
#     - **dataframe Documentation:**
#     {dataframe_documentation}
#     - **Arguments:**
#     - `language`: Always `"python"`
#     - `requirements`: Describe what the code needs to compute
#     - `context`: What inputs are available (e.g., from previous tasks or tools)
#     - `input_keys`: List of dataframes used as inputs. Can only be ["spending_df", "poi_df", "parking_df", "zone_df"] or a subset of these.
#     - `output_key`: Name of the dataframe returned after processing. Can only be "spending_df", "poi_df", "parking_df", or "zone_df". Optional if no dataframe is modified.

#     ---

#     ### **Task Dependencies & Efficiency Rules**
#     - **Minimize redundant `code_writer` calls**: Instead of multiple code-writing steps, generate a **single Python script** that:
#     - Reads all required files.
#     - Processes data as needed.
#     - Outputs all final results.
#     - **Tasks should be dependent only when necessary.** Use the minimum number of dependencies to maximize efficiency.
#     - **A task can have multiple dependencies** (e.g., the output of two different tasks can be combined as input for another task).
#     - **Ensure outputs are reusable** to avoid unnecessary recomputation.

#     ---

#     ### **Example JSON Output**
#     Return a structured JSON list where each task has:
#     1. A descriptive `"task_type"` (one of the above).
#     2. A unique integer `"id"` (e.g., `0, 1, 2, ...`).
#     3. A `"dep"` array specifying which tasks must complete before this one.
#     4. An `"args"` dictionary with task-specific arguments.

#     **
#     Example 1 user request: "Find zones in Boston where the **parking density** (parking spaces per km²) is more than **1.5× the city average**, and the zone is within top 20% demand."


#     ```json
#     [
#         {{
#             "task_type": "filter_by_city",
#             "id": 0,
#             "dep": [],
#             "args": {{
#                 "city": "Boston"
#             }}
#         }},
#         {{
#             "task_type": "filter_by_demand",
#             "id": 1,
#             "dep": [0],
#             "args": {{
#                 "percentile": 20
#             }}
#         }},
#         {{
#             "task_type": "code_writer",
#             "id": 2,
#             "dep": [1],
#             "args": {{
#                 "language": "python",
#                 "requirements": "Calculate parking_density = parking_spaces / zone_area for each zone, and filter zones where this is greater than 1.5× the city average.",
#                 "context": "Use output from Task 1: filtered POIs and parking lot data with zone IDs and WKT_AREA_SQ_METERS.",
#                 "input_keys": ["poi_df", "parking_df", "zone_df"],
#                 "output_key": "zone_df"
#             }}
#         }}
# ]
#     ```
#     **Example 2: "Find me retail zones in Boston with top 15% demand, less than 10 competitors, transit access within 300m, major road within 400m, at least 250 parking spaces."**
    
#     ```json
#     [
#         {{
#             "task_type": "filter_by_city",
#             "id": 0,
#             "dep": [],
#             "args": {{
#                 "city": "Boston"
#             }}
#         }},
#         {{
#             "task_type": "filter_by_demand",
#             "id": 1,
#             "dep": [0],
#             "args": {{
#                 "percentile": 15
#             }}
#         }},
#         {{
#             "task_type": "filter_by_competition",
#             "id": 2,
#             "dep": [1],
#             "args": {{
#                 "max_competitors": 10
#             }}
#         }},
#         {{
#             "task_type": "filter_by_transit_distance",
#             "id": 3,
#             "dep": [2],
#             "args": {{
#                 "place": "Boston",
#                 "max_transit_distance": 300
#             }}
#         }},
#         {{
#             "task_type": "filter_by_road_distance",
#             "id": 4,
#             "dep": [3],
#             "args": {{ 
#                 "place": "Boston",
#                 "max_road_distance": 400
#             }}
#         }},
#         {{
#             "task_type": "filter_by_parking_spaces",
#             "id": 5,
#             "dep": [4],
#             "args": {{
#                 "min_parking_spaces": 250
#             }}
#         }}
#     ]
#     ```

#     """

#     # Invoke LLM to generate structured tasks
#     tasks_json = llm.invoke(prompt).content
#     if tasks_json.startswith("```json"):
#         tasks_json = tasks_json[7:]  # Remove the first 7 characters (` ```json `)
#     if tasks_json.endswith("```"):
#         tasks_json = tasks_json[:-3]
        
#     # Attempt to parse the response as JSON
#     try:
#         structured_tasks = json.loads(tasks_json)
#         if isinstance(structured_tasks, list):  
#             return {"tasks": structured_tasks}
#     except json.JSONDecodeError:
#         pass  # If LLM output is invalid, default to a fallback task


