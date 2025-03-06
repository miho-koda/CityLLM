import unittest
import json
import os
from agents.router import router
from agents.task_executor import main

class TestTaskExecutor(unittest.TestCase):
    def setUp(self):
        """Set up test inputs and expected outputs."""
        self.test_inputs = [
            "Search for public transport stops in San Francisco, use a Python script to compute a 500-meter buffer around each stop using GeoPandas, and save the resulting accessibility zones as a GeoJSON file.",
            "Compare land use data from two different years (stored in separate shapefiles) and identify areas where land use has changed.",
            "Search for public transport stations near Cambridge, Massachusetts, extract their names and coordinates, and generate a map visualizing their locations.",
            "Retrieve satellite land cover data for New York City from 2000 and 2020, process the data using a Python script to compute changes in green space coverage, and save the results as a CSV file."
        ]
    


        # Expected outputs for each test input
        self.expected_outputs = [
            "output/map_visualization.html",  # Expected output file for Test Case 1
            "land_use_changes.shp"            # Expected output file for Test Case 2
        ]

    def tearDown(self):
        """Clean up test files."""
        if os.path.exists(self.test_json_file):
            os.remove(self.test_json_file)
        if os.path.exists("simple_test.txt"):
            os.remove("simple_test.txt")

    def test_router_and_task_executor(self):
        """Test the router and task_executor to ensure they work together correctly."""
        self.test_json_file = "test_simple_tasks.json"
        for i, user_input in enumerate(self.test_inputs):
            print(f"\n**Test Case {i + 1}:** {user_input}")


            # Step 1: Call the router function to decompose the user input into tasks
            tasks = router(user_input)

            print("🔍 **Decomposed Tasks:**")
            print(json.dumps(tasks, indent=4))

            # Step 2: Call the task_executor to execute the tasks
            try:
                main(tasks)  # Directly pass the tasks list to execute_tasks
                print("✅ **Task Execution Completed Successfully**")
            except Exception as e:
                self.fail(f"Task execution failed due to exception: {e}")

            # Step 3: Validate the actual output
            expected_output = self.expected_outputs[i]
            self.assertTrue(os.path.exists(expected_output), f"Test case {i + 1} failed: Expected output file '{expected_output}' was not created.")

            # Additional validation (if applicable)
            if expected_output == "output/map_visualization.html":
                # Check if the map visualization file is not empty
                with open(expected_output, "r") as f:
                    content = f.read()
                    self.assertTrue(content.strip(), "Map visualization file is empty.")
            elif expected_output == "land_use_changes.shp":
                # Check if the shapefile exists and is not empty
                self.assertTrue(os.path.getsize(expected_output) > 0, "Land use changes file is empty.")

if __name__ == "__main__":
    unittest.main()