import unittest
import json
import os
from agents.task_executor import main

class TestSimpleTaskExecutor(unittest.TestCase):

    def setUp(self):
        """Create a simple JSON file for testing."""
        self.test_json_file = "test_simple_tasks.json"

        # simple_tasks = [
        #     {
        #         "task_type": "file_io",
        #         "id": 1,
        #         "dep": [],
        #         "args": {
        #             "operation": "write",
        #             "file_path": "/Users/mihokoda/Desktop/CityLLM/code files/test data/data.txt", 
        #             "content": "Simple Test!"
        #         }
        #     },
        #     {
        #         "task_type": "file_io",
        #         "id": 2,
        #         "dep": [1],
        #         "args": {
        #             "operation": "read",
        #             "file_path": "/Users/mihokoda/Desktop/CityLLM/code files/test data/data.txt"
        #         }
        #     }
        # ]



        simple_tasks = [
            {
                "task_type": "search_web",
                "id": 0,
                "dep": [],
                "args": {
                    "query": "what is the latitudes, and longtitudes of the public transport stations near Cambridge Massachusetts",
                    "focus_area": "transportation"
                }
            },
            {
                "task_type": "llm_extraction",
                "id": 1,
                "dep": [
                    0
                ],
                "args": {
                    "input": "Search results from task 0",
                    "instructions": "Extract the names, latitudes, and longitudes of public transport stations"
                }
            },
            {
                "task_type": "code_writer",
                "id": 2,
                "dep": [
                    1
                ],
                "args": {
                    "language": "javascript",
                    "requirements": "Generate a map visualizing locations of public transport stations",
                    "context": "Data points with names and coordinates from task 1"
                }
            },
            {
                "task_type": "file_io",
                "id": 3,
                "dep": [
                    2
                ],
                "args": {
                    "operation": "write",
                    "file_path": "output/map_visualization.html",
                    "content": "Generated map code from task 2"
                }
            }
        ]



        with open(self.test_json_file, "w") as f:
            json.dump(simple_tasks, f)

    def tearDown(self):
        """Clean up test files."""
        if os.path.exists(self.test_json_file):
            os.remove(self.test_json_file)
        if os.path.exists("simple_test.txt"):
            os.remove("simple_test.txt")

    def test_main_execution(self):
        """Test that main() runs successfully with the simple task JSON file."""
        try:
            main(self.test_json_file)
            self.assertTrue(True)  # If no exceptions, the test passes
        except Exception as e:
            self.fail(f"Test failed due to exception: {e}")

if __name__ == "__main__":
    unittest.main()
