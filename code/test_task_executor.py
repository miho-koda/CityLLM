import unittest
import json
import os
import pandas as pd
import datetime
from code.router import router
from code.task_executor import main, execute_tasks

class TestTaskExecutor(unittest.TestCase):
    def setUp(self):
        """Set up test inputs and expected outputs."""
        self.test_inputs = [
            "I’m thinking about opening a small shop in Georgetown — category is Clothing Stores, sub-category Women’s Clothing Stores. Just wanna make sure the demand is solid, like top 25% or something. Can you check that for me?",
            "Yo, looking at Bridgewater for a Shoe Stores as the sub category. I don’t want to deal with a ton of other competitors tho… so maybe keep it under like 30 nearby?",
            "Thinking about a New Car Dealers spot in MT — anywhere is fine really, but I’d love if it had a big parking area, maybe 700 square meters or more. Is that doable?",
            "Hey, I’m trying to put a Meat Markets store (under Specialty Food Stores) in Florence, KY. Could you just make sure it’s near a main road? Something close-ish, like 100 meters max would be ideal.",
            "Just exploring ideas. What would be some good zones in Island Lake, IL for a Restaurants and Other Eating Places, more specifically Full-Service Restaurants? Just show me areas with top 20% demand.",
            "I was in Clinton, MO last week and thought it might be cute for a Personal Care Service, maybe a nail salon. Could you just find a zone with less than 50 competitors? And if it’s got decent ratings? Like over 4.2 would be perfect.",
            "Any parts of MO where it makes sense to open something under Health and Personal Care Stores — especially Optical Goods Stores? I’m not picky, but maybe keep it to places with decent demand (top 30%).",
            "I'm looking for potential locations in Boston with at least 5 parking spaces and a minimum parking area of 250 square meters. Where should I build?",
            "I want to open a business somewhere in Texas. Show me zones in the top 20% of demand across the state.",
            "Show me places in Texas where there are no more than 10 competitors in the same category. I want to avoid saturated markets.",
            "I’m looking at Chicago, but only if the location is within 300 meters of a transit stop. Accessibility is key for my business.",
            "Can you show me spots in Florida that are no more than 500 meters away from a major road? I need good car access.",
            "I want to open in Los Angeles, but only if there’s a parking area of at least 2500 square meters nearby. This will be a large-scale project.",
            "Find me zones in New York State with parking lots of at least 200 square meters. I'm planning for moderate foot traffic and car access.",
            "Can you find me highly-rated zones in San Francisco where the average rating is at least 4.5? I'm thinking about opening Mueseums.",
            "Find me retail zones in Seattle, WA that are in the top 20% demand.",
            "Considering opening a Travel Arrangement and Reservation Services business in Red Oak, IA. Just wondering if public transit is nearby? Like within 300 meters?",
            "Wanna open a Jewelry Stores, which is a sub category location somewhere in PA. Don’t care where exactly, just want low competition. Like under 25 other jewelry stores ideally.",
            "Hey just curious, if I were to put a Convenience Stores shop as a sub category in Jacksonville, could you check if there’s enough parking space? Not huge, like just 20 spots or so would work."
        ]
        
        # Create the results directory if it doesn't exist
        self.results_dir = "test_results"
        os.makedirs(self.results_dir, exist_ok=True)
        
        # Current timestamp for unique filenames
        self.timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    def tearDown(self):
        """Clean up test files - we're keeping result files."""
        pass

    def test_router_and_task_executor(self):
        """Test the router and task_executor to ensure they work together correctly."""
        for i, user_input in enumerate(self.test_inputs):
            print(f"\n{'='*80}")
            print(f"Test Case {i + 1}: {user_input}")
            print(f"{'='*80}")

            # Step 1: Call the router function to decompose the user input into tasks
            tasks = router(user_input)

            # Save router output with simple consistent naming format
            router_json_file = os.path.join(self.results_dir, f"test_{i+1}_router.json")
            with open(router_json_file, 'w') as f:
                json.dump(tasks, f, indent=2)
            
            print(f"Tasks saved to: {router_json_file}")
            print("\nDecomposed Tasks:")
            print(json.dumps(tasks, indent=4))

            # Step 2: Call the task_executor to execute the tasks
            try:
                print("\nExecuting tasks...")
                results = main(tasks)  # Execute tasks and get results
                
                print("Task Execution Completed Successfully")
                print("\nResults:")
                print(results)
                
                # Convert results to DataFrame if not already
                if isinstance(results, pd.DataFrame):
                    results_df = results
                elif isinstance(results, pd.Series):
                    results_df = results.to_frame()
                else:
                    # Convert other types (dict, list, etc.) to DataFrame
                    try:
                        if isinstance(results, dict):
                            results_df = pd.DataFrame([results])
                        elif isinstance(results, list):
                            results_df = pd.DataFrame(results)
                        else:
                            results_df = pd.DataFrame([{"result": str(results)}])
                    except:
                        results_df = pd.DataFrame([{"error": "Could not convert results to DataFrame"}])
                
                # Save results as Excel file
                excel_file = os.path.join(self.results_dir, f"test_{i+1}_results.xlsx")
                results_df.to_excel(excel_file, index=False)
                print(f"Results saved to: {excel_file}")
                
            except Exception as e:
                print(f"Task execution failed due to exception: {str(e)}")
                import traceback
                traceback.print_exc()
                
                # Save error information as DataFrame
                error_df = pd.DataFrame([{
                    "error": str(e),
                    "traceback": traceback.format_exc(),
                    "test_case": user_input
                }])
                excel_file = os.path.join(self.results_dir, f"test_{i+1}_results.xlsx")
                error_df.to_excel(excel_file, index=False)

def run_tests():
    """Run the tests in a way that doesn't terminate on first failure."""
    test_suite = unittest.TestSuite()
    test_suite.addTest(TestTaskExecutor('test_router_and_task_executor'))
    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(test_suite)

if __name__ == "__main__":
    # Run individual tests that won't exit on first error
    run_tests()