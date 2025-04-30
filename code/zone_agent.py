import os
import sys
import re
import importlib
import pandas as pd
import geopandas as gpd
import dotenv
from typing import List, Dict, Any
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage
from prompt import (
    ZEROSHOT_REACT_INSTRUCTION,
    COT_ANALYZER_INSTRUCTION,
    REACT_ANALYZER_INSTRUCTION
)
import numpy as np
from langchain_openai import ChatOpenAI

dotenv.load_dotenv()
llm = ChatOpenAI(openai_api_key=os.getenv("OPENAI_API_KEY"))

# Set up paths for imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from site_selection.loader import get_poi_spend_dataset, get_parking_dataset
from site_selection.zone import create_zone, assign_parking_zones, get_zone_center, get_neighbor_zones
from site_selection.analysis import get_spendparam_years, get_num_parking, get_largest_parking_lot_area, get_largest_parking_capacity, get_distance_km
from site_selection.filter import filter_df_based_on_zone, filter_pois_by_top_category, filter_pois_by_sub_category, get_transport_pois_in_zone
from site_selection.population import get_population

class ZoneAgent:
    def __init__(self,
                 mode: str = 'zero_shot',
                 model_name: str = 'gpt-4-turbo',
                 max_steps: int = 30,
                 max_retries: int = 3) -> None:

        self.mode = mode
        self.model_name = model_name
        self.max_steps = max_steps
        self.max_retries = max_retries
        self.constraint_threshold = None  # Default: no threshold
        self.action_results = {}
        
        self.llm = ChatOpenAI(
            temperature=0,
            model_name=self.model_name,
            max_tokens=512,  # Increased from 128 to 512 to handle longer answers
            openai_api_key=os.getenv("OPENAI_API_KEY")
        )

        self._reset_agent()
        
        print("\nLoading functions...")
        self.functions = self._load_functions()
        print(f"Loaded {len(self.functions)} functions")

        if self.mode == 'zero_shot':
            self.prompt_template = ZEROSHOT_REACT_INSTRUCTION
        elif self.mode == 'chain_of_thought':
            self.prompt_template = COT_ANALYZER_INSTRUCTION
        elif self.mode == 'react':
            self.prompt_template = REACT_ANALYZER_INSTRUCTION
        else:
            raise ValueError(f"Unknown mode: {mode}")
    
    def _clean_action_format(self, action: str) -> str:
        action = action.strip()

        # Remove "Action N:" prefix if it somehow exists
        if re.match(r'^Action\s+\d+:', action):
            action = re.sub(r'^Action\s+\d+:\s*', '', action)

        # Detect if this is a self_defined_logic block (even if messy line breaks)
        first_line = action.split('\n', 1)[0].strip()
        if first_line.startswith('self_defined_logic['):
            return action  # 🚀 Do not modify self_defined_logic actions

        # Otherwise, clean normally
        lines = action.split('\n')
        cleaned_lines = [re.sub(r'\s+', ' ', line.strip()) for line in lines if line.strip()]
        action = '\n'.join(cleaned_lines)

        # Handle parentheses to square brackets
        if '(' in action and ')' in action:
            action = re.sub(r'(\w+)\((.*?)\)', lambda m: f"{m.group(1)}[{m.group(2).strip()}]", action)

        # If model outputs multiple lines, only take the first valid one
        if '\n' in action:
            action = action.split('\n')[0]

        # Normalize spaces
        action = re.sub(r'\s+', ' ', action)

        return action

    def run(self, query: str, reset: bool = True) -> tuple:
        print(f"\n=== Starting Run ===")
        print(f"Query: {query}")
        self.query = query
        if reset:
            self._reset_agent()

        self.preload_datasets()

        while not self.is_finished() and not self.is_halted():
            print(f"\nStep {self.step_n}:")
            self.step()

        print("\n=== Run Complete ===")
        print(f"Final Answer: {self.answer[:100]}...")
        return self.answer, self.scratchpad

    def step(self) -> None:

        thought = self._query_llm()
        # Extract only the thought part if it contains both thought and action
        if 'Action:' in thought:
            thought = thought.split('Action:')[0].strip()
        # Remove the "Thought:" prefix if it exists
        if thought.startswith('Thought:'):
            thought = thought[8:].strip()
        
        self.scratchpad += f'\nThought {self.step_n}: {thought}'
        print(f"Thought: {thought}")

        # Now get the action specifically
        action = self._query_llm(force_action=True)
        if action.startswith('Finish['):
            # Retry to get full Finish result without token limit
            action = self._query_llm(force_finish_action=True)
        # Extract just the action if it includes other parts
        if 'Action:' in action:
            action = action.split('Action:')[1]
            if 'Thought:' in action:
                action = action.split('Thought:')[0]
            action = action.strip()
        
        
        action = self._clean_action_format(action)
        self.scratchpad += f'\nAction {self.step_n}: {action}'
        print(f"Action: {action}")

        # Check if the action is to finish
        if action.startswith('Finish['):
            # Extract the content between [ and ]
            match = re.match(r'^Finish\[(.*)', action)
            if match:
                self.answer = match.group(1)
            else:
                # If no content in brackets, look for zone IDs in the last observation
                # The last observation should contain "Survived Zones: [...]"
                if "Survived Zones:" in self.scratchpad:
                    # Find the last occurrence of Survived Zones:
                    scratchpad_parts = self.scratchpad.split("Survived Zones:")
                    if len(scratchpad_parts) > 1:
                        last_survived_zones = scratchpad_parts[-1].strip()
                        # Extract the list of zones
                        match = re.search(r'\[(.*?)\]', last_survived_zones)
                        if match:
                            # Extract just the numbers as a comma-separated string
                            zone_list = match.group(1)
                            self.answer = zone_list
                        else:
                            # If we can't find brackets, just use the whole thing
                            self.answer = last_survived_zones
                    else:
                        self.answer = ""
                else:
                    self.answer = ""
            
            self.finished = True
            self.scratchpad += f'\nObservation {self.step_n}: Finished analysis.'
            print(f"Observation: Finished analysis.")
            print(f"[DEBUG] Full answer captured: {self.answer}")
            return
        action_type, action_args, needs_loop, operator_symbol, threshold = self._parse_action(action)

        if action_type is None:
            print("Invalid action format detected. Retrying step...")
            self.scratchpad += "\nInvalid action format detected. Retrying step..."
            return

        print("\nExecuting action...")
        self.scratchpad += f'\nObservation {self.step_n}: '
        observation = self._execute_action(action_type, action_args, needs_loop, operator_symbol, threshold)
        
        action_key = f"action{self.step_n}"  # <<< only 'actionN', clean
        if hasattr(self, 'current_data') and self.current_data is not None:
            self.action_results[action_key] = self.current_data

            
        self.scratchpad += str(observation)
        print(f"Observation: {observation[:100]}...")

        print(self.scratchpad)

        self.step_n += 1  # Increment step counter


    def _query_llm(self, force_action: bool = False, force_finish_action: bool = False) -> str:
        content = self._build_prompt()
        
        if force_finish_action:
            # If we need the complete Finish action, ask explicitly for it
            content += "\nProvide the COMPLETE Finish action with ALL zone IDs. Just list the numbers separated by commas. Example: Finish[840, 1660, 1281, ...]"
            
            # ⚡ Important: Temporarily use a no-token-limit LLM
            temp_llm = ChatOpenAI(
                temperature=0,
                model_name=self.model_name,
                max_tokens=None,  # No token limit
                openai_api_key=self.llm.openai_api_key
            )
            response = temp_llm.invoke([HumanMessage(content=content)])
        else:
            if force_action:
                # Explicitly ask for just the next Action
                content += "\nNow, provide ONLY the next Action. If using Finish with zones, list just the numbers: Finish[840, 1660, ...]"
            else:
                # Explicitly ask for just the current Thought
                content += "\nNow, provide ONLY the current Thought without including any Action."
            
            # Normal LLM usage with current token limit
            response = self.llm.invoke([HumanMessage(content=content)])
        
        return response.content.strip()

    def _build_prompt(self) -> str:
        MAX_SCRATCHPAD_TOKENS = 8000  # or 4000, or 6000
        scratchpad_tokens = len(self.scratchpad) // 4  # Rough estimate: 1 token ≈ 4 chars
        if scratchpad_tokens > MAX_SCRATCHPAD_TOKENS:
            # Keep only the last 8000 tokens worth of text
            scratchpad_cutoff = int(MAX_SCRATCHPAD_TOKENS * 4)
            scratchpad = self.scratchpad[-scratchpad_cutoff:]
        else:
            scratchpad = self.scratchpad

        return self.prompt_template.format(
            query=self.query,
            scratchpad=scratchpad
        )

    def _resolve_argument(self, arg: str):
        """Resolve argument name to actual object, or return a warning if missing."""
        arg = arg.strip()
        if arg == 'poi_spend_df':
            if hasattr(self, 'poi_spend_df'):
                return self.poi_spend_df
            else:
                raise ValueError("poi_spend_df not available. Please call get_poi_spend_dataset[] first.")
        if arg == 'parking_df':
            if hasattr(self, 'parking_df'):
                return self.parking_df
            else:
                raise ValueError("parking_df not available. Please call get_parking_dataset[] first.")
        if arg == 'zone_df':
            if hasattr(self, 'zone_df'):
                return self.zone_df
            else:
                raise ValueError("zone_df not available. Please call create_zone[poi_spend_df] first.")
        return arg  # If unknown, pass raw

    def _parse_action(self, action: str) -> tuple:
        """
        Parses a multi-line action string into (action_type, action_args, needs_loop, operator_symbol, threshold).

        Supports regular actions and self_defined_logic actions.
        """

        if action.startswith('Finish['):
            return 'Finish', None, False, None, None

        lines = action.strip().split('\n')
        if not lines:
            return None, None, None, None, None

        first_line = lines[0].strip()

        if 'self_defined_logic' in first_line:
            # Special case: self_defined_logic custom code
            action_type = 'self_defined_logic'
            open_idx = first_line.find('[')
            close_idx = first_line.rfind(']')
            if open_idx != -1 and close_idx != -1 and close_idx > open_idx:
                action_args = first_line[open_idx + 1 : close_idx].strip()
            else:
                # In case code spans multiple lines inside brackets
                action_args = '\n'.join(lines)[len('self_defined_logic['):-1].strip()
        else:
            # Normal case
            func_pattern = r'^(\w+)\[(.*?)\]$'
            func_match = re.match(func_pattern, first_line)
            if not func_match:
                return None, None, None, None, None
            action_type = func_match.group(1)
            action_args = func_match.group(2)

        needs_loop = action_type in NEED_LOOP_FUNCTIONS
        operator_symbol = None
        threshold = None

        for line in lines[1:]:
            line = line.strip()
            if line.startswith('Needs Loop Over Zones:'):
                needs_loop_value = line.split(':', 1)[1].strip()
                needs_loop = (needs_loop_value == 'Yes')
            elif line.startswith('Threshold:'):
                threshold_line = line.split(':', 1)[1].strip()
                threshold_line = threshold_line.replace('[', '').replace(']', '')
                parts = threshold_line.split()
                if len(parts) == 2:
                    if parts[0] != "None" and parts[1] != "None":
                        operator_symbol = parts[0]
                        threshold = int(parts[1])
                    else:
                        operator_symbol = None
                        threshold = None

        return action_type, action_args, needs_loop, operator_symbol, threshold

    def _is_valid_output(self, output):
        return True
    
    def preload_datasets(self):
        print("\n=== Preloading Datasets ===")
        self.poi_spend_df = self.functions['get_poi_spend_dataset']()
        self.parking_df = self.functions['get_parking_dataset']()
        self.zone_df = self.functions['create_zone'](self.poi_spend_df)
        self.parking_df = self.functions['assign_parking_zones'](self.parking_df, self.zone_df)
        print("=== Dataset Preloading Complete ===")

    def _execute_action(self, action_type: str, action_args: str, needs_loop: bool, operator_symbol: str, threshold: int) -> str:
        if action_type is None:
            self.finished = True
            return "Invalid action format. Halting."

        
        if action_type not in self.functions:
            return f"Invalid action: {action_type}"

        import inspect

        # 🌟 NEW: Resolve and autofill missing args
        func = self.functions[action_type]
        sig = inspect.signature(func)
        param_names = list(sig.parameters.keys())

        if action_args:
            # Special case: self_defined_logic must receive raw code
            if action_type == "self_defined_logic":
                code_arg = action_args.strip()
                
                # Step 1: Find real closing triple quotes
                if code_arg.startswith("'''"):
                    end_idx = code_arg.find("'''", 3)
                    if end_idx != -1:
                        code_arg = code_arg[3:end_idx]
                elif code_arg.startswith('"""'):
                    end_idx = code_arg.find('"""', 3)
                    if end_idx != -1:
                        code_arg = code_arg[3:end_idx]
                else:
                    # Fallback: regular single or double quotes
                    if code_arg.startswith("'") and code_arg.endswith("'"):
                        code_arg = code_arg[1:-1]
                    elif code_arg.startswith('"') and code_arg.endswith('"'):
                        code_arg = code_arg[1:-1]
                
                args_list = [code_arg]

            else:
                # Regular case
                args_list = []
                for arg in re.split(r',\s*(?=(?:[^"]*"[^"]*")*[^"]*$)', action_args):
                    arg = arg.strip()
                    if arg.startswith('"') and arg.endswith('"'):
                        arg = arg[1:-1]
                    if arg.startswith("'") and arg.endswith("'"):
                        arg = arg[1:-1]
                    args_list.append(self._resolve_argument(arg))
        else:
            args_list = []



        missing_params = [p for p in param_names if p not in (action_args or '')]

        for p in missing_params:
            if p == 'zone_df' and hasattr(self, 'zone_df'):
                args_list.append(self.zone_df)
            elif p == 'poi_spend_df' and hasattr(self, 'poi_spend_df'):
                args_list.append(self.poi_spend_df)
            elif p == 'parking_df' and hasattr(self, 'parking_df'):
                args_list.append(self.parking_df)
        if action_type == "self_defined_logic":
            # Special execution
            result = self.self_defined_logic(args_list[0])
            return result
        if needs_loop:
            return self._execute_with_loop(func, args_list, operator_symbol, threshold)
        else:
            return self._execute_normal(func, args_list, operator_symbol, threshold)
    
    import numpy as np  # Make sure you have this
    def _check_threshold(self, output: Any, operator_symbol: str, threshold: int) -> bool:
        if operator_symbol is None or threshold is None:
            return self._is_valid_output(output)

        if isinstance(output, (int, float, np.integer, np.floating)):
            if operator_symbol == ">=":
                return output >= threshold
            elif operator_symbol == "<=":
                return output <= threshold
            elif operator_symbol == ">":
                return output > threshold
            elif operator_symbol == "<":
                return output < threshold
            elif operator_symbol == "=":
                return output == threshold

        return self._is_valid_output(output)



    def _execute_normal(self, func, args_list: list, operator_symbol: str, threshold: int) -> str:
        try:
            result = func(*args_list)
            self.current_data = result

            if not self._check_threshold(result, operator_symbol, threshold):
                # Don't halt - let the LLM decide what to do
                return f"Constraint not satisfied. Result: {result}, Operator: {operator_symbol}, Threshold: {threshold}"

            return str(result)
        except Exception as e:
            print(f"Error executing normal action: {str(e)}")
            # On error, let the LLM decide what to do next
            return f"Error executing action: {str(e)}"

    def _execute_with_loop(self, func, args_list: list, operator_symbol: str, threshold: int) -> str:
        try:
            if not hasattr(self, 'zone_df'):
                print("[Error] zone_df not available. Halting.")
                return "zone_df not available for looping. Halting."

            survived_zone_ids = []

            # Debug: Let's see what's in args_list

            # First, let's create a mapping of positions and types
            arg_info = []
            for i, arg in enumerate(args_list):
                if isinstance(arg, (pd.DataFrame, gpd.GeoDataFrame)):
                    # Determine which DataFrame this is
                    is_parking = False
                    is_poi_spend = False
                    if hasattr(self, 'parking_df') and arg is self.parking_df:
                        is_parking = True
                    elif hasattr(self, 'poi_spend_df') and arg is self.poi_spend_df:
                        is_poi_spend = True
                    arg_info.append({'pos': i, 'type': 'df', 'is_parking': is_parking, 'is_poi_spend': is_poi_spend})
                elif isinstance(arg, str) and arg == 'zone_id':
                    arg_info.append({'pos': i, 'type': 'zone_id'})
                else:
                    arg_info.append({'pos': i, 'type': 'other', 'value': arg})


            for zone_id in self.zone_df['zone_id'].unique():
                try:
                    # Create the arguments for this specific zone
                    loop_args = [None] * len(args_list)
                    
                    for info in arg_info:
                        if info['type'] == 'df':
                            # Filter the DataFrame for this zone
                            df_arg = args_list[info['pos']]
                            filtered_df = filter_df_based_on_zone(df_arg, zone_id)
                            loop_args[info['pos']] = filtered_df
                        elif info['type'] == 'zone_id':
                            loop_args[info['pos']] = zone_id
                        else:
                            loop_args[info['pos']] = info['value']

                    output = func(*loop_args)

                    if self._is_valid_output(output):
                        survived_zone_ids.append(zone_id)

                except Exception as zone_error:
                    print(f"[Zone Error] Failed processing zone {zone_id}: {str(zone_error)}")
                    continue

            print("\n[Loop Execution] Finalizing results...")
            if survived_zone_ids:
                filtered_zone_df = self.zone_df[self.zone_df['zone_id'].isin(survived_zone_ids)]
                self.current_data = filtered_zone_df
                print(f"[Loop Execution] Survived Zones: {survived_zone_ids}")
                return f"Survived Zones: {survived_zone_ids}"
            else:
                print("[Loop Execution] No zones survived.")
                return "No zones survived the loop."

        except Exception as e:
            print(f"[Critical Error] Loop execution failed: {str(e)}")
            return f"Error during looping execution: {str(e)}"

    def is_finished(self) -> bool:
        return self.finished

    def is_halted(self) -> bool:
        return self.step_n > self.max_steps

    def _reset_agent(self) -> None:
        self.step_n = 1
        self.finished = False
        self.answer = ''
        self.scratchpad = ''
        self.retry_record = {}
        self.current_data = None

    def _load_functions(self) -> Dict[str, Any]:
        return {
            'get_poi_spend_dataset': get_poi_spend_dataset,
            'get_parking_dataset': get_parking_dataset,
            'create_zone': create_zone,
            'assign_parking_zones': assign_parking_zones,
            'filter_df_based_on_zone': filter_df_based_on_zone,
            'filter_pois_by_top_category': filter_pois_by_top_category,
            'filter_pois_by_sub_category': filter_pois_by_sub_category,
            'get_zone_center': get_zone_center,
            'get_spendparam_years': get_spendparam_years,
            'get_num_parking': get_num_parking,
            'get_largest_parking_lot_area': get_largest_parking_lot_area,
            'get_largest_parking_capacity': get_largest_parking_capacity,
            'get_distance_km': get_distance_km,
            'get_neighbor_zones': get_neighbor_zones,
            'get_population': get_population,
            'get_transport_pois_in_zone': get_transport_pois_in_zone,
            'self_defined_logic': self.self_defined_logic,
            'Finish': self._finish  # Special finish action
        }

    def _finish(self, args: str) -> str:
        self.answer = args
        self.finished = True
        return "Finished analysis."

        

    import re
    import textwrap
    def self_defined_logic(self, code: str):
        """
        Executes custom code provided in self_defined_logic[].
        Replaces $actionN references with actual previous action outputs.
        """

        # Step 1: Prepare local environment
        local_env = {}

        # Step 2: Inject previous action results
        for key, value in self.action_results.items():
            injected_var = f"_injected_{key}"
            local_env[injected_var] = value

        # Step 3: Process the code
        code_processed = code.strip()
        
        # Handle triple quotes
        if code_processed.startswith("'''"):
            end_idx = code_processed.find("'''", 3)
            if end_idx != -1:
                code_processed = code_processed[3:end_idx]
        elif code_processed.startswith('"""'):
            end_idx = code_processed.find('"""', 3)
            if end_idx != -1:
                code_processed = code_processed[3:end_idx]
        
        # Replace action references
        for key in self.action_results.keys():
            pattern = re.compile(rf"\${key}(?!\w)")  # match $action2 but NOT $action22 etc
            code_processed = pattern.sub(f"_injected_{key}", code_processed)

        # Ensure proper indentation
        lines = code_processed.split('\n')
        if len(lines) > 1:
            # Find minimum indentation
            min_indent = float('inf')
            for line in lines[1:]:  # Skip first line
                if line.strip():  # Only consider non-empty lines
                    indent = len(line) - len(line.lstrip())
                    min_indent = min(min_indent, indent)
            
            # Remove minimum indentation from all lines
            if min_indent != float('inf'):
                code_processed = '\n'.join(
                    line[min_indent:] if line.strip() else line
                    for line in lines
                )

        # Step 4: Execute
        try:
            # Create a new scope for execution
            exec_scope = {}
            exec(code_processed, exec_scope, local_env)
        except Exception as e:
            raise RuntimeError(f"Error executing self_defined_logic code: {e}")

        # Step 5: Make sure 'result' exists
        if "result" not in local_env:
            raise ValueError("Custom self_defined_logic code must assign a 'result' variable.")

        return local_env["result"]







##################################################################################333333333
# THESE FUNCTIONS NEED TO BE LOOPED OVER (FOR SURE, some of them the llm can be creative then I didnt include here)
NEED_LOOP_FUNCTIONS = [
    'filter_df_based_on_zone',
    'get_spendparam_years',
    'get_population',
    'get_distance_km',
    'get_zone_center',
    'get_neighbor_zones',
    'get_num_parking', 
    'get_largest_parking_lot_area',
    'get_largest_parking_capacity'
]