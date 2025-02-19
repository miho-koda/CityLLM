# openai_module.py
import types


def run_code_capture_output(code_str):
# Create a fresh namespace
    namespace = {}
    
    try:
        # Execute the code string to define the function
        exec(code_str, namespace)
        
        # Get all user-defined functions from the namespace
        user_functions = {
            name: obj for name, obj in namespace.items()
            if isinstance(obj, types.FunctionType)  # Check if it's a function
            and not name.startswith('__')  # Exclude special methods
            and obj.__module__ is None  # Functions defined in exec have module=None
        }
        
        if not user_functions:
            raise RuntimeError("No user-defined function found in the code string")
        
        # Get the first user-defined function
        function_name, func = next(iter(user_functions.items()))
        
        # Call the function and return its result
        return func()
        
    except Exception as e:
        if isinstance(e, RuntimeError):
            raise e
        raise RuntimeError(f"Error during execution: {e}")

def generate_code_for_question(question, client, geo_docs):
    """
    Call the OpenAI API with the provided question, then extract and clean the returned Python code.
    """
    response = client.chat.completions.create(
        model='gpt-4-turbo',
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a Python function generator specialized in geospatial data processing. "
                    "Return only valid and executable Python code with no explanations. "
                    "Use GeoPandas version 1.0.1 and Geodatasets version 2024.8.0. "
                    "Always use 'geodatasets.get_path()' to access datasets. "
                    "Use the provided documentation below to generate responses.\n\n"
                    f"{geo_docs}\n\n"
                    "Never generate responses outside this documentation."
                    "Create a Python function that returns the final result instead of printing it\n"
                )
            },
            {"role": "user", "content": question}
        ]
    )

    generated_code = response.choices[0].message.content

    # Remove code block formatting if present
    if generated_code.startswith("```python"):
        generated_code = generated_code[9:]
    if generated_code.endswith("```"):
        generated_code = generated_code[:-3]
    
    return generated_code.strip()

def process_question(question, client, geo_docs):
    """
    Generate code for a given question using the OpenAI API, execute it, and return its printed output.
    """
    # Generate the code
    generated_code = generate_code_for_question(question, client, geo_docs)
    
    # Print the generated code for inspection (optional)
    print(f"\n### Response for: {question} ###\n")
    print(generated_code)
    print("\n" + "="*80 + "\n")
    
    # Execute the generated code and capture its output
    output = run_code_capture_output(generated_code)
    return output

