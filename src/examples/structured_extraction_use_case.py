
import asyncio
import httpx
import json
import ollama

# --- Configuration ---
MODEL_TO_USE = "llama3"  # A model capable of following JSON instructions
UNSTRUCTURED_TEXT = """
I recently purchased the "Quantum Weaver Pro" keyboard, and it's been a game-changer!
The typing experience is sublime, and the customizable RGB lighting is a fantastic touch.
I'd easily give it 5 out of 5 stars. My only minor gripe is that the software for
customizing macros was a bit clunky to install, but once set up, it works flawlessly.
Highly recommended for both gaming and productivity.
"""

# --- Use Case: Extract structured data from the text above ---

async def extract_with_direct_api():
    """
    Uses a direct HTTP request to the Ollama API to leverage JSON Mode.
    This is the superior method for reliable, structured output.
    """
    print("--- Attempting extraction with Direct API (JSON Mode) ---")

    prompt = f"""
    From the following customer review, extract the product name, the star rating,
    and a one-sentence summary. Please provide the output in a JSON object with the
    keys 'product_name', 'rating', and 'summary'.

    Review:
    {UNSTRUCTURED_TEXT}
    """

    try:
        # Using httpx to make a direct web request to the Ollama application
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                "http://localhost:11434/api/generate",
                json={
                    "model": MODEL_TO_USE,
                    "prompt": prompt,
                    "format": "json",  # This forces the model to output valid JSON
                    "stream": False
                },
            )
            response.raise_for_status()  # Raise an exception for bad status codes

            # The response from the API is a JSON string, which we parse
            response_data = response.json()
            
            # The actual model output is in the 'response' key, and it's a JSON string itself
            extracted_json = json.loads(response_data["response"])

            print("✅ Success! Received clean, valid JSON:")
            print(json.dumps(extracted_json, indent=2))
            return extracted_json

    except Exception as e:
        print(f"❌ Direct API call failed: {e}")
        print("Is the Ollama Application running? Is the model '{MODEL_TO_USE}' installed?")
        return None


async def extract_with_python_library():
    """
    Uses the 'ollama' Python library.
    This method is less reliable for structured data because it can't force JSON output.
    """
    print("\n--- Attempting extraction with Python Library (No JSON Mode) ---")

    prompt = f"""
    From the following customer review, extract the product name, the star rating,
    and a one-sentence summary. Please provide the output *only* in a JSON object
    with the keys 'product_name', 'rating', and 'summary'.

    Review:
    {UNSTRUCTURED_TEXT}
    """
    
    try:
        # Using the installed ollama python library
        response = await ollama.AsyncClient().generate(
            model=MODEL_TO_USE,
            prompt=prompt,
            # Note: The library (v0.5.1) has no 'format' parameter.
        )

        model_output = response['response']
        print("✅ Success! Received a response string from the library:")
        print(f"Raw output:\n---\n{model_output}\n---")

        # Now, we have to try and parse the string, which might fail
        try:
            # Attempt to find the JSON within the potentially messy string
            json_part = model_output[model_output.find('{'):model_output.rfind('}')+1]
            extracted_json = json.loads(json_part)
            print("\n✅ Library output was successfully parsed into JSON.")
            print(json.dumps(extracted_json, indent=2))
        except json.JSONDecodeError:
            print("\n❌ Failed to parse the library's output into JSON. The string is not clean.")

    except Exception as e:
        print(f"❌ Python library call failed: {e}")


async def main():
    """Main function to run the comparison."""
    # First, ensure the model is available by pulling it.
    print(f"Ensuring model '{MODEL_TO_USE}' is available locally...")
    try:
        await ollama.AsyncClient().pull(MODEL_TO_USE)
    except Exception as e:
        print(f"Could not pull model. Please ensure Ollama is running. Error: {e}")
        return

    await extract_with_direct_api()
    await extract_with_python_library()
    
    print("\n---")
    print("Conclusion: The Direct API with JSON Mode is the clear winner for reliable structured data.")


if __name__ == "__main__":
    # To run this example:
    # 1. Make sure the Ollama application is running on your computer.
    # 2. Run this script from your terminal:
    #    python -m src.examples.structured_extraction_use_case
    asyncio.run(main())
