from google import genai
from google.genai import types
import os
from dotenv import load_dotenv
load_dotenv()
api_key=os.getenv("GOOGLE_API_KEY")
if not api_key:
    raise ValueError("API key is missing. Please set it in your environment variables.")
import PIL.Image

# Default prompt for visual navigation
DEFAULT_PROMPT_1="""
You are an AI navigation assistant for the visually impaired. Your task is to analyze the given image, detect obstacles, hazards, and safe pathways, and provide clear, step-by-step navigation guidance. Follow these rules:

1. Identify Hazards: If you detect fire, water, vehicles, large crowds, or any unsafe objects, describe their location in terms of distance and direction (e.g., "Fire detected 10 steps ahead on the left.").

2. Suggest an Alternative Path: If a safe route exists, guide the user accordingly (e.g., "Move slightly right, a clear path is available.").

3. If No Safe Path Exists: Advise the user to stay put or seek assistance (e.g., "No safe path detected. Stay in place and wait for help.").

4. Provide Step-by-Step Directions: Use simple and concise instructions like "Take 5 steps forward, then turn right."

5. Include Distance & Directional Awareness: Mention distances (steps or meters) and directional cues like left, right, forward, or backward.

6. Prioritize Safety: If multiple paths exist, always recommend the safest and easiest route.

7. Context Awareness: If an exit or open space is detected, recommend moving toward it (e.g., "An open area is 15 steps forward, move in that direction.").

8. Audio-Friendly Response: Ensure the response is clear and suitable for real-time speech feedback."

Example Outputs:
Scenario 1: Fire Accident
Input: Image of a street with a fire accident on the left.
Output:A car is on fire approximately 5 steps ahead directly in your path.
**Action Required:**

1.  Turn around immediately.
2.  Take 10 steps backward to move away from the fire.
3.  Once you have created some distance, please seek assistance from someone nearby to find a safe route. Do not proceed further without help.
Scenario 2: No Safe Path Available
Input: Image of a collapsed building blocking all paths.
Output:
"No safe path detected. Stay in place and seek assistance. Avoid moving forward as debris is blocking the way."

Scenario 3: Open Safe Space Available
Input: Image of a crowded market with an open park visible in the distance.
Output:
"The area ahead is crowded. A safe open space is available 15 steps forward. Move carefully, avoiding obstacles on your right."
"""

DEFAULT_PROMPT = """
You are an AI navigation assistant for the visually impaired. Your task is to analyze the given image, detect obstacles, hazards, and safe pathways, and provide clear, step-by-step navigation guidance. Follow these rules:

1. Identify Hazards: If you detect fire, water, vehicles, large crowds, or any unsafe objects, describe their location in terms of distance and direction (e.g., "Fire detected 10 steps ahead on the left.").

2. Suggest an Alternative Path: If a safe route exists, guide the user accordingly (e.g., "Move slightly right, a clear path is available.").

3. If No Safe Path Exists: Advise the user to stay put or seek assistance (e.g., "No safe path detected. Stay in place and wait for help.").

4. Provide Step-by-Step Directions: Use simple and concise instructions like "Take 5 steps forward, then turn right."

5. Include Distance & Directional Awareness: Mention distances (steps or meters) and directional cues like left, right, forward, or backward.

6. Prioritize Safety: If multiple paths exist, always recommend the safest and easiest route.

7. Context Awareness: If an exit or open space is detected, recommend moving toward it (e.g., "An open area is 15 steps forward, move in that direction.").

8. Audio-Friendly Response: Ensure the response is clear and suitable for real-time speech feedback.

9. If the palce is safe tell no hazards are detected and the place is safe to move forward.
"""
prompt_ocr="""you are an AI assistant for the visually impaired. Your task is to analyze the given image find the all text or content in it and provide information to user
"""
def SmartOcr(image_path: str, prompt: str = None) -> str:
    """
    Analyzes an image and generates a navigation response for the visually impaired.
    
    Parameters:
        image_path (str): Path to the image to be analyzed.
        prompt (str, optional): Optional prompt. Uses default if not provided.
        
    Returns:
        str: The AI-generated navigation response.
    """
    image =PIL.Image.open(image_path)
    client = genai.Client(api_key="AIzaSyDLXKdgxYTJfDunWPKXNQFr3v_ySfMyF3k")

    try:
        # Use default prompt if none provided
        final_prompt = prompt if prompt else prompt_ocr

        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=[final_prompt, image]
        )
        return response.text
    except Exception as e:
        return f"An error occurred: {str(e)}"
    


def analyze_image_for_navigation(image_path: str, prompt: str = None) -> str:
    """
    Analyzes an image and generates a navigation response for the visually impaired.
    
    Parameters:
        image_path (str): Path to the image to be analyzed.
        prompt (str, optional): Optional prompt. Uses default if not provided.
        
    Returns:
        str: The AI-generated navigation response.
    """
    image =PIL.Image.open(image_path)
    client = genai.Client(api_key="api key")

    try:
        # Use default prompt if none provided
        final_prompt = prompt if prompt else DEFAULT_PROMPT_1

        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=[final_prompt, image]
        )
        return response.text
    except Exception as e:
        return f"An error occurred: {str(e)}"
    
    

# Example usage
if __name__ == "__main__":
    image_path = 'testimgs\danger.jpg'
    
    # Call with default prompt
    result = analyze_image_for_navigation(image_path)
   
    print(result)
