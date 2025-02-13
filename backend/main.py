import os
import json
import openai
from pydantic import BaseModel, ValidationError
from typing import Dict, Any
import agentic_ai

# ------------------------------------------------------------------------------
# Configuration: Choose the API base.
# Set USE_OPENROUTER = True to use OpenRouter's API; otherwise, use OpenAI's default.
# Ensure you have set the appropriate API key in your environment variables.
# ------------------------------------------------------------------------------
openai.api_key = os.getenv("LLM_API_KEY")
openai.api_base = os.getenv("LLM_API_URL", "https://openrouter.ai/api/v1")


# ------------------------------------------------------------------------------
# Define a Pydantic model to enforce the expected structured output.
# ------------------------------------------------------------------------------
class ToolCall(BaseModel):
    tool: str
    arguments: Dict[str, Any]

class GraphToolResponse(BaseModel):
    success: bool
    imageID: str

class RespondOutput(BaseModel):
    text: str

# Define tools using the decorator
@agentic_ai.register_tool(
    description="Searches for a food item based on the query.",
    arguments=(
        "query: string; search query for food details.\n"
        "region: string; regional style for the recipe."
    ),
    response=(
        "name: string; food name.\n"
        "PPP: number; price per piece."
    ),
    example_in='{"query":"A sweet, soft-textured dish","region":"Indian"}',
    # example_out='{"name":"Gulab Jamun", "PPP":0.2}'
)
def get_food_details(query: str, region: str) -> dict:
    """Returns food item details based on the query and region."""
    return {"name": "Gulab Jamun", "PPP": 0.2}


# Generate tool instructions
print("=== Tool Instructions ===")
print(agentic_ai.generate_tool_instructions())

# Call a tool dynamically
print("=== Calling 'get_recipe' ===")
print(agentic_ai.call_tool("get_food_details", {"dish": "Lasagna"}))
