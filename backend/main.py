import os
import json
from pydantic import BaseModel
from typing import Dict, Any
import agentic_ai
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables from a .env file
load_dotenv()

# ------------------------------------------------------------------------------
# Ensure you have set the appropriate API key in your environment variables.
# ------------------------------------------------------------------------------
LLM_API_KEY = os.getenv("LLM_API_KEY")
LLM_API_URL = os.getenv("LLM_API_URL", "https://openrouter.ai/api/v1")
CHAT_MODEL = os.getenv("CHAT_MODEL")

# Configure the llm model
client = OpenAI(api_key=LLM_API_KEY, base_url=LLM_API_URL)

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
)
def get_food_details(query: str, region: str) -> dict:
    return {"name": "Gulab Jamun", "PPP": 0.2}


# Generate tool instructions
message_history = agentic_ai.initialize_message_history("You are an expert chef with mastery food related maters.")

agentic_ai.append_to_history("user","what indian desert is soft and sweet?", message_history)
assistant_reply = agentic_ai.generate_response(client, CHAT_MODEL, message_history)

# Call the tool
tool_output = agentic_ai.call_tool_from_json(assistant_reply, message_history)


# Final call to generate reply
final_reply = agentic_ai.generate_response(client, CHAT_MODEL, message_history)

print("Message history:", message_history[1:])

