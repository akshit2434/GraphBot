import functools
from pydantic import BaseModel
from typing import Union
import json
import re
import openai

# Global dictionary to store tools
TOOLS = {}

# Define a structured response model for tool calling
class ToolCall(BaseModel):
    tool_name: str
    arguments: dict

# Define a flexible response model that can be either structured or plain text
class ResponseModel(BaseModel):
    data: Union[ToolCall, str]  # Can be structured data or plain text


def register_tool(
    *,
    description: str,
    arguments: str,
    response: str,
    example_in: str = None,
    example_out: str = None
):
    """
    Decorator to register a tool with inline metadata.

    Parameters:
      description: Short explanation of what the tool does.
      arguments: Expected arguments (name, type, description).
      response: Expected response format.
      example_in: Example input JSON.
      example_out: Example output JSON.
    """
    def decorator(func):
        metadata = {
            "description": description,
            "arguments": arguments,
            "response": response,
            "example_in": example_in,
            "example_out": example_out,
            "function": func,
        }
        TOOLS[func.__name__] = metadata

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)
        wrapper.__tool_metadata__ = metadata
        return wrapper
    return decorator

def generate_tool_instructions(*tool_names):
    """
    Generates formatted tool descriptions for given tools.
    If no tool names are provided, it generates instructions for all.

    Returns:
      A formatted string containing tool descriptions.
    """
    selected_tools = TOOLS if not tool_names else {name: TOOLS.get(name) for name in tool_names if name in TOOLS}

    if not selected_tools:
        return "No tools are available."

    tool_info = "You have access to the following tools:\n\n"
    tool_info += "\n\n".join(
        f"**{tool_name}**: {meta['description']}\n\n"
        f"Arguments:\n{meta['arguments']}\n\n"
        f"Response Format:\n{meta['response']}\n\n"
        + (f"Example Arguments: {meta['example_in']}\n" if meta.get('example_in') else "")
        + (f"Example Response: {meta['example_out']}\n" if meta.get('example_out') else "")
        for tool_name, meta in selected_tools.items()
    )

    return tool_info

def generate_system_prompt(prompt, tool_instructions: str = None):
    """
    Generates a system prompt with tool instructions.

    Parameters:
        - prompt: The system prompt given by the developer.
        - tool_instructions: Formatted tool descriptions. (optional)
            When not specified, it generates instructions for all tools.

    Returns:
        - A system prompt string with tool instructions.
    """
    tool_instructions = tool_instructions or generate_tool_instructions()
    system_prompt = (
        prompt + "\n\n" +
        f"{tool_instructions}\n\n"
        "\n\nYou can either call a tool, or respond to the user. If you want to call a tool, use the following format:\n\n"
        """
        {
            "tool_name": "Name of the tool",
            "arguments": {
                // Arguments in JSON format
            }
        }
        
        Example tool call:
        {
            "tool_name": "get_food_details_tool",
            "arguments": {
                "query": "A sweet, soft-textured dish",
                "region": "Indian"
            }
        }
        """
        """If you want to respond to the user, simply provide the text response in the following format:
        {
            "text": "Your response goes here."
        }
        """
        
    )

    return system_prompt

async def call_tool(tool_name: str, args: dict) -> dict:
    """
    Calls a registered tool dynamically.

    Parameters:
      tool_name: Name of the tool function.
      args: Dictionary of arguments to pass.

    Returns:
      A structured response (textual and JSON) that can be used by an LLM.
    """
    if tool_name not in TOOLS:
        return {
            "success": False,
            "message": f"Error: The tool '{tool_name}' does not exist. Please refer to the available tools.",
            "available_tools": list(TOOLS.keys()),
        }

    tool_metadata = TOOLS[tool_name]
    tool_func = tool_metadata["function"]

    # Validate missing arguments
    expected_args = [
        arg.split(":")[0].strip()
        for arg in tool_metadata["arguments"].split("\n")
        if "optional" not in arg.split(":")[0].lower()
    ]
    missing_args = [arg for arg in expected_args if arg not in args]

    if missing_args:
        return {
            "success": False,
            "message": f"Error: Missing required arguments: {', '.join(missing_args)}.",
            "expected_arguments": tool_metadata["arguments"],
            "example_usage": tool_metadata.get("example_in", "N/A"),
        }

    # Execute the tool
    try:
        result = await tool_func(**args)
        if "success" not in result.keys() or result.success == False:
            return {
                "success": True,
                "tool_name": tool_name,
                "output": json.dumps(result)
            }
        else:
            return json.dumps(result)
    except Exception as e:
        return {
            "success": False,
            "tool_name": tool_name,
            "error": f"An error occurred while executing '{tool_name}': {str(e)}"
        }

def remove_markdown(text: str) -> str:
    """Remove markdown syntax from the text."""
    # Remove markdown code block syntax if present
    text = re.sub(r'^```[a-zA-Z]*\n*', '', text)
    text = re.sub(r'\n*```$', '', text)
    text = text.strip()
    return text

async def call_tool_from_json(text:str|dict, history:dict, auto_append:bool=True) -> dict:
    """
    Calls a tool based on a JSON input.

    Parameters:
      text: JSON-formatted string containing tool name and arguments.
        history: List of message dictionaries.
        auto_append: Whether to automatically append the tool response to the history.

    Returns:
      A structured response (textual and JSON) that can be used by an LLM.
    """
    try:
        if type(text)==str:
            remove_markdown(text)
            data = json.loads(text)
        else:
            data = text
        tool_name = data.get("tool_name")
        arguments = data.get("arguments", {})
        
        tool_response = await call_tool(tool_name, arguments)
        if auto_append:
            append_to_history("tool", await tool_response, history)
        return tool_response
    except json.JSONDecodeError:
        return {
            "success": False,
            "error": "Error: Invalid JSON format. Please provide a valid JSON input."
        }
        
from typing import Union

def append_to_history(role: str, content: dict | str, history: list) -> list:
    """
    Append a message to the chat history.

    Parameters:
      role: Role of the message sender (user, assistant, system, tool).
      content: Message content (text or structured data).
      history: List of message dictionaries.
    """
    message = ""
    if role=="tool":
        role="user"
        message+="Tool Output:\n"
    
    if type(content)==dict and content.keys() == {"text"}:    
        content = content["text"]
    
    if type(content)==dict:
        message+=json.dumps(content)
    else:
        message+=content
    
    history.append({"role": role, "content": message})
    
    return history
    
def initialize_message_history(system_prompt: str) -> list:
    """
    Initialize the message history with a system prompt.

    Parameters:
      system_prompt: The initial system prompt.

    Returns:
      A list containing the initial system prompt.
    """
    return [{"role": "system", "content": generate_system_prompt(system_prompt)}]

async def generate_response(client:openai.OpenAI, model_name:str, message_history:list, auto_append:bool=True, chain:bool=False) -> str:
    """
    Generate a response using the specified model and message history.

    Parameters:
        client: The OpenAI client instance.
        model_name: The name of the model to use for generating the response.
        message_history: List of message dictionaries.
        auto_append: Whether to automatically append the response to the history.
        chain: Whether to continue generating responses until a text response is obtained.

    Returns:
        The generated response text.
    """
    print("\n\t\t", message_history[1:])
    # Execute the API request with error handling and standard error codes
    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=message_history
        )
        output=remove_markdown(response.choices[0].message.content)
        if len(output)==0:
            return False
        output = json.loads(output)
        
        if auto_append:
            append_to_history("assistant",output,message_history)
        
        if chain:
            while not list(output.keys()) == ["text"]:
                if "tool_name" in output.keys():
                    await call_tool_from_json(output, message_history)
                output2 = await generate_response(client, model_name, message_history, True, True)
                if output2==output:
                    return output
                if list(output.keys()) == ["text"]:
                    return output2
                output=output2
        return output
    except openai.RateLimitError as e:
        output = "Error:429 - Rate limit exceeded. Please try again later."
    except openai.APIError as e:
        output = "Error:500 - API returned an error. Please check your request."
    except openai.Timeout as e:
        output = "Error:504 - Request timed out."
    except openai.ServiceUnavailableError as e:
        output = "Error:503 - Service is currently unavailable. Please try again later."
    except openai.APIConnectionError as e:
        output = "Error:503 - Failed to connect to the API. Please check your network connection."
    except openai.InvalidRequestError as e:
        output = "Error:400 - Invalid request parameters. Please verify your input."
    except openai.AuthenticationError as e:
        output = "Error:401 - Authentication failed. Please check your API key."
    except openai.PermissionError as e:
        output = "Error:403 - You do not have permission to access this model."
    except Exception as e:
        output = "Error:500 - An unexpected error occurred."
    
    return output
    
    
    