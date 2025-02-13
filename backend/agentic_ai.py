import functools

# Global dictionary to store tools
TOOLS = {}

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
            "use_tool": {
                "tool_name": "Name of the tool",
                "arguments": {
                    // Arguments in JSON format
                }
            }
        }
        
        Example tool call:
        {
            "use_tool": {
                "tool_name": "get_food_details_tool",
                "arguments": {
                    "query": "A sweet, soft-textured dish",
                    "region": "Indian"
                }
            }
        }
        """
        "If you want to respond to the user, simply provide the text response."
    )

    return system_prompt

def call_tool(tool_name: str, args: dict) -> dict:
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
    expected_args = [arg.split(":")[0].strip() for arg in tool_metadata["arguments"].split("\n")]
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
        result = tool_func(**args)
        return {
            "success": True,
            "tool": tool_name,
            "output": result
        }
    except Exception as e:
        return {
            "success": False,
            "message": f"An error occurred while executing '{tool_name}': {str(e)}"
        }
