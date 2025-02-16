
from langchain_openai import ChatOpenAI
import os
from getpass import getpass
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
import matplotlib.pyplot as plt
import base64
import re
from langchain_core.tools import tool
from pydantic import BaseModel, Field
import io
import numpy as np
import dotenv

dotenv.load_dotenv()

if "OPENROUTER_API_KEY" not in os.environ:
    print("Enter Openrouter API Key: ")
    os.environ["OPENROUTER_API_KEY"] = getpass()

if "OPENROUTER_BASE_URL" not in os.environ:
    print("Enter Openrouter Base URL: ")
    os.environ["OPENROUTER_BASE_URL"] = getpass()


def get_openrouter(model: str = "openai/gpt-4o") -> ChatOpenAI:
    return ChatOpenAI(model=model,
        openai_api_key=os.getenv("OPENROUTER_API_KEY"),
        openai_api_base=os.getenv("OPENROUTER_BASE_URL"))
    
llm = get_openrouter(model="google/gemini-2.0-flash-001")

def sanitize_code(code: str) -> str:
    """Basic security check for the generated code"""
    if any(x in code for x in ['eval', 'exec', 'open', '__import__']):
        raise RuntimeError("Potentially dangerous code detected")
    
    allowed = ['matplotlib', 'numpy', 'mpl_toolkits']
    if 'import' in code and not any(x in code for x in allowed):
        raise RuntimeError("Only matplotlib and numpy imports are allowed")
        
    return code

def remove_markdown(text: str) -> str:
    """Remove markdown syntax from the text."""
    # Remove markdown code block syntax if present
    text = re.sub(r'^```[a-zA-Z]*\n*', '', text)
    text = re.sub(r'\n*```$', '', text)
    text = text.strip()
    return text

# --- Define Tool and Helper Function ---
class generate_graph_tool(BaseModel):
    """A smart tool that generates all kinds of 2D and 3D graphs using matplotlib based on the query.
    
    Example input: '{"query":"Bar chart of sales data","style":"ggplot","data":{"sales":[100,200,300]}}'
    """
    query: str = Field(..., description="The query for the graph in simple English, can be vague or specific")
    style: str = Field(None, description="Custom styles for the graph")
    data: dict = Field(None, description="The data required for the graph")

def generateGraph(query: str, style: str = None, data: dict = None) -> str:
    """Generates matplotlib code and returns result."""
    # Create graph LLM code generation instance
    graph_llm_context = SystemMessage(
        content=(
            "You are a Python code generator for matplotlib graphs. Create complete, working code that:"
            "\n1. Uses matplotlib and numpy for plotting"
            "\n2. For 3D plots, uses mpl_toolkits.mplot3d"
            "\n3. Creates and configures the figure with appropriate size"
            "\n4. Includes proper labels, titles, and scales"
            "\n5. Returns only the Python code, no explanations"
        )
    )

    messages = [
        graph_llm_context,
        HumanMessage(
            content=(
                f"Create a complete matplotlib visualization for: {query}\n"
                "The variables 'plt', 'np', and 'fig' are already defined."
            )
        )
    ]
    if data:
        messages.append(HumanMessage(content=f"Use this data: {data}"))
    if style:
        messages.append(HumanMessage(content=f"Apply this style: {style}"))
    
    response = llm.invoke(messages)
    code = sanitize_code(remove_markdown(response.content.strip()))

     # Setup execution environment with all necessary objects
    namespace = {
      'plt': plt,
      'np': __import__('numpy'),
      'Axes3D': __import__('mpl_toolkits.mplot3d').mplot3d.Axes3D,
      'fig': plt.figure()
    }
    
    # Execute the code
    try:
      exec(code, namespace)
    except Exception as e:
      raise RuntimeError(f"Failed to execute graph code: {str(e)}")

    # Get the figure from namespace
    fig = namespace.get('fig') or plt.gcf()
        
    # Save the figure to a buffer
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    buf.seek(0)
    
    # Generate unique identifier for this graph
    image_id = base64.urlsafe_b64encode(os.urandom(12)).decode('ascii')
    
    # Save the image to a file using the image_id
    os.makedirs('generated_graphs', exist_ok=True)
    with open(f'generated_graphs/{image_id}.png', 'wb') as f:
        f.write(buf.getvalue())
    
    buf.close()
    plt.close(fig)
    
    return {"success": True, "image_id": image_id}

@tool
def generate_graph(query: str, style: str = None, data: dict = None) -> dict:
    """Generates a graph based on the provided parameters."""
    result = generateGraph(query, style, data)
    return result

tools = [generate_graph_tool]

llm_with_tools = llm.bind_tools(tools)

messages=[SystemMessage("You are a helpful AI assistant called GraphBot that can generate graphs and provide textual responses. "
        "Not every response needs a graph - only generate graphs when they add value to the response. "
        "You can only respond in textual format and dont have the ability to generate images yourself. Use suitable tools for generating whatever is required. "
        "To respond with any image, mention it using the format <image>imageID</image> in the text response.")]

user_input = input("\tYour Response: ")
if user_input.lower() == "exit":
    exit()
messages.append(HumanMessage(user_input))

while True:
    ai_msg = llm_with_tools.invoke(messages)
    messages.append(ai_msg)
    # print(ai_msg, getattr(ai_msg['additional_kwargs'].keys(), "additional_kwargs", []),'\n\n11\n\n', getattr(ai_msg, "tool_calls", []),'\n\n22\n\n', getattr(getattr(ai_msg, "additional_kwargs", []), "tool_calls", []))
    # Debugging: Check if tool_calls exist
    tool_calls = dict(ai_msg)['additional_kwargs'].get("tool_calls", [])
    print("Tool calls found:", tool_calls)

    # Process tool calls if any exist
    for call in tool_calls:
        try:
            tool_output = generate_graph.invoke(call)
            tool_response = ToolMessage(content=tool_output, tool_call_id=call["id"])
            messages.append(tool_response)
        except Exception as e:
            print(f"Error calling tool: {e}")

    # If no tool was invoked, print response and ask for user input
    if not tool_calls:
        print(ai_msg)
        user_input = input("\tYour Response: ")
        if user_input.lower() == "exit":
            for message in messages:
                message.pretty_print()
            break
        messages.append(HumanMessage(user_input))
