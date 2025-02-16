import matplotlib
matplotlib.use('Agg')

from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
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
import secrets
import traceback

dotenv.load_dotenv()

app = Flask(__name__)
CORS(app, supports_credentials=True)
app.config['SECRET_KEY'] = secrets.token_hex(32)

# Dictionary to store chat histories for each session
chat_histories = {}
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
class DataSchema(BaseModel):
    """Schema for the data parameter"""
    sales: list[float] = Field(None, description="Sample sales data")
    x: list[float] = Field(None, description="X-axis data")
    y: list[float] = Field(None, description="Y-axis data")
    z: list[float] = Field(None, description="Z-axis data for 3D plots")
    categories: list[str] = Field(None, description="Categories for categorical plots")
    values: list[float] = Field(None, description="Values for the categories")

class generate_graph_tool(BaseModel):
    """A smart tool that generates all kinds of 2D and 3D graphs using matplotlib based on the query.
    
    Example input: '{"query":"Bar chart of sales data","style":"ggplot","data":{"sales":[100,200,300]}}'
    """
    query: str = Field(..., description="The query for the graph in simple English, can be vague or specific")
    style: str = Field(None, description="Custom styles for the graph")
    data: DataSchema = Field(None, description="The data required for the graph")

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

# Initialize system prompt for new sessions
SYSTEM_PROMPT = ("You are a helpful AI assistant called GraphBot that can generate graphs and provide textual responses. "
        "Not every response needs a graph - only generate graphs when they add value to the response. "
        "You can only respond in textual format and dont have the ability to generate images yourself. Use suitable tools for generating whatever is required. "
        "To respond with any image, mention it using the format <image>imageID</image> in the text response.")

def get_chat_history(session_id: str):
    """Get or create chat history for a user session"""
    if session_id not in chat_histories:
        chat_histories[session_id] = [SystemMessage(SYSTEM_PROMPT)]
    return chat_histories[session_id]

@app.route('/get_session', methods=['GET'])
def get_session():
    """Generate a new session ID for the client"""
    session_id = secrets.token_urlsafe(32)
    return jsonify({"session_id": session_id})

@app.route('/generate_response', methods=['POST'])
def generate_response_api():
    req_id = base64.urlsafe_b64encode(os.urandom(12)).decode('ascii')
    print(f"Request {req_id}: Received new request")
    
    # Require session_id in request
    session_id = request.json.get('session_id')
    if not session_id:
        return jsonify({"error": "Session ID is required"}), 400
        
    query = request.json.get('query')
    if not query:
        print(f"Request {req_id}: Missing query parameter")
        return jsonify({"error": "Query parameter is missing"}), 400
    
    try:
        # Get chat history for this session
        messages = get_chat_history(session_id)
        messages.append(HumanMessage(query))
        
        def debug_print_messages(messages, prefix=""):
            print(f"\n{prefix} Message History:")
            for idx, msg in enumerate(messages):
                print(f"{idx}. Type: {type(msg).__name__}")
                print(f"   Content: {msg.content}")
                if hasattr(msg, 'tool_call_id'):
                    print(f"   Tool Call ID: {msg.tool_call_id}")
                print()

        # Create a copy of the LLM with bound tools for this conversation
        conversation_llm = llm.bind_tools(tools)
        # debug_print_messages(messages, "Initial")
        
        conversation_state = {
            "needs_tool_response": False,
            "last_tool_calls": []
        }

        while True:
            try:
                # Generate AI response
                print("\nInvoking LLM with messages...")
                ai_msg = conversation_llm.invoke(messages)
                print(f"LLM Response: {ai_msg.content}")
                
                # Check for tool calls
                tool_calls = getattr(ai_msg, "tool_calls", [])
                
                # If we have tool calls
                if tool_calls:
                    print(f"\nFound {len(tool_calls)} tool calls")
                    conversation_state["needs_tool_response"] = True
                    conversation_state["last_tool_calls"] = tool_calls
                    
                    # Process all tool calls before next LLM invocation
                    tool_responses = []
                    for call in tool_calls:
                        try:
                            print(f"\nProcessing tool call: {call}")
                            tool_output = generate_graph.invoke(call)
                            print(f"Tool output: {tool_output}")
                            tool_responses.append({
                                "content": str(tool_output),
                                "tool_call_id": call["id"]
                            })
                        except Exception as e:
                            print(f"Error calling tool: {e}")
                            return jsonify({
                                "success": False,
                                "messages": [{"type": "text", "content": f"Error generating graph: {str(e)}"}]
                            }), 500
                    
                    # Add AI message and all tool responses together
                    messages.append(ai_msg)
                    for tool_response in tool_responses:
                        messages.append(ToolMessage(**tool_response))
                    
                    # debug_print_messages(messages, "After tool calls")
                    conversation_state["needs_tool_response"] = False
                    
                # If no tool calls and we don't need tool responses
                elif not conversation_state["needs_tool_response"]:
                    print("\nNo tool calls, finalizing response")
                    messages.append(ai_msg)
                    response_text = ai_msg.content
                    # debug_print_messages(messages, "Final")
                    break
                        
            except Exception as e:
                print(f"Error in LLM conversation: {e}")
                print(f"Current message history:")
                # debug_print_messages(messages, "Error state")
                return jsonify({
                    "success": False,
                    "messages": [{"type": "text", "content": f"Error in conversation: {str(e)}"}]
                }), 500

        # Now parse the final response to extract text and graph references
        result_parts = []
        text_parts = response_text.split('<image>')
        
        # Handle first text part
        if text_parts[0].strip():
            result_parts.append({"type": "text", "content": text_parts[0].strip()})
        
        # Handle remaining parts (alternating between graph and text)
        for part in text_parts[1:]:
            if '</image>' in part:
                graph_id, remaining_text = part.split('</image>', 1)
                result_parts.append({"type": "graph", "content": graph_id})
                if remaining_text.strip():
                    result_parts.append({"type": "text", "content": remaining_text.strip()})
            else:
                if part.strip():
                    result_parts.append({"type": "text", "content": part.strip()})

        return jsonify({"success": True, "messages": result_parts})
        
    except Exception as e:
        error_msg = f"Error processing request: {str(e)}"
        print(f"Request {req_id}: {error_msg}\n{traceback.format_exc()}")
        return jsonify({
            "success": False,
            "messages": [{"type": "text", "content": error_msg}],
            "error": str(e)
        }), 500

@app.route('/generated_graphs/<image_id>.png')
def get_graph_image(image_id):
    try:
        print(f"Serving graph image: {image_id}")
        return send_file(f'generated_graphs/{image_id}.png', mimetype='image/png')
    except FileNotFoundError:
        print(f"Graph image not found: {image_id}")
        return jsonify({"error": "Image not found"}), 404
    except Exception as e:
        print(f"Error serving graph image {image_id}: {str(e)}")
        return jsonify({"error": "Internal server error"}), 500

if __name__ == "__main__":
    app.run(debug=True, port=5001)