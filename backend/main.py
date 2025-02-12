# Set matplotlib backend before any matplotlib imports
import matplotlib
matplotlib.use('Agg')

from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from datetime import datetime
import traceback
import os
import dotenv
from pydantic_ai import Agent, RunContext
from pydantic_ai.models.openai import OpenAIModel

from logger_config import logger
from models import BotResponse, MessagePart
from helper import generateGraph

# Load environment variables
dotenv.load_dotenv()

# Configure OpenAI models
model = OpenAIModel(
    os.getenv('CHAT_MODEL'),
    base_url='https://openrouter.ai/api/v1',
    api_key=os.getenv('OPENROUTER_API_KEY'),
)

graph_model = OpenAIModel(
    os.getenv('CODE_MODEL'),
    base_url='https://openrouter.ai/api/v1',
    api_key=os.getenv('OPENROUTER_API_KEY'),
)

# Create the chat agent
chat_agent = Agent(
    model,
    system_prompt=(
        "You are a helpful AI assistant called GraphBot that can generate graphs and provide textual responses. "
        "Not every response needs a graph - only generate graphs when they add value to the response. "
        "You can only respond in textual format and dont have the ability to generate images yourself. Use suitable tools for generating whatever is required. "
        "To respond with any image, mention it using the format <image>imageID</image> in the text response."
    ),
)

# Create the graph agent for code generation
graph_agent = Agent(
    graph_model,
    system_prompt=(
        "You are a Python code generator for matplotlib graphs. Generate clean, minimal code that: "
        "1. Supported librarier: matplotlib; numpy; You cannot import or use any other libraries; "
        "2. Sets appropriate labels and titles; "
        "3. Uses a clear style and color scheme; "
        "4. Uses the current axes (plt.gca()) for all plotting; "
        "5. Does NOT create or close figures; "
        "6. Properly scales axes and sets limits; "
        "7. For 3D plots, use methods like plot3D(), scatter3D(), or set_zlabel() directly; "
        "8. For polar plots, use polar-specific methods without setting projection."
        "\nReturn only the Python code with no additional statements or other info. Ensure the labels are visible properly and not overlapping. Make code minimal with no unnecessary lines."
    )
)

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Register graph generation tool with chat agent
@chat_agent.tool
async def generate_graph_tool(ctx: RunContext[str], query: str, style:str = None, data: dict = None) -> dict:
    """
    Even if the query is vague, if the user is asking for a graph regardless, this can generate a graph. "
    This tool can generate all kinds of 2D and 3D graphs using matplotlib. It takes the following arguments:
    
    Args:
        query (str): A description of the graph to be generated.
        style (str, optional): The style of the graph to be generated.
        data (dict, optional): Additional data required for generating the graph.
    
    Returns:
        dict: A dictionary containing the result of the graph generation, including the path to the generated graph image. (Ex. {success: True, image_id: 'image_id'}, {success: False, error: 'error_message'})
    """
    
    logger.info(f"Generating graph with query: {query}")
    result = await generateGraph(query, style, data)
    logger.info(f"Graph generation result: {result}")
    return result

@app.route('/generate_response', methods=['POST'])
async def generate_response_api():
    req_id = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    logger.info(f"Request {req_id}: Received new request")
    
    query = request.json.get('query')
    if not query:
        logger.warning(f"Request {req_id}: Missing query parameter")
        return jsonify({"error": "Query parameter is missing"}), 400
    
    logger.info(f"Request {req_id}: Processing query: {query}")
    
    try:
        # Get response from chat agent
        response = await chat_agent.run(query)
        response_text = response.data
        
        # Parse <image> tags to extract graph IDs
        result_parts = []
        text_parts = response_text.split('<image>')
        
        # Handle first text part
        if text_parts[0]:
            result_parts.append(MessagePart(type='text', content=text_parts[0]))
        
        # Handle remaining parts (alternating between graph and text)
        for part in text_parts[1:]:
            if '</image>' in part:
                graph_id, remaining_text = part.split('</image>', 1)
                result_parts.append(MessagePart(type='graph', content=graph_id))
                if remaining_text:
                    result_parts.append(MessagePart(type='text', content=remaining_text))
            else:
                result_parts.append(MessagePart(type='text', content=part))
        
        logger.info(f"Request {req_id}: Successfully processed request")
        return jsonify(BotResponse(success=True, messages=result_parts).dict())
        
    except Exception as e:
        error_msg = f"Error processing request: {str(e)}"
        logger.error(f"Request {req_id}: {error_msg}\n{traceback.format_exc()}")
        return jsonify(BotResponse(
            success=False,
            messages=[MessagePart(type='text', content=error_msg)],
            error=str(e)
        ).model_dump()), 500

@app.route('/generated_graphs/<image_id>.png')
def get_graph_image(image_id):
    try:
        logger.info(f"Serving graph image: {image_id}")
        return send_file(f'generated_graphs/{image_id}.png', mimetype='image/png')
    except FileNotFoundError:
        logger.error(f"Graph image not found: {image_id}")
        return jsonify({"error": "Image not found"}), 404
    except Exception as e:
        logger.error(f"Error serving graph image {image_id}: {str(e)}")
        return jsonify({"error": "Internal server error"}), 500

@app.errorhandler(Exception)
def handle_error(error):
    logger.error(f"Unhandled error: {str(error)}\n{traceback.format_exc()}")
    return jsonify({
        "error": "An unexpected error occurred",
        "details": str(error) if app.debug else "Please check server logs for details"
    }), 500

if __name__ == "__main__":
    logger.info("Starting server with Agg backend...")
    app.run(debug=True, port=5001)