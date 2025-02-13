# Set matplotlib backend before any matplotlib imports
import matplotlib
matplotlib.use('Agg')

from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from datetime import datetime
import traceback
import os
from dotenv import load_dotenv
from helper import generateGraph
import agentic_ai
from openai import OpenAI

# Load environment variables from a .env file
load_dotenv()

# ------------------------------------------------------------------------------
# Ensure you have set the appropriate API key in your environment variables.
# ------------------------------------------------------------------------------
LLM_API_KEY = os.getenv("LLM_API_KEY")
LLM_API_URL = os.getenv("LLM_API_URL", "https://openrouter.ai/api/v1")
CHAT_MODEL = os.getenv("CHAT_MODEL")
CODE_MODEL = os.getenv("CODE_MODEL")


client = OpenAI(api_key=LLM_API_KEY, base_url=LLM_API_URL)




app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Register graph generation tool with chat agent
@agentic_ai.register_tool(
    description="Generates all kinds of 2D and 3D graphs using matplotlib based on the query.",
    arguments=(
        "query: string; A description of the graph to be generated.\n"
        "style (optional): string; The style of the graph to be generated.\n"
        "data (optional): dict; Additional data need to generate the graph."
    ),
    response=(
        "success: boolean; Indicates if the graph was generated successfully.\n"
        "image_id: string; The ID of the generated graph image.\n"
        "error: string; Error message if the graph generation failed."
    ),
    example_in='{"query":"Bar chart of sales data","style":"ggplot","data":{"sales":[100,200,300]}}',
    example_out='{"success": true, "image_id": "image_id"}'
)
async def generate_graph_tool(query: str, style: str = None, data: dict = None) -> dict:
  
    print(f"Starting graph generation for: {query}")
    result = await generateGraph(query, style, data)
    print(f"Graph generation result: {result}")
    return result

@app.route('/generate_response', methods=['POST'])
async def generate_response_api():
    req_id = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    print(f"Request {req_id}: Received new request")
    
    query = request.json.get('query')
    if not query:
        print(f"Request {req_id}: Missing query parameter")
        return jsonify({"error": "Query parameter is missing"}), 400
    
    print(f"Request {req_id}: Processing query: {query}")
    
    try:
        agentic_ai.append_to_history("user", query, chat_history)
        response = await agentic_ai.generate_response(client, CHAT_MODEL, chat_history, chain=True)
        if isinstance(response, dict) and "text" in response:
            response_text = response["text"].strip()
        else:
            return {"success": False, "messages": [{"type": 'text', "content": "An error occurred while processing your request"}]}, 500
        
        # Parse <image> tags to extract graph IDs
        result_parts = []
        text_parts = response_text.split('<image>')
        
        # Handle first text part
        if text_parts[0]:
            result_parts.append({"type":'text', "content":text_parts[0]})
        
        # Handle remaining parts (alternating between graph and text)
        for part in text_parts[1:]:
            if '</image>' in part:
                graph_id, remaining_text = part.split('</image>', 1)
                result_parts.append({"type":'graph', "content":graph_id})
                if remaining_text:
                    result_parts.append({"type":'text', "content":remaining_text})
            else:
                result_parts.append({"type":'text', "content":part})
        
        print(f"Request {req_id}: Successfully processed request")
        return {"success": True, "messages": result_parts}
        
    except Exception as e:
        error_msg = f"Error processing request: {str(e)}"
        print(f"Request {req_id}: {error_msg}\n{traceback.format_exc()}")
        return {"success": False, "messages": [{"type":'text', "content":error_msg}], "error":str(e)}, 500

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

@app.errorhandler(Exception)
def handle_error(error):
    print(f"Unhandled error: {str(error)}\n{traceback.format_exc()}")
    return jsonify({
        "error": "An unexpected error occurred",
        "details": str(error) if app.debug else "Please check server logs for details"
    }), 500
    

chat_history=agentic_ai.initialize_message_history("You are a helpful AI assistant called GraphBot that can generate graphs and provide textual responses. "
        "Not every response needs a graph - only generate graphs when they add value to the response. "
        "You can only respond in textual format and dont have the ability to generate images yourself. Use suitable tools for generating whatever is required. "
        "To respond with any image, mention it using the format <image>imageID</image> in the text response."
    )



if __name__ == "__main__":
    print("Starting server with Agg backend...")
    app.run(debug=True, port=5001)