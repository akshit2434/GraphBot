# Set matplotlib backend before any matplotlib imports
import matplotlib
matplotlib.use('Agg')

from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from datetime import datetime
import traceback
import os

from logger_config import logger
from models import BotResponse, MessagePart
from graph_generator import generate_graph
from response_parser import parse_response
from agent_config import main_agent

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

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
        # Get response from main agent
        response = await main_agent.run(query)
        parsed_parts = parse_response(response.data)
        
        # Process each part and replace graph descriptions with actual graph IDs
        result_parts = []
        for idx, part in enumerate(parsed_parts):
            try:
                if part.type == 'text':
                    result_parts.append(part)
                else:  # type == 'graph'
                    logger.info(f"Request {req_id}: Generating graph {idx+1}")
                    graph_response = await generate_graph(part.content)
                    if graph_response.success:
                        result_parts.append(MessagePart(type='graph', content=graph_response.image_id))
                    else:
                        error_msg = f"Failed to generate graph: {graph_response.error}"
                        logger.error(f"Request {req_id}: {error_msg}")
                        result_parts.append(MessagePart(type='text', content=error_msg))
            except Exception as e:
                logger.error(f"Request {req_id}: Error processing part {idx}: {str(e)}")
                result_parts.append(MessagePart(type='text', content=f"Error: {str(e)}"))
        
        logger.info(f"Request {req_id}: Successfully processed request")
        return jsonify(BotResponse(success=True, messages=result_parts).dict())
        
    except Exception as e:
        error_msg = f"Error processing request: {str(e)}"
        logger.error(f"Request {req_id}: {error_msg}\n{traceback.format_exc()}")
        return jsonify(BotResponse(
            success=False,
            messages=[MessagePart(type='text', content=error_msg)],
            error=str(e)
        ).dict()), 500

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