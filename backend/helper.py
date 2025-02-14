import matplotlib.pyplot as plt
import io
import base64
import os
import re
import logging
import agentic_ai
from dotenv import load_dotenv
from openai import OpenAI
from agentic_ai import remove_markdown

# Load environment variables
load_dotenv()
LLM_API_KEY = os.getenv("LLM_API_KEY")
LLM_API_URL = os.getenv("LLM_API_URL", "https://openrouter.ai/api/v1")
CODE_MODEL = os.getenv("CODE_MODEL")

# Initialize OpenAI client
client = OpenAI(api_key=LLM_API_KEY, base_url=LLM_API_URL)

logger = logging.getLogger(__name__)

def sanitize_code(code: str) -> str:
    """Basic security check for the generated code"""
    if any(x in code for x in ['eval', 'exec', 'open', '__import__']):
        raise RuntimeError("Potentially dangerous code detected")
    
    allowed = ['matplotlib', 'numpy', 'mpl_toolkits']
    if 'import' in code and not any(x in code for x in allowed):
        raise RuntimeError("Only matplotlib and numpy imports are allowed")
        
    return code

async def generateGraph(query, style=None, data=None):
    fig = None
    try:
        # Create graph LLM code generation instance
        system_prompt = (
            "You are a Python code generator for matplotlib graphs. Create complete, working code that:"
            "\n1. Uses matplotlib and numpy for plotting"
            "\n2. For 3D plots, uses mpl_toolkits.mplot3d"
            "\n3. Creates and configures the figure with appropriate size"
            "\n4. Includes proper labels, titles, and scales"
            "\n5. Returns only the Python code, no explanations"
        )
        
        prompt = (
            f"Create a complete matplotlib visualization for: {query}\n"
            "The variables 'plt', 'np', and 'fig' are already defined."
        )
        if data:
            prompt += f"\nUse this data: {data}"
        if style:
            prompt += f"\nApply this style: {style}"
        
        graph_history = agentic_ai.initialize_message_history(system_prompt, tool_instructions=[])
        agentic_ai.append_to_history("user", prompt, graph_history)
        response = await agentic_ai.generate_response(client, CODE_MODEL, graph_history, False)
        
        if isinstance(response, dict) and "text" in response:
            generated_code = remove_markdown(response["text"].strip())

        # Validate and sanitize code
        generated_code = sanitize_code(generated_code)

        # Setup execution environment with all necessary objects
        namespace = {
            'plt': plt,
            'np': __import__('numpy'),
            'Axes3D': __import__('mpl_toolkits.mplot3d').mplot3d.Axes3D,
            'fig': plt.figure()
        }
        
        # Execute the code
        try:
            exec(generated_code, namespace)
        except Exception as e:
            logger.error(f"Error executing graph code: {str(e)}")
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
        
        logger.info(f"Successfully generated graph with ID: {image_id}")
        return {"success": True, "image_id": image_id}

    except (RuntimeError, ValueError) as e:
        logger.error(f"Graph generation error: {str(e)}")
        return {"success": False, "error": str(e)}
    except Exception as e:
        logger.error(f"Error in graph generation: {str(e)}")
        return {"success":False, "error":str(e)}
        
    finally:
        # Clean up the figure
        if fig is not None:
            plt.close(fig)