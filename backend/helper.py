import matplotlib.pyplot as plt
import io
import base64
import os
import re
import logging
from models import GraphResponse


logger = logging.getLogger(__name__)


def sanitize_code(code: str) -> str:
    """Sanitize and validate the generated Python code"""
    try:
        forbidden_imports = ['os', 'sys', 'subprocess', 'eval', 'exec']
        forbidden_commands = ['plt.figure', 'plt.close', 'plt.show']
        code_lines = code.split('\n')
        
        code_lines = [line for line in code_lines if not any(cmd in line for cmd in forbidden_commands)]
        
        for line in code_lines:
            for forbidden in forbidden_imports:
                if f"import {forbidden}" in line or f"from {forbidden}" in line:
                    logger.warning(f"Forbidden import detected: {forbidden}")
                    raise ValueError(f"Forbidden import detected: {forbidden}")
                    
            if any(dangerous_func in line for dangerous_func in ['eval', 'exec', 'open']):
                logger.warning("Potentially dangerous function call detected")
                raise ValueError("Potentially dangerous function calls detected")
        
        return "\n".join(code_lines)
    except Exception as e:
        logger.error(f"Code sanitization failed: {str(e)}")
        raise

async def generateGraph(query,style, data):
    fig = None
    try:
        print(f"Generating graph for query: {query}")
        # Generate matplotlib code based on query
        prompt = f"Create a matplotlib graph for: {query}."
        if data:
            prompt += f"\nData: {data}"
        if style:
            prompt += f"\nStyle: {style}"
        prompt += "\nReturn only the Python code with no additional statements or other info. Ensure the labels are visible properly and not overlapping. Make code minimal with no unnecessary lines."
        
        from main import graph_agent
        response = await graph_agent.run(prompt)
        generated_code = response.data.strip()

        # Remove markdown code block syntax if present
        generated_code = re.sub(r'^```[a-zA-Z]*\n*', '', generated_code)
        generated_code = re.sub(r'\n*```$', '', generated_code)
        generated_code = generated_code.strip()

        # Validate and sanitize the code
        generated_code = sanitize_code(generated_code)
        
        try:
            # Determine plot type from code
            needs_3d = (
                'set_zlabel' in generated_code or
                'plot3D' in generated_code or
                'scatter3D' in generated_code or
                re.search(r'projection=[\'"]3d[\'"]', generated_code)
            )
            needs_polar = re.search(r'projection=[\'"]polar[\'"]', generated_code)
            
            # Create figure with appropriate subplot
            fig = plt.figure(figsize=(10, 6))
            if needs_3d:
                ax = fig.add_subplot(111, projection='3d')
                generated_code = re.sub(r'plt\.gca\([^)]*\)', 'ax', generated_code)
            elif needs_polar:
                ax = fig.add_subplot(111, projection='polar')
                generated_code = re.sub(r'plt\.gca\([^)]*\)', 'ax', generated_code)
            else:
                ax = fig.add_subplot(111)
                generated_code = re.sub(r'plt\.gca\([^)]*\)', 'ax', generated_code)
                
            # Execute validated code in a restricted namespace
            namespace = {
                'plt': plt,
                'np': __import__('numpy'),
                'ax': ax
            }
            exec(generated_code, namespace)
            
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
            return GraphResponse(success=True, image_id=image_id)
            
        except Exception as e:
            logger.error(f"Error executing graph code: {str(e)}")
            return GraphResponse(success=False, error=f"Failed to generate graph: {str(e)}")

    except Exception as e:
        logger.error(f"Error in graph generation: {str(e)}")
        return GraphResponse(success=False, error=str(e))
        
    finally:
        # Clean up the figure
        if fig is not None:
            plt.close(fig)