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
        forbidden_commands = ['plt.close', 'plt.show']  # Allow plt.figure for size control
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

def determine_figure_size(style=None, needs_3d=False, needs_polar=False):
    """Determine fallback figure size based on plot type and style"""
    if style and isinstance(style, dict) and 'figsize' in style:
        return style['figsize']
    
    # Default sizes based on plot type
    if needs_3d:
        return (10, 8)  # 3D plots need more height
    elif needs_polar:
        return (8, 8)  # Polar plots work best with equal dimensions
    return (10, 6)  # Default size for standard plots

def extract_figure_size(code: str) -> tuple[float, float] | None:
    """Extract figure size from matplotlib code if specified"""
    size_match = re.search(r'plt\.figure\(.*?figsize=\(([\d.]+),\s*([\d.]+)\)', code)
    if size_match:
        try:
            width = float(size_match.group(1))
            height = float(size_match.group(2))
            return (width, height)
        except (ValueError, IndexError):
            return None
    return None

async def generateGraph(query, style=None, data=None):
    fig = None
    try:
        print(f"Generating graph for query: {query}")
        # Generate matplotlib code based on query
        prompt = f"Create a matplotlib graph for: {query}. You may specify figure size using plt.figure(figsize=(width, height)) if needed for optimal visualization."
        if data:
            prompt += f"\nData: {data}"
        if style:
            prompt += f"\nStyle: {style}"
        
        from main import graph_agent
        response = await graph_agent.run(prompt)
        generated_code = response.data.strip()

        # Remove markdown code block syntax if present
        generated_code = re.sub(r'^```[a-zA-Z]*\n*', '', generated_code)
        generated_code = re.sub(r'\n*```$', '', generated_code)
        generated_code = generated_code.strip()

        # Extract figure size before sanitization
        figsize = extract_figure_size(generated_code)

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
            
            # Use extracted size or fallback
            final_figsize = figsize or determine_figure_size(style, needs_3d, needs_polar)
            
            # Create figure with appropriate subplot
            fig = plt.figure(figsize=final_figsize)
            
            # Remove any plt.figure calls from generated code since we handle it
            generated_code = re.sub(r'plt\.figure\([^)]*\)', '', generated_code)
            
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