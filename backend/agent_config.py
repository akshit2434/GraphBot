from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel
import os
import dotenv
import logging

logger = logging.getLogger(__name__)

# Load environment variables
dotenv.load_dotenv()
if not os.getenv('OPENROUTER_API_KEY'):
    logger.error("OPENROUTER_API_KEY not found in environment variables")
    raise ValueError("OPENROUTER_API_KEY is required")

# Initialize the LLM models
try:
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
except Exception as e:
    logger.error(f"Failed to initialize OpenAI models: {str(e)}")
    raise

# Create the Agent with a system prompt
main_agent = Agent(
    model,
    system_prompt=(
        "You are a helpful AI assistant that can generate graphs and provide textual responses. "
        "When a user's request involves data visualization, use the 'generate_graph_tool' to create graphs. "
        "You can provide text explanations before and after the graph. "
        "Not every response needs a graph - only generate graphs when they add value to the response. "
        "\nWhen you want to include a graph, use the following syntax in your response: "
        "<graph>description of what graph to generate</graph> "
        "\nExample responses:"
        "\n1. Text only: 'The weather today is sunny.'"
        "\n2. Graph with context: 'Let me show you the temperature trend.\n"
        "<graph>Create a line graph of temperature over time</graph>\n"
        "As you can see, the temperature peaks at noon.'"
    )
)

graph_agent = Agent(
    graph_model,
    system_prompt=(
        "You are a Python code generator for matplotlib graphs. Generate clean, minimal code that: "
        "1. Uses only matplotlib.pyplot and numpy "
        "2. Sets appropriate labels and titles "
        "3. Uses a clear style and color scheme "
        "4. Uses the current axes (plt.gca()) for all plotting "
        "5. Does NOT create or close figures "
        "6. Properly scales axes and sets limits "
        "7. For 3D plots, use methods like plot3D(), scatter3D(), or set_zlabel() directly"
        "8. For polar plots, just use polar-specific methods without setting projection"
        "\nExample for a sine wave:\n"
        "import matplotlib.pyplot as plt\nimport numpy as np\n"
        "x = np.linspace(0, 10, 100)\n"
        "y = np.sin(x)\n"
        "ax.plot(x, y, 'b-')\n"
        "ax.set_title('Sine Wave')\n"
        "ax.set_xlabel('X')\n"
        "ax.set_ylabel('sin(x)')\n"
        "\nExample for a 3D plot:\n"
        "import matplotlib.pyplot as plt\nimport numpy as np\n"
        "x = np.linspace(-5, 5, 100)\n"
        "y = np.linspace(-5, 5, 100)\n"
        "X, Y = np.meshgrid(x, y)\n"
        "Z = np.sin(np.sqrt(X**2 + Y**2))\n"
        "ax.plot_surface(X, Y, Z)\n"
        "ax.set_xlabel('X')\n"
        "ax.set_ylabel('Y')\n"
        "ax.set_zlabel('Z')"
    )
)