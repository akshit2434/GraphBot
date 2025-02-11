import re
import logging
from typing import List
from models import MessagePart

logger = logging.getLogger(__name__)

def parse_response(response: str) -> List[MessagePart]:
    """Parse the agent's response into a list of text and graph parts"""
    try:
        parts = []
        # Split the response by graph tags
        segments = re.split(r'(<graph>.*?</graph>)', response, flags=re.DOTALL)
        
        for segment in segments:
            if segment.strip():
                if segment.startswith('<graph>') and segment.endswith('</graph>'):
                    # Extract graph description from tags
                    graph_desc = segment[7:-8].strip()
                    parts.append(MessagePart(type='graph', content=graph_desc))
                else:
                    parts.append(MessagePart(type='text', content=segment.strip()))
        
        return parts
    except Exception as e:
        logger.error(f"Error parsing response: {str(e)}")
        return [MessagePart(type='text', content=str(response))]