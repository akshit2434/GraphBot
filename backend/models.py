from pydantic import BaseModel, Field
from typing import Optional, List

class GraphResponse(BaseModel):
    success: bool
    image_id: str = Field(default="", description="Unique identifier for the generated graph")
    error: Optional[str] = None

class MessagePart(BaseModel):
    type: str  # 'text' or 'graph'
    content: str  # text content or image_id for graphs

class BotResponse(BaseModel):
    success: bool
    messages: List[MessagePart]
    error: Optional[str] = None