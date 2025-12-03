from pydantic import BaseModel
from typing import Optional


class GenerateDesignResponse(BaseModel):
    status: str
    generated_image_url: str
    llm_prompt: str
    notes: Optional[str] = None
