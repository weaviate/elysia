from pydantic import BaseModel
from typing import List, Dict, Any


class ProcessData(BaseModel):
    user_prompt: str