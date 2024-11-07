from pydantic import BaseModel
from typing import List, Dict, Any


class ProcessData(BaseModel):
    user_prompt: str

class GetCollectionData(BaseModel):
    collection_name: str
    page: int
    pageSize: int