from pydantic import BaseModel
from typing import List, Dict, Any


class QueryData(BaseModel):
    user_id: str
    conversation_id: str
    query: str

class GetCollectionsData(BaseModel):
    user_id: str

class GetCollectionData(BaseModel):
    collection_name: str
    page: int
    pageSize: int

class NERData(BaseModel):
    text: str

class TitleData(BaseModel):
    text: str