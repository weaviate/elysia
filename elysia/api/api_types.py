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

class SetCollectionsData(BaseModel):
    collection_names: List[str]
    remove_data: bool
    conversation_id: str
    user_id: str

class GetObjectData(BaseModel):
    collection_name: str
    uuid: str