from pydantic import BaseModel
from typing import List, Dict, Any, Optional


class QueryData(BaseModel):
    user_id: str
    conversation_id: str
    query_id: str
    query: str
    route: Optional[str] = None
    mimick: Optional[bool] = False
    
class GetCollectionsData(BaseModel):
    user_id: str

class GetCollectionData(BaseModel):
    collection_name: str
    page: int
    pageSize: int

class InitialiseTreeData(BaseModel):
    user_id: str
    conversation_id: str

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

class ObjectRelevanceData(BaseModel):
    user_id: str
    conversation_id: str
    query_id: str
    objects: list[dict]

class ProcessCollectionData(BaseModel):
    user_id: str
    collection_name: str
    force: bool
    lm: Optional[str] = "claude-3-5-haiku-20241022"

class DebugData(BaseModel):
    conversation_id: str
    user_id: str