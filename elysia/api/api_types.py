from typing import Any, Dict, List, Optional, Literal
from typing_extensions import Self

from pydantic import BaseModel, Field, field_validator, model_validator

from uuid import uuid4


class QueryData(BaseModel):
    user_id: str
    conversation_id: str
    query_id: str
    preset_id: Optional[str] = None
    query: str
    collection_names: list[str]
    route: Optional[str] = ""
    mimick: Optional[bool] = False


class ViewPaginatedCollectionData(BaseModel):
    page_size: int
    page_number: int
    query: str = ""
    sort_on: Optional[str] = None
    ascending: bool = False
    filter_config: dict[str, Any] = {}


class InitialiseTreeData(BaseModel):
    low_memory: bool = False


class MetadataNamedVectorData(BaseModel):
    name: str
    enabled: Optional[bool] = None
    description: Optional[str] = None


class MetadataFieldData(BaseModel):
    name: str
    description: str


class UpdateCollectionMetadataData(BaseModel):
    named_vectors: Optional[List[MetadataNamedVectorData]] = None
    summary: Optional[str] = None
    mappings: Optional[dict[str, dict[str, str]]] = None
    fields: Optional[List[MetadataFieldData]] = None


class NERData(BaseModel):
    text: str


class TitleData(BaseModel):
    user_id: str
    conversation_id: str
    text: str


class ObjectRelevanceData(BaseModel):
    user_id: str
    conversation_id: str
    query_id: str
    objects: list[dict]


class ProcessCollectionData(BaseModel):
    user_id: str
    collection_name: str


class DebugData(BaseModel):
    conversation_id: str
    user_id: str


class AddFeedbackData(BaseModel):
    user_id: str
    conversation_id: str
    query_id: str
    feedback: int


class RemoveFeedbackData(BaseModel):
    user_id: str
    conversation_id: str
    query_id: str


class GetUserRequestsData(BaseModel):
    user_id: str


class InstantReplyData(BaseModel):
    user_id: str
    user_prompt: str


class FollowUpSuggestionsData(BaseModel):
    user_id: str
    conversation_id: str


# class BackendConfig(BaseModel):
#     settings: dict[str, Any]
#     style: str
#     agent_description: str
#     end_goal: str
#     branch_initialisation: str


class SaveConfigUserData(BaseModel):
    name: str
    config: dict[str, Any]
    frontend_config: dict[str, Any]
    default: bool


class SaveConfigTreeData(BaseModel):
    settings: Optional[dict[str, Any]] = None
    style: Optional[str] = None
    agent_description: Optional[str] = None
    end_goal: Optional[str] = None
    branch_initialisation: Optional[str] = None


class UpdateFrontendConfigData(BaseModel):
    config: dict[str, Any]


class AvailableModelsData(BaseModel):
    user_id: str


class ToolItem(BaseModel):
    instance_id: Optional[str] = None
    name: str
    from_branch: str
    from_tools: list[str]
    is_branch: bool

    @field_validator("instance_id")
    def validate_instance_id(self, instance_id: Optional[str]) -> Optional[str]:
        if instance_id is None:
            return str(uuid4())
        return instance_id


class BranchInfo(BaseModel):
    name: str
    is_root: bool
    description: str
    instruction: str


class ToolPreset(BaseModel):
    preset_id: str
    name: str
    order: list[ToolItem]
    branches: list[BranchInfo]
    default: bool = Field(default=False)

    @field_validator("branches")
    def validate_branches(self, branches: list[BranchInfo]) -> list[BranchInfo]:
        if len(branches) != len(set(branch.name for branch in branches)):
            raise ValueError("Branch names must be unique")

        return branches

    @model_validator(mode="after")
    def validate_branch_tools_have_info(self) -> Self:
        branch_names = {branch.name for branch in self.branches}
        branch_tool_names = {tool.name for tool in self.order if tool.is_branch}
        missing_branches = branch_tool_names - branch_names
        if missing_branches:
            raise ValueError(
                f"Branch tools {missing_branches} in order do not have corresponding info in branches"
            )

        return self
