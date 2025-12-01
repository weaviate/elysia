from math import e
import uuid
import json
from typing import AsyncGenerator, overload

# dspy requires a 'base' LM but this should not be used
import dspy
from dspy.streaming import StreamResponse

import types
from pydantic import BaseModel, ConfigDict

from typing import Callable, Union, Any, Optional, Sequence

from weaviate.util import generate_uuid5
from weaviate.classes.query import MetadataQuery, Sort, Filter

from elysia.objects import (
    Response,
    Result,
    Error,
    Text,
    Tool,
    Update,
    Status,
    StreamedReasoning,
)

from elysia.tree.objects import TreeData
from elysia.util.objects import (
    TrainingUpdate,
    TreeUpdate,
    FewShotExamples,
    ViewEnvironment,
)

from elysia.util.elysia_modules import ElysiaChainOfThought, AssertedModule
from elysia.util.parsing import format_datetime
from elysia.util.client import ClientManager
from elysia.tree.prompt_templates import (
    FollowUpSuggestionsPrompt,
    TitleCreatorPrompt,
    DPWithEnv,
    DPWithEnvMetadata,
    DPWithEnvMetadataResponse,
)
from elysia.tools.text.prompt_templates import TextResponsePrompt
from elysia.util.retrieve_feedback import retrieve_feedback


class ForcedTextResponse(Tool):
    """
    A tool that creates a new text response via a new LLM call.
    This is used in the decision tree when the tree reaches the end of its process, but no text is being displayed to the user.
    Then this tool is automaticaly called to create a new text response.
    """

    def __init__(self, **kwargs):
        super().__init__(
            name="final_text_response",
            description="",
            status="Writing response...",
            inputs={},
            end=True,
        )

    async def __call__(
        self,
        tree_data: TreeData,
        inputs: dict,
        base_lm: dspy.LM,
        complex_lm: dspy.LM,
        client_manager: ClientManager | None = None,
        **kwargs,
    ):
        text_response = ElysiaChainOfThought(
            TextResponsePrompt,
            tree_data=tree_data,
            environment=True,
            tasks_completed=True,
            message_update=False,
        )

        output = await text_response.aforward(
            lm=base_lm,
        )

        yield Response(text=output.response)


class Decision:
    """
    Simple decision object to store the decision made by the decision node.
    """

    def __init__(
        self,
        function_name: str,
        function_inputs: dict,
        reasoning: str,
        end_actions: bool,
        last_in_tree: bool = False,
    ):
        self.function_name = function_name
        self.function_inputs = function_inputs
        self.reasoning = reasoning
        self.end_actions = end_actions


class ToolInput(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    name: str
    description: str
    type: type | types.GenericAlias
    default: Any
    required: Optional[bool] = False


class ToolOption(BaseModel):
    name: str
    available: bool
    description: str
    inputs: list[ToolInput]
    unavailable_reason: Optional[str] = None


class Node:
    def __init__(
        self,
        id: str,
        name: str,
        branch: bool,
        root: bool,
        options: list[str] = [],
        end: bool = False,
        status: str = "",
        instruction: str = "",
        description: str = "",
    ):
        self.id = id
        self.name = name
        self.branch = branch
        self.root = root
        self.options = options
        self.end = end
        self.status = status
        self.instruction = instruction
        self.description = description

        if self.branch and self.end:
            raise ValueError("A branch cannot be at the end of the tree.")

    def to_json(self):
        return {
            "id": self.id,
            "name": self.name,
            "branch": self.branch,
            "root": self.root,
            "options": self.options,
            "end": self.end,
            "status": self.status,
            "instruction": self.instruction,
            "description": self.description,
        }

    def _get_view_environment(self) -> dict:
        return {
            "name": "view_environment",
            "description": (
                "An auxiliary tool to inspect the current environment containing all objects from this session. "
                "Use this tool when you need to reference existing data before making decisions or selecting other actions. "
                "This tool does NOT complete the user's request immediately, it only helps you gather information to inform your next action choice or inputs. "
                "After viewing the environment, you can select another tool after selecting this tool from available_actions. "
                "Retrieved objects from previous tool calls are stored here. "
                "Optionally input tool_name to filter by a specific tool's results. "
                "Optionally input metadata_key and metadata_value alongside tool_name to filter by specific metadata criteria. "
            ),
            "inputs": [
                {
                    "name": "tool_name",
                    "description": (
                        "The name of the tool to view the environment for. Top level key for the environment. "
                        "If not provided or None, view the entire environment."
                    ),
                    "default": None,
                    "type": str,
                    "required": False,
                },
                {
                    "name": "metadata_key",
                    "description": (
                        "A key of the metadata to view the environment for. Subkey for the environment. "
                        "If not provided or None, view all objects for the given tool_name. "
                        "If provided metadata_key, must provide metadata_value. "
                    ),
                    "default": None,
                    "type": str,
                    "required": False,
                },
                {
                    "name": "metadata_value",
                    "description": (
                        "A value of the metadata to view the environment for. Subkey for the environment. "
                        "If not provided or None, view all objects for the given tool_name."
                        "If provided metadata_value, must provide metadata_key. "
                    ),
                    "default": None,
                    "type": str,
                    "required": False,
                },
            ],
        }

    def _get_function_inputs(
        self, llm_inputs: dict[str, Any], real_inputs: list[ToolInput]
    ) -> dict[str, Any]:

        # if the inputs match the 'schema' of keys: description, type, default, value, then take the value
        for input_name in llm_inputs:
            if (
                isinstance(llm_inputs[input_name], dict)
                and "value" in llm_inputs[input_name]
            ):
                llm_inputs[input_name] = llm_inputs[input_name]["value"]

        # any non-provided inputs are set to the default
        default_inputs = {value.name: value.default for value in real_inputs}
        for default_input_name in default_inputs:
            if default_input_name not in llm_inputs:
                llm_inputs[default_input_name] = default_inputs[default_input_name]

        return llm_inputs

    def _tool_assertion(self, pred: dspy.Prediction, kwargs: dict) -> tuple[bool, str]:
        available_option_names = [
            action["name"] for action in kwargs["available_actions"]
        ]
        return (
            pred.function_name in available_option_names,
            f"You picked the action `{pred.function_name}` - that is not in `available_actions`! "
            f"Your output MUST be one of the following: {available_option_names}",
        )

    async def _execute_view_environment(
        self, kwargs: dict, tree_data: TreeData, inputs: dict, lm: dspy.LM
    ) -> AsyncGenerator[StreamedReasoning | dspy.Prediction | ViewEnvironment, None]:

        history = dspy.History(messages=[])
        view_env_inputs = self._get_view_environment()["inputs"]

        chosen_function = "view_environment"
        max_env_views = 3
        i = 0

        while chosen_function == "view_environment" and i < max_env_views:

            history.messages.append({**kwargs})

            function_inputs = self._get_function_inputs(
                llm_inputs=inputs,
                real_inputs=[
                    ToolInput(
                        name=i.get("name", ""),
                        description=i.get("description", ""),
                        type=i.get("type", Any),
                        default=i.get("default", None),
                        required=i.get("required", False),
                    )
                    for i in view_env_inputs
                ],
            )
            tool_name = function_inputs["tool_name"]
            metadata_key = function_inputs["metadata_key"]
            metadata_value = function_inputs["metadata_value"]

            if (
                metadata_key
                and metadata_value
                and tool_name
                and tool_name in tree_data.environment.environment
            ):
                environment = (
                    tree_data.environment.get_objects(
                        tool_name=tool_name,
                        metadata_key=metadata_key,
                        metadata_value=metadata_value,
                    )
                    or tree_data.environment.to_json()["environment"]
                )
            elif tool_name and tool_name in tree_data.environment.environment:
                environment = [
                    {
                        "metadata": item.metadata,
                        "objects": item.objects,
                    }
                    for item in tree_data.environment.get(tool_name) or []
                ]
            else:
                environment = tree_data.environment.to_json()["environment"]

            yield ViewEnvironment(
                tool_name=tool_name,
                metadata_key=metadata_key,
                metadata_value=metadata_value,
                environment_preview=environment[:5],
            )

            environment_decision_executor = ElysiaChainOfThought(
                DPWithEnvMetadataResponse,
                tree_data=tree_data,
                collection_schemas=tree_data.use_weaviate_collections,
                tasks_completed=True,
                message_update=True,
                reasoning=tree_data.settings.BASE_USE_REASONING,
            )

            async for chunk in environment_decision_executor.aforward_streaming(
                streamed_field="reasoning",
                environment=environment,
                available_actions=kwargs["available_actions"],
                history=history,
                lm=lm,
                add_tree_data_inputs=False,
            ):
                if (
                    isinstance(chunk, StreamResponse)
                    and chunk.signature_field_name == "reasoning"
                ):
                    yield StreamedReasoning(chunk=chunk.chunk, last=chunk.is_last_chunk)
                elif isinstance(chunk, dspy.Prediction):
                    pred = chunk

            chosen_function = pred.function_name
            inputs = pred.function_inputs
            i += 1

            kwargs = {
                "environment": environment,
                "available_actions": kwargs["available_actions"],
                **pred,
            }

        yield pred

    async def decide(
        self,
        tree_data: TreeData,
        base_lm: dspy.LM,
        complex_lm: dspy.LM,
        options: list[ToolOption],
        client_manager: ClientManager | None = None,
    ) -> AsyncGenerator[
        Decision
        | Sequence[Result | Update | Text | Error | TrainingUpdate | TreeUpdate]
        | StreamedReasoning
        | ViewEnvironment,
        None,
    ]:

        available_options = [
            {
                "name": option.name,
                "description": option.description,
                "inputs": [o.model_dump() for o in option.inputs],
            }
            for option in options
            if option.available
        ]
        unavailable_options = [
            {
                "name": option.name,
                "available_at": option.unavailable_reason,
            }
            for option in options
            if not option.available and option.unavailable_reason != ""
        ]

        if len(available_options) == 1:
            only_option = next(option for option in options if option.available)
            if only_option.inputs == []:
                yield Decision(
                    function_name=only_option.name,
                    function_inputs={},
                    reasoning="Only one option available, no inputs required.",
                    end_actions=False,
                )
                yield []
                return

        # TODO: replace this with actual token counting
        env_token_limit_reached = (
            len(json.dumps(tree_data.environment._unhidden_to_json()))
            > tree_data.env_token_limit
        )
        if env_token_limit_reached:
            available_options.append(self._get_view_environment())
            signature = DPWithEnvMetadata
        else:
            signature = DPWithEnv

        decision_executor = ElysiaChainOfThought(
            signature,
            tree_data=tree_data,
            collection_schemas=tree_data.use_weaviate_collections,
            tasks_completed=True,
            message_update=False,
            impossible=False,
            reasoning=tree_data.settings.BASE_USE_REASONING,
            environment=not env_token_limit_reached,
        )
        decision_executor = AssertedModule(
            decision_executor,
            self._tool_assertion,
        )

        if tree_data.streaming:
            async for chunk in decision_executor.aforward_streaming(
                streamed_field="reasoning",
                instruction=self.instruction,
                tree_count=tree_data.tree_count_string(),
                environment_metadata=tree_data.environment.output_llm_metadata(),
                available_actions=available_options,
                unavailable_actions=unavailable_options,
                lm=base_lm,
            ):
                if (
                    isinstance(chunk, StreamResponse)
                    and chunk.signature_field_name == "reasoning"
                ):
                    yield StreamedReasoning(chunk=chunk.chunk, last=chunk.is_last_chunk)
                elif isinstance(chunk, dspy.Prediction):
                    pred = chunk
        else:
            pred = await decision_executor.aforward(
                instruction=self.instruction,
                tree_count=tree_data.tree_count_string(),
                environment_metadata=tree_data.environment.output_llm_metadata(),
                available_actions=available_options,
                unavailable_actions=unavailable_options,
                lm=base_lm,
            )

        results = [
            TrainingUpdate(
                module_name="decision",
                inputs={
                    "instruction": self.instruction,
                    "tree_count": tree_data.tree_count_string(),
                    "environment_metadata": tree_data.environment.output_llm_metadata(),
                    "available_actions": available_options,
                    "unavailable_actions": unavailable_options,
                    **decision_executor.module._add_tree_data_inputs({}),  # type: ignore
                },
                outputs={k: v for k, v in pred.__dict__["_store"].items()},
            )
        ]
        if pred.function_name == "view_environment":
            async for chunk in self._execute_view_environment(
                kwargs={
                    "user_prompt": tree_data.user_prompt,
                    "instruction": self.instruction,
                    "tree_count": tree_data.tree_count_string(),
                    "environment_metadata": tree_data.environment.output_llm_metadata(),
                    "available_actions": available_options,
                    "unavailable_actions": unavailable_options,
                    **decision_executor.module._add_tree_data_inputs({}),  # type: ignore
                    **pred,
                },
                tree_data=tree_data,
                inputs=pred.function_inputs,
                lm=base_lm,
            ):
                if isinstance(chunk, (StreamResponse, ViewEnvironment)):
                    if isinstance(chunk, StreamResponse):
                        if chunk.signature_field_name == "reasoning":
                            yield StreamedReasoning(
                                chunk=chunk.chunk, last=chunk.is_last_chunk
                            )
                    else:
                        yield chunk
                elif isinstance(chunk, dspy.Prediction):
                    pred = chunk

        yield Decision(
            function_name=pred.function_name,
            function_inputs=self._get_function_inputs(
                llm_inputs=pred.function_inputs,
                real_inputs=next(
                    option for option in options if pred.function_name == option.name
                ).inputs,
            ),
            reasoning=pred.reasoning,
            end_actions=pred.end_actions,
        )
        yield results


class TreeReturner:
    """
    Class to parse the output of the tree to the frontend.
    """

    def __init__(
        self,
        user_id: str,
        conversation_id: str,
        tree_index: int = 0,
    ):
        self.user_id = user_id
        self.conversation_id = conversation_id
        self.tree_index = tree_index
        self.store = []

    def set_tree_index(self, tree_index: int):
        self.tree_index = tree_index

    def clear_store(self):
        self.store = []

    def add_prompt(self, prompt: str, query_id: str):
        self.store.append(
            {
                "type": "user_prompt",
                "id": str(uuid.uuid4()),
                "user_id": self.user_id,
                "conversation_id": self.conversation_id,
                "query_id": query_id,
                "payload": {
                    "prompt": prompt,
                },
            }
        )

    async def __call__(
        self,
        result: Result | TreeUpdate | Update | Text | Error | StreamedReasoning,
        query_id: str,
    ) -> dict[str, str | dict] | None:

        if isinstance(result, (Update, Text, Result, Error)):
            payload = await result.to_frontend(
                self.user_id, self.conversation_id, query_id
            )
            self.store.append(payload)
            return payload

        elif isinstance(result, StreamedReasoning):
            payload = await result.to_frontend(
                self.user_id, self.conversation_id, query_id
            )
            return payload

        elif isinstance(result, TreeUpdate):
            payload = await result.to_frontend(
                self.user_id,
                self.conversation_id,
                query_id,
                self.tree_index,
            )
            self.store.append(payload)
            return payload


async def create_conversation_title(conversation: list[dict], lm: dspy.LM):
    title_creator = dspy.Predict(TitleCreatorPrompt)
    title = await title_creator.aforward(
        conversation=conversation,
        lm=lm,
    )
    return title.title


async def get_follow_up_suggestions(
    tree_data: TreeData,
    current_suggestions: list[str],
    lm: dspy.LM,
    num_suggestions: int = 2,
    context: str | None = None,
):

    # load dspy model for suggestor
    follow_up_suggestor = dspy.Predict(FollowUpSuggestionsPrompt)

    if context is None:
        context = (
            "System context: You are an agentic RAG service querying or aggregating information from Weaviate collections via filtering, "
            "sorting, multi-collection queries, summary statistics. "
            "Uses tree-based approach with specialized agents and decision nodes for dynamic information retrieval, "
            "summary generation, and real-time results display while maintaining natural conversation flow. "
            "Create questions that are natural follow-ups to the user's prompt, "
            "which they may find interesting or create relevant insights to the already retrieved data. "
            "Or, questions which span across other collections, but are still relevant to the user's prompt."
        )

    # get prediction
    prediction = await follow_up_suggestor.aforward(
        user_prompt=tree_data.user_prompt,
        reference=tree_data.atlas.datetime_reference,
        conversation_history=tree_data.conversation_history,
        environment=tree_data.environment.environment,
        data_information=tree_data.output_collection_metadata(with_mappings=False),
        old_suggestions=current_suggestions,
        context=context,
        num_suggestions=num_suggestions,
        lm=lm,
    )

    return prediction.suggestions


async def get_saved_trees_weaviate(
    collection_name: str,
    user_id: str,
    client_manager: ClientManager | None = None,
):
    """
    Get all saved trees from a Weaviate collection.

    Args:
        collection_name (str): The name of the collection to get the trees from.
        user_id (str): The user ID to get the trees for.
        client_manager (ClientManager): The client manager to use.
            If not provided, a new ClientManager will be created from environment variables.

    Returns:
        dict: A dictionary of tree UUIDs and their titles.
    """
    if client_manager is None:
        client_manager = ClientManager()
        close_after_use = True
    else:
        close_after_use = False

    async with client_manager.connect_to_async_client() as client:

        if not await client.collections.exists(collection_name):
            return {}

        collection = client.collections.get(collection_name)
        if not await collection.tenants.exists(user_id):
            return {}
        user_collection = collection.with_tenant(user_id)

        len_collection = (
            await user_collection.aggregate.over_all(
                total_count=True,
            )
        ).total_count

        response = await user_collection.query.fetch_objects(
            limit=len_collection,
            sort=Sort.by_update_time(ascending=False),
            return_metadata=MetadataQuery(last_update_time=True),
        )

    if close_after_use:
        await client_manager.close_clients()

    trees = {
        obj.properties["conversation_id"]: {
            "title": obj.properties["title"],
            "last_update_time": format_datetime(obj.metadata.last_update_time),
        }
        for obj in response.objects
    }

    return trees


async def delete_tree_from_weaviate(
    user_id: str,
    conversation_id: str,
    collection_name: str,
    client_manager: ClientManager | None = None,
):
    """
    Delete a tree from a Weaviate collection.

    Args:
        user_id (str): The user ID to delete the tree from.
        conversation_id (str): The conversation ID of the tree to delete.
        collection_name (str): The name of the collection to delete the tree from.
        client_manager (ClientManager): The client manager to use.
            If not provided, a new ClientManager will be created from environment variables.
    """

    if client_manager is None:
        client_manager = ClientManager()

    async with client_manager.connect_to_async_client() as client:
        collection = client.collections.get(collection_name)
        if not await collection.tenants.exists(user_id):
            return

        user_collection = collection.with_tenant(user_id)
        uuid = generate_uuid5(conversation_id)
        if await user_collection.data.exists(uuid):
            await user_collection.data.delete_by_id(uuid)
