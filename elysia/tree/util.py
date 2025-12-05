import uuid
import json
from typing import AsyncGenerator

# dspy requires a 'base' LM but this should not be used
import dspy
from dspy.streaming import StreamResponse

import types
from pydantic import BaseModel, ConfigDict

from typing import Any, Optional

from weaviate.util import generate_uuid5
from weaviate.classes.query import MetadataQuery, Sort

from elysia.objects import (
    Response,
    Result,
    Error,
    Text,
    Tool,
    Update,
    StreamedReasoning,
)

from elysia.tree.objects import TreeData
from elysia.util.objects import (
    TrainingUpdate,
    EdgeUpdate,
    FewShotExamples,
    ViewEnvironment,
    TreeNode,
)

from elysia.util.modules import ElysiaPrompt, AssertedModule
from elysia.util.parsing import format_datetime, estimate_tokens
from elysia.util.client import ClientManager
from elysia.tree.prompt_templates import (
    FollowUpSuggestionsPrompt,
    TitleCreatorPrompt,
    DPWithEnv,
    DPWithEnvMetadata,
    DPWithEnvMetadataResponse,
)
from elysia.tools.text.prompt_templates import TextResponsePrompt


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
        text_response = ElysiaPrompt(
            TextResponsePrompt,
            tree_data=tree_data,
            environment_level="dynamic",
            collection_schemas="none",
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

    def to_tree_node(self) -> TreeNode:
        return TreeNode(
            id=self.id,
            name=self.name,
            is_branch=self.branch,
            description=self.description,
            instruction=self.instruction,
        )

    def _get_view_environment(self) -> dict:
        return {
            "name": "view_environment",
            "description": (
                "An auxiliary tool to inspect the current environment containing all objects from this session. "
                "Use this tool when you need to reference existing data before making decisions or selecting other actions. "
                "This tool does NOT complete the user's request immediately, it only helps you gather information to inform your next action choice or inputs. "
                "After viewing the environment, you can select another tool after selecting this tool from available_actions. "
                "Retrieved objects from previous tool calls are stored here. "
                "Optionally input tool_names to filter by specific tools' results. "
                "Optionally input metadata_keys and metadata_values alongside tool_names to filter by specific metadata criteria. "
            ),
            "inputs": [
                {
                    "name": "tool_names",
                    "description": (
                        "The name(s) of the tool to view the environment for. Top level key for the environment. "
                        "If not provided or None, view the entire environment."
                    ),
                    "default": None,
                    "type": list[str],
                    "required": False,
                },
                {
                    "name": "metadata_keys",
                    "description": (
                        "A key of the metadata to view the environment for. Subkey for the environment. "
                        "If not provided or None, view all objects for the given tool_name. "
                        "If provided metadata_key, must provide metadata_value. "
                        "If giving more than one, the position of the key must match the position of the corresponding tool_name/metadata_value it is associated with"
                    ),
                    "default": None,
                    "type": list[str],
                    "required": False,
                },
                {
                    "name": "metadata_values",
                    "description": (
                        "A value of the metadata to view the environment for. Subkey for the environment. "
                        "If not provided or None, view all objects for the given tool_name."
                        "If provided metadata_value, must provide metadata_key. "
                        "If giving more than one, the position of the value must match the position of the corresponding tool_name/metadata_key it is associated with"
                    ),
                    "default": None,
                    "type": list[str],
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

        # any extra inputs are removed
        real_input_names = [inp.name for inp in real_inputs]
        llm_inputs = {
            input_name: input_value
            for input_name, input_value in llm_inputs.items()
            if input_name in real_input_names
        }

        return llm_inputs

    def _tool_assertion(self, pred: dspy.Prediction, kwargs: dict) -> tuple[bool, str]:
        available_option_names = [
            action["name"] for action in kwargs["available_actions"]
        ]

        # Custom logic if view environment is called incorrectly
        if pred.function_name == "view_environment":
            inputs = pred.function_inputs
            has_keys = "metadata_keys" in inputs
            has_values = "metadata_values" in inputs

            # Check if only one is provided
            if has_keys and not has_values:
                return (
                    False,
                    "If providing metadata_keys to view_environment, must also provide metadata_values.",
                )
            if has_values and not has_keys:
                return (
                    False,
                    "If providing metadata_values to view_environment, must also provide metadata_keys.",
                )

            # Check if both are provided but one is None
            if has_keys and has_values:
                keys = inputs["metadata_keys"]
                values = inputs["metadata_values"]

                if keys is not None and values is None:
                    return (
                        False,
                        "If providing metadata_keys to view_environment, must also provide metadata_values.",
                    )
                if keys is None and values is not None:
                    return (
                        False,
                        "If providing metadata_values to view_environment, must also provide metadata_keys.",
                    )

                # Check if lengths match when both are not None
                if keys is not None and values is not None and len(keys) != len(values):
                    return (
                        False,
                        (
                            "If providing metadata_keys and metadata_values to view_environment, "
                            "the number of keys must match the number of values. "
                            f"Number of keys: {len(keys)}, "
                            f"Number of values: {len(values)}"
                        ),
                    )
        return (
            pred.function_name in available_option_names,
            f"You picked the action `{pred.function_name}` - that is not in `available_actions`! "
            f"Your output MUST be one of the following: {available_option_names}",
        )

    async def _execute_view_environment(
        self,
        kwargs: dict,
        tree_data: TreeData,
        inputs: dict,
        base_lm: dspy.LM,
        complex_lm: dspy.LM,
        client_manager: ClientManager,
    ) -> AsyncGenerator[
        StreamedReasoning | dspy.Prediction | ViewEnvironment | list, None
    ]:

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
            tool_names = function_inputs["tool_names"]
            metadata_keys = function_inputs["metadata_keys"]
            metadata_values = function_inputs["metadata_values"]

            environment = tree_data.environment._view(
                tool_names, metadata_keys, metadata_values, False
            )

            # Build preview of first 5 objects for ViewEnvironment object
            first_key = (
                tool_names[0] if tool_names and len(tool_names) > 0 else None
            ) or (list(environment.keys())[0] if environment else None)
            preview_items = environment.get(first_key, []) if first_key else []
            preview = [
                item["objects"] for item in preview_items[:5] if isinstance(item, dict)
            ]

            yield ViewEnvironment(
                tool_names=tool_names,
                metadata_keys=metadata_keys,
                metadata_values=metadata_values,
                environment_preview=preview,
            )

            environment_decision_executor = ElysiaPrompt(
                DPWithEnvMetadataResponse,
                tree_data=tree_data,
                collection_schemas=(
                    "summaries" if tree_data.use_weaviate_collections else "none"
                ),
                tasks_completed=True,
                message_update=True,
                reasoning=tree_data.settings.BASE_USE_REASONING,
                environment_level="none",
            )

            if tree_data.streaming:

                if tree_data.settings.USE_FEEDBACK:
                    aforward_fn = (
                        environment_decision_executor.aforward_streaming_with_feedback_examples
                    )

                else:
                    aforward_fn = environment_decision_executor.aforward_streaming

                async for chunk in aforward_fn(
                    streamed_field="reasoning",
                    environment=environment,
                    available_actions=kwargs["available_actions"],
                    history=history,
                    lm=base_lm,
                    add_tree_data_inputs=False,
                    base_lm=base_lm,
                    complex_lm=complex_lm,
                    client_manager=client_manager,
                    feedback_model="decision",
                ):
                    if (
                        isinstance(chunk, StreamResponse)
                        and chunk.signature_field_name == "reasoning"
                    ):
                        yield StreamedReasoning(
                            chunk=chunk.chunk, last=chunk.is_last_chunk
                        )
                    elif isinstance(chunk, dspy.Prediction):
                        pred = chunk
                    elif isinstance(chunk, list):
                        yield chunk
            else:
                if tree_data.settings.USE_FEEDBACK:
                    pred, uuids = (
                        await environment_decision_executor.aforward_with_feedback_examples(
                            feedback_model="decision",
                            client_manager=client_manager,
                            streamed_field="reasoning",
                            environment=environment,
                            available_actions=kwargs["available_actions"],
                            history=history,
                            base_lm=base_lm,
                            complex_lm=complex_lm,
                            add_tree_data_inputs=False,
                        )
                    )
                    yield uuids
                else:
                    pred = await environment_decision_executor.aforward(
                        streamed_field="reasoning",
                        environment=environment,
                        available_actions=kwargs["available_actions"],
                        history=history,
                        lm=base_lm,
                        add_tree_data_inputs=False,
                    )

            chosen_function = pred.function_name
            inputs = pred.function_inputs
            i += 1

            kwargs = {
                "environment": environment,
                "available_actions": kwargs["available_actions"],
                **pred,
            }

        yield pred

    def _build_decision_kwargs(
        self,
        tree_data: TreeData,
        available_options: list[dict],
        unavailable_options: list[dict],
        base_lm: dspy.LM,
        complex_lm: dspy.LM,
        client_manager: ClientManager,
        feedback: bool,
    ) -> dict:

        if feedback:
            return {
                "instruction": self.instruction,
                "tree_count": tree_data.tree_count_string(),
                "environment_metadata": tree_data.environment.output_llm_metadata(),
                "available_actions": available_options,
                "unavailable_actions": unavailable_options,
                "base_lm": base_lm,
                "complex_lm": complex_lm,
                "client_manager": client_manager,
                "feedback_model": "decision",
            }
        else:
            return {
                "instruction": self.instruction,
                "tree_count": tree_data.tree_count_string(),
                "environment_metadata": tree_data.environment.output_llm_metadata(),
                "available_actions": available_options,
                "unavailable_actions": unavailable_options,
                "lm": base_lm,
            }

    def _build_training_inputs(
        self,
        tree_data: TreeData,
        available_options: list[dict],
        unavailable_options: list[dict],
        decision_executor,
    ) -> dict:
        return {
            "instruction": self.instruction,
            "tree_count": tree_data.tree_count_string(),
            "environment_metadata": tree_data.environment.output_llm_metadata(),
            "available_actions": available_options,
            "unavailable_actions": unavailable_options,
            **decision_executor.module._add_tree_data_inputs({}),  # type: ignore
        }

    def _process_stream_chunk(self, chunk) -> tuple[Any | None, dspy.Prediction | None]:
        if (
            isinstance(chunk, StreamResponse)
            and chunk.signature_field_name == "reasoning"
        ):
            return StreamedReasoning(chunk=chunk.chunk, last=chunk.is_last_chunk), None
        elif isinstance(chunk, dspy.Prediction):
            return None, chunk
        elif isinstance(chunk, list):
            return FewShotExamples(chunk), None
        return None, None

    async def decide(
        self,
        tree_data: TreeData,
        base_lm: dspy.LM,
        complex_lm: dspy.LM,
        options: list[ToolOption],
        client_manager: ClientManager | None = None,
    ) -> AsyncGenerator[
        Decision
        | Result
        | Update
        | Text
        | Error
        | TrainingUpdate
        | EdgeUpdate
        | StreamedReasoning
        | ViewEnvironment,
        None,
    ]:
        if client_manager is None:
            client_manager = ClientManager(settings=tree_data.settings)

        # Build option lists
        available_options = [
            {
                "name": opt.name,
                "description": opt.description,
                "inputs": [o.model_dump() for o in opt.inputs],
            }
            for opt in options
            if opt.available
        ]
        unavailable_options = [
            {"name": opt.name, "available_at": opt.unavailable_reason}
            for opt in options
            if not opt.available and opt.unavailable_reason
        ]

        # Fast path: single option with no inputs
        if len(available_options) == 1:
            only_option = next(opt for opt in options if opt.available)
            if not only_option.inputs:
                yield Decision(
                    function_name=only_option.name,
                    function_inputs={},
                    reasoning="Only one option available, no inputs required.",
                    end_actions=False,
                )
                return

        # Determine signature based on environment size
        env_token_limit_reached = (
            estimate_tokens(json.dumps(tree_data.environment._unhidden_to_json()))
            > tree_data.env_token_limit
        )
        if env_token_limit_reached:
            available_options.append(self._get_view_environment())
            signature = DPWithEnvMetadata
        else:
            signature = DPWithEnv

        decision_executor = AssertedModule(
            ElysiaPrompt(
                signature,
                tree_data=tree_data,
                collection_schemas=(
                    "summaries" if tree_data.use_weaviate_collections else "none"
                ),
                tasks_completed=True,
                message_update=False,
                impossible=False,
                reasoning=tree_data.settings.BASE_USE_REASONING,
                environment="none" if env_token_limit_reached else "full",
            ),
            self._tool_assertion,
        )

        kwargs = self._build_decision_kwargs(
            tree_data,
            available_options,
            unavailable_options,
            base_lm,
            complex_lm,
            client_manager,
            tree_data.settings.USE_FEEDBACK,
        )

        pred = None
        if tree_data.streaming:
            aforward_fn = (
                decision_executor.aforward_streaming_with_feedback_examples
                if tree_data.settings.USE_FEEDBACK
                else decision_executor.aforward_streaming
            )
            async for chunk in aforward_fn(streamed_field="reasoning", **kwargs):
                yield_val, pred_val = self._process_stream_chunk(chunk)
                if yield_val is not None:
                    yield yield_val
                if pred_val is not None:
                    pred = pred_val
        else:
            if tree_data.settings.USE_FEEDBACK:
                pred, uuids = await decision_executor.aforward_with_feedback_examples(
                    **kwargs
                )
                yield FewShotExamples(uuids)
            else:
                pred = await decision_executor.aforward(**kwargs)

        if pred is None:
            raise ValueError("No prediction was returned from decision executor")

        training_inputs = self._build_training_inputs(
            tree_data, available_options, unavailable_options, decision_executor
        )
        yield TrainingUpdate(
            module_name="decision",
            inputs=training_inputs,
            outputs=dict(pred.__dict__["_store"]),
        )

        if pred.function_name == "view_environment":
            view_kwargs = {**training_inputs, **dict(pred)}
            async for chunk in self._execute_view_environment(
                kwargs=view_kwargs,
                tree_data=tree_data,
                inputs=pred.function_inputs,
                base_lm=base_lm,
                complex_lm=complex_lm,
                client_manager=client_manager,
            ):
                if (
                    isinstance(chunk, StreamResponse)
                    and chunk.signature_field_name == "reasoning"
                ):
                    yield StreamedReasoning(chunk=chunk.chunk, last=chunk.is_last_chunk)
                elif isinstance(chunk, ViewEnvironment):
                    yield chunk
                elif isinstance(chunk, dspy.Prediction):
                    pred = chunk
                elif isinstance(chunk, list):
                    yield FewShotExamples(chunk)

        target_option = next(opt for opt in options if pred.function_name == opt.name)
        yield Decision(
            function_name=pred.function_name,
            function_inputs=self._get_function_inputs(
                pred.function_inputs, target_option.inputs
            ),
            reasoning=pred.reasoning,
            end_actions=pred.end_actions,
        )


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
        result: Result | EdgeUpdate | Update | Text | Error | StreamedReasoning,
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
        environment=tree_data.environment.toon,
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
