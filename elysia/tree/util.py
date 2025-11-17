import uuid
import json

# dspy requires a 'base' LM but this should not be used
import dspy
from pympler import asizeof
from logging import Logger

from typing import Callable, Union

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
)

from elysia.tree.objects import TreeData
from elysia.util.objects import TrainingUpdate, TreeUpdate, FewShotExamples
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
        impossible: bool,
        end_actions: bool,
        last_in_tree: bool = False,
    ):
        self.function_name = function_name
        self.function_inputs = function_inputs
        self.reasoning = reasoning
        self.impossible = impossible
        self.end_actions = end_actions
        self.last_in_tree = last_in_tree


class DecisionNode:
    """
    A decision node is a node in the tree that makes a decision based on the available options.
    This class is essentially the executor of the decision node.
    """

    def __init__(
        self,
        id: str,
        instruction: str,
        options: dict[
            str, dict[str, Union[str, dict, bool, Tool, "DecisionNode", None]]
        ],
        root: bool = False,
        logger: Logger | None = None,
        use_elysia_collections: bool = True,
    ):
        self.id = id
        self.instruction = instruction
        self.options = options
        self.root = root
        self.logger = logger
        self.use_elysia_collections = use_elysia_collections

    def _get_options(self):
        return self.options

    def add_option(
        self,
        id: str,
        description: str,
        inputs: dict,
        action: Tool | None = None,
        end: bool = True,
        status: str = "",
        next: "DecisionNode | None" = None,
    ):
        if status == "":
            status = f"Running {id}..."

        self.options[id] = {
            "description": description,  # type: str
            "inputs": inputs,  # type: dict
            "action": action,  # type: Tool | None
            "end": end,  # type: bool
            "status": status,  # type: str
            "next": next,  # type: DecisionNode | None
        }

    def remove_option(self, id: str):
        if id in self.options:
            del self.options[id]

    def _options_to_json(self, available_tools: list[str]):
        """
        Options that get shown to the LLM.
        Remove any that are empty branches.
        """
        out = {}
        for node in self.options:
            if node not in available_tools:  # empty branch
                continue

            out[node] = {
                "function_name": node,
                "description": self.options[node]["description"],
            }

            if self.options[node]["inputs"] != {}:
                out[node]["inputs"] = self.options[node]["inputs"]

                for input_dict in out[node]["inputs"].values():
                    if hasattr(input_dict["type"], "model_json_schema"):
                        type_overwrite = f"A JSON object of the following properties: {input_dict['type'].model_json_schema()['properties']}"
                        if "$defs" in input_dict["type"].model_json_schema():
                            type_overwrite += f"\nWhere the values are: {input_dict['type'].model_json_schema()['$defs']}"
                        input_dict["type"] = type_overwrite
            else:
                out[node]["inputs"] = "No inputs are needed for this function."

        return out

    def _unavailable_options_to_json(self, unavailable_tools: list[tuple[str, str]]):
        """
        Options unavailable at this time and why.
        """
        out = {}
        for tool, reason in unavailable_tools:
            if reason == "":
                reason = "No reason provided."
            out[tool] = {
                "function_name": tool,
                "description": self.options[tool]["description"],
                "available_at": reason,
            }
        return out

    def decide_from_route(self, route: list[str]):
        possible_nodes = self._get_options()

        next_route = route[0]
        if next_route not in possible_nodes:
            raise Exception(
                f"Next node in training route ({next_route}) not in possible nodes ({possible_nodes})"
            )

        route = route[1:]
        completed = len(route) == 0

        return (
            Decision(
                function_name=next_route,
                reasoning=f"Decided to run {next_route} from route {route}",
                impossible=False,
                function_inputs={},
                end_actions=completed,
                last_in_tree=completed,
            ),
            "/".join(route),
        )

    def _get_function_inputs(self, llm_inputs: dict, real_inputs: dict) -> dict:
        # any non-provided inputs are set to the default
        default_inputs = {
            key: value["default"] if "default" in value else None
            for key, value in real_inputs.items()
        }
        for default_input_name in default_inputs:
            if default_input_name not in llm_inputs:
                llm_inputs[default_input_name] = default_inputs[default_input_name]

        # if the inputs match the 'schema' of keys: description, type, default, value, then take the value
        for input_name in llm_inputs:
            if (
                isinstance(llm_inputs[input_name], dict)
                and "value" in llm_inputs[input_name]
            ):
                llm_inputs[input_name] = llm_inputs[input_name]["value"]

        return llm_inputs

    async def load_model_from_examples(
        self,
        decision_executor: dspy.Module,
        client_manager: ClientManager,
        user_prompt: str,
    ) -> dspy.Module:

        examples = await retrieve_feedback(client_manager, user_prompt, "decision")

        if len(examples) == 0:
            return decision_executor

        optimizer = dspy.LabeledFewShot(k=10)
        compiled_executor = optimizer.compile(decision_executor, trainset=examples)
        return compiled_executor

    def _tool_assertion(self, pred: dspy.Prediction, kwargs: dict) -> tuple[bool, str]:
        return (
            pred.function_name in kwargs["available_actions"],
            f"You picked the action `{pred.function_name}` - that is not in `available_actions`! "
            f"Your output MUST be one of the following: {list(kwargs['available_actions'].keys())}",
        )

    def _get_view_environment(self):
        return {
            "function_name": "view_environment",
            "description": (
                "An auxiliary tool to inspect the current environment containing all objects from this session. "
                "Use this tool when you need to reference existing data before making decisions or selecting other actions. "
                "This tool does NOT complete the user's request immediately, it only helps you gather information to inform your next action choice or inputs. "
                "After viewing the environment, you can select another tool after selecting this tool from available_actions. "
                "Retrieved objects from previous tool calls are stored here. "
                "Optionally input tool_name to filter by a specific tool's results. "
                "Optionally input metadata_key and metadata_value to filter by specific metadata criteria. "
            ),
            "inputs": {
                "tool_name": {
                    "description": (
                        "The name of the tool to view the environment for. Top level key for the environment. "
                        "If not provided or None, view the entire environment."
                    ),
                    "default": None,
                    "type": str,
                },
                "metadata_key": {
                    "description": (
                        "A key of the metadata to view the environment for. Subkey for the environment. "
                        "If not provided or None, view all objects for the given tool_name. "
                        "If provided metadata_key, must provide metadata_value. "
                    ),
                    "default": None,
                    "type": str,
                },
                "metadata_value": {
                    "description": (
                        "A value of the metadata to view the environment for. Subkey for the environment. "
                        "If not provided or None, view all objects for the given tool_name."
                        "If provided metadata_value, must provide metadata_key. "
                    ),
                    "default": None,
                    "type": str,
                },
            },
        }

    async def _execute_view_environment(
        self, kwargs: dict, tree_data: TreeData, inputs: dict, lm: dspy.LM
    ):

        history = dspy.History(messages=[])
        history.messages.append({**kwargs})

        if self.logger:
            self.logger.debug(
                f"Model picked view_environment, using environment decision executor."
            )

        function_inputs = self._get_function_inputs(
            llm_inputs=inputs,
            real_inputs=self._get_view_environment()["inputs"],
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

        environment_decision_executor = ElysiaChainOfThought(
            DPWithEnvMetadataResponse,
            tree_data=tree_data,
            reasoning=tree_data.settings.BASE_USE_REASONING,
        )
        available_actions = kwargs["available_actions"]
        available_actions.pop("view_environment")
        pred = await environment_decision_executor.predict.aforward(
            environment=environment,
            available_actions=available_actions,
            history=history,
            lm=lm,
        )

        return pred

    async def __call__(
        self,
        tree_data: TreeData,
        base_lm: dspy.LM,
        complex_lm: dspy.LM,
        available_tools: list[str],
        unavailable_tools: list[tuple[str, str]],
        successive_actions: dict,
        client_manager: ClientManager,
        **kwargs,
    ) -> tuple[Decision, list[TrainingUpdate | Status | Response | FewShotExamples]]:
        """
        Make a decision from the current node.
        If only one option is available, and there are no function inputs, that is the decision.
        Otherwise, use the decision executor to make a decision:
        1. Picks a tool from the available options
        2. (Optional) If an incorrect decision is made, add feedback in conversation history and try again. Errors on reaching max attempts.
        3. (Optional) If the decision needs to look at the environment, use the environment executor to make a decision.
        """

        available_options = self._options_to_json(available_tools)
        unavailable_options = self._unavailable_options_to_json(unavailable_tools)

        if len(available_options) == 0:
            raise RuntimeError(
                "No available tools to call! Make sure you have added some tools to the tree. "
                "Or the .is_tool_available() method is returning True for at least one tool."
            )

        if self.logger:
            self.logger.debug(f"Available options: {list(available_options.keys())}")

        one_choice = (
            all(
                option["inputs"] == "No inputs are needed for this function."
                for option in available_options.values()
            )
            and len(available_options) == 1
        )

        # add environment view to tools
        # TODO: replace this with actual token counting
        env_token_limit_reached = (
            len(json.dumps(tree_data.environment.to_json())) > tree_data.env_token_limit
        )
        if env_token_limit_reached:
            available_options["view_environment"] = self._get_view_environment()
            signature = DPWithEnvMetadata
        else:
            signature = DPWithEnv

        if one_choice:

            if self.logger:
                self.logger.debug(
                    f"Only one option available: {list(available_options.keys())[0]} (and no function inputs are needed)."
                )

            decision = Decision(
                function_name=list(available_options.keys())[0],
                reasoning=f"Only one option available: {list(available_options.keys())[0]} (and no function inputs are needed).",
                impossible=False,
                function_inputs={},
                end_actions=(
                    bool(self.options[list(available_options.keys())[0]]["end"])
                    and self.options[list(available_options.keys())[0]]["next"] is None
                ),
            )

            results: list[TrainingUpdate | Status | Response | FewShotExamples] = [
                Status(str(self.options[list(available_options.keys())[0]]["status"])),
            ]

            return decision, results

        decision_executor = ElysiaChainOfThought(
            signature,
            tree_data=tree_data,
            environment=not env_token_limit_reached,
            collection_schemas=self.use_elysia_collections,
            tasks_completed=True,
            message_update=True,
            reasoning=tree_data.settings.BASE_USE_REASONING,
        )
        decision_executor = AssertedModule(
            decision_executor,
            self._tool_assertion,
        )

        # TODO: do feedback thing
        # if tree_data.settings.USE_FEEDBACK:
        #     if not client_manager.is_client:
        #         raise ValueError(
        #             "A Weaviate connection is required for the experimental `use_feedback` method. "
        #             "Please set the WCD_URL and WCD_API_KEY in the settings. "
        #             "Or, set `use_feedback` to False."
        #         )

        #     if self.logger:
        #         self.logger.debug(f"Using feedback examples for decision executor.")

        #     pred, uuids = await decision_executor.aforward_with_feedback_examples(
        #         feedback_model="decision",
        #         client_manager=client_manager,
        #         base_lm=base_lm,
        #         complex_lm=complex_lm,
        #         instruction=self.instruction,
        #         tree_count=tree_data.tree_count_string(),
        #         environment_metadata=tree_data.environment.output_llm_metadata(),
        #         available_actions=available_options,
        #         unavailable_actions=unavailable_options,
        #         successive_actions=successive_actions,
        #         num_base_lm_examples=3,
        #         return_example_uuids=True,
        #     )
        # else:

        if self.logger:
            self.logger.debug(f"Using base LM for decision executor.")

        pred = await decision_executor.aforward(
            instruction=self.instruction,
            tree_count=tree_data.tree_count_string(),
            environment_metadata=tree_data.environment.output_llm_metadata(),
            available_actions=available_options,
            unavailable_actions=unavailable_options,
            successive_actions=successive_actions,
            lm=base_lm,
        )

        if pred.function_name.startswith("'") and pred.function_name.endswith("'"):
            pred.function_name = pred.function_name[1:-1]
        elif pred.function_name.startswith('"') and pred.function_name.endswith('"'):
            pred.function_name = pred.function_name[1:-1]
        elif pred.function_name.startswith("`") and pred.function_name.endswith("`"):
            pred.function_name = pred.function_name[1:-1]

        if pred.function_name not in available_options:
            raise Exception(
                f"Model picked an action `{pred.function_name}` that is not in the available tools: {available_tools}"
            )

        if pred.function_name == "view_environment":
            pred = await self._execute_view_environment(
                kwargs={
                    "user_prompt": tree_data.user_prompt,
                    "instruction": self.instruction,
                    "tree_count": tree_data.tree_count_string(),
                    "environment_metadata": tree_data.environment.output_llm_metadata(),
                    "available_actions": available_options,
                    "unavailable_actions": unavailable_options,
                    "successive_actions": successive_actions,
                    **decision_executor.module._add_tree_data_inputs({}),  # type: ignore
                    **pred,
                },
                tree_data=tree_data,
                inputs=pred.function_inputs,
                lm=base_lm,
            )

        decision = Decision(
            pred.function_name,
            self._get_function_inputs(
                llm_inputs=pred.function_inputs,
                real_inputs=available_options[pred.function_name]["inputs"],
            ),
            pred.reasoning if tree_data.settings.BASE_USE_REASONING else "",
            pred.impossible,
            pred.end_actions and bool(self.options[pred.function_name]["end"]),
        )

        results = [
            TrainingUpdate(
                module_name="decision",
                inputs=tree_data.to_json(),
                outputs={k: v for k, v in pred.__dict__["_store"].items()},
            ),
            Status(str(self.options[pred.function_name]["status"])),
        ]

        if pred.function_name != "text_response":
            results.append(Response(pred.message_update))

        # TODO: do feedback thing
        # if tree_data.settings.USE_FEEDBACK and len(uuids) > 0:
        #     results.append(FewShotExamples(uuids))

        return decision, results


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
        result: Result | TreeUpdate | Update | Text | Error,
        query_id: str,
    ) -> dict[str, str | dict] | None:

        if isinstance(result, (Update, Text, Result, Error)):
            payload = await result.to_frontend(
                self.user_id, self.conversation_id, query_id
            )
            self.store.append(payload)
            return payload

        if isinstance(result, TreeUpdate):
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
    client_manager: ClientManager | None = None,
    user_id: str | None = None,
):
    """
    Get all saved trees from a Weaviate collection.

    Args:
        collection_name (str): The name of the collection to get the trees from.
        client_manager (ClientManager): The client manager to use.
            If not provided, a new ClientManager will be created from environment variables.
        user_id (str): The user ID to get the trees from.
            If not provided, the trees will be retrieved from the collection without any filters.

    Returns:
        dict: A dictionary of tree UUIDs and their titles.
    """
    if client_manager is None:
        client_manager = ClientManager()
        close_after_use = True
    else:
        close_after_use = False

    if user_id is not None:
        user_id_filter = Filter.by_property("user_id").equal(user_id)
    else:
        user_id_filter = None

    async with client_manager.connect_to_async_client() as client:

        if not await client.collections.exists(collection_name):
            return {}

        collection = client.collections.get(collection_name)

        len_collection = (
            await collection.aggregate.over_all(
                total_count=True,
                filters=user_id_filter,
            )
        ).total_count

        response = await collection.query.fetch_objects(
            limit=len_collection,
            sort=Sort.by_update_time(ascending=False),
            return_metadata=MetadataQuery(last_update_time=True),
            filters=user_id_filter,
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
    conversation_id: str,
    collection_name: str,
    client_manager: ClientManager | None = None,
):
    """
    Delete a tree from a Weaviate collection.

    Args:
        conversation_id (str): The conversation ID of the tree to delete.
        collection_name (str): The name of the collection to delete the tree from.
        client_manager (ClientManager): The client manager to use.
            If not provided, a new ClientManager will be created from environment variables.
    """

    if client_manager is None:
        client_manager = ClientManager()

    async with client_manager.connect_to_async_client() as client:
        collection = client.collections.get(collection_name)
        uuid = generate_uuid5(conversation_id)
        if await collection.data.exists(uuid):
            await collection.data.delete_by_id(uuid)
