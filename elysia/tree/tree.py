import inspect
import json
import time
import textwrap
from copy import deepcopy
from typing import AsyncGenerator, Literal

import dspy
from pympler import asizeof
from rich import print
from rich.console import Console
from rich.panel import Panel

import uuid

# Weaviate
import weaviate.classes.config as wc
from weaviate.util import generate_uuid5

# Elysia
from elysia.config import Settings, ElysiaKeyManager
from elysia.config import settings as environment_settings
from elysia.tree.util import (
    DecisionNode,
    TreeReturner,
    Decision,
    get_follow_up_suggestions,
    create_conversation_title,
)
from elysia.objects import (
    Completed,
    Result,
    Return,
    Update,
    Text,
    Tool,
    Warning,
    Error,
)
from elysia.tools.retrieval.aggregate import Aggregate
from elysia.tools.retrieval.query import Query
from elysia.tools.visualisation.visualise import Visualise
from elysia.tools.postprocessing.summarise_items import SummariseItems
from elysia.tools.text.text import (
    FakeTextResponse,
    CitedSummarizer,
)
from elysia.tree.util import ForcedTextResponse
from elysia.util.async_util import asyncio_run
from elysia.tree.objects import CollectionData, TreeData, Atlas, Environment
from elysia.util.client import ClientManager
from elysia.config import (
    check_base_lm_settings,
    check_complex_lm_settings,
    load_base_lm,
    load_complex_lm,
)
from elysia.util.objects import Tracker, TrainingUpdate, TreeUpdate
from elysia.util.parsing import remove_whitespace
from elysia.util.collection import retrieve_all_collection_names


class Tree:
    """
    The main class for the Elysia decision tree.
    Calling this method will execute the decision tree based on the user's prompt, and available collections and tools.
    """

    def __init__(
        self,
        branch_initialisation: Literal[
            "default", "one_branch", "multi_branch", "empty"
        ] = "default",
        style: str = "No style provided.",
        agent_description: str = "No description provided.",
        end_goal: str = "No end goal provided.",
        user_id: str | None = None,
        conversation_id: str | None = None,
        low_memory: bool = False,
        use_elysia_collections: bool = True,
        settings: Settings | None = None,
    ) -> None:
        """
        Args:
            branch_initialisation (str): The initialisation method for the branches,
                currently supports some pre-defined initialisations: "multi_branch", "one_branch".
                Set to "empty" to start with no branches and to add them, and the tools, yourself.
            style (str): The writing style of the agent. Automatically set for "multi_branch" and "one_branch" initialisation, but overrided if non-empty.
            agent_description (str): The description of the agent. Automatically set for "multi_branch" and "one_branch" initialisation, but overrided if non-empty.
            end_goal (str): The end goal of the agent. Automatically set for "multi_branch" and "one_branch" initialisation, but overrided if non-empty.
            user_id (str): The id of the user, e.g. "123-456",
                unneeded outside of user management/hosting Elysia app
            conversation_id (str): The id of the conversation, e.g. "123-456",
                unneeded outside of conversation management/hosting Elysia app
            low_memory (bool): Whether to run the tree in low memory mode.
                If True, the tree will not load the (dspy) models within the tree.
                Set to False for normal operation.
            use_elysia_collections (bool): Whether to use weaviate collections as processed by Elysia.
                If False, the tree will not use the processed collections.
            settings (Settings): The settings for the tree, an object of elysia.Settings.
                This is automatically set to the environment settings if not provided.
        """
        # Define base variables of the tree
        if user_id is None:
            self.user_id = str(uuid.uuid4())
        else:
            self.user_id = user_id

        if conversation_id is None:
            self.conversation_id = str(uuid.uuid4())
        else:
            self.conversation_id = conversation_id

        if settings is None:
            self.settings = environment_settings
        else:
            assert isinstance(
                settings, Settings
            ), "settings must be an instance of Settings"
            self.settings = settings

        self.use_elysia_collections = use_elysia_collections

        # Initialise some tree variables
        self.decision_nodes: dict[str, DecisionNode] = {}
        self.decision_history = [[]]
        self.tree_index = -1
        self.suggestions = []
        self.actions_called = {}
        self.query_id_to_prompt = {}
        self.prompt_to_query_id = {}
        self.retrieved_objects = []
        self.store_retrieved_objects = False
        self.conversation_title = None
        self.low_memory = low_memory
        self._base_lm = None
        self._complex_lm = None
        self._config_modified = False
        self.root = None

        # Define the inputs to prompts
        self.tree_data = TreeData(
            environment=Environment(),
            collection_data=CollectionData(
                collection_names=[], logger=self.settings.logger
            ),
            atlas=Atlas(
                style=style,
                agent_description=agent_description,
                end_goal=end_goal,
            ),
            recursion_limit=5,
            settings=self.settings,
        )

        # initialise the timers
        self.tracker = Tracker(
            tracker_names=["decision_node"],
            logger=self.settings.logger,
        )

        # Set the initialisations
        self.tools = {}
        self.set_branch_initialisation(branch_initialisation)
        self.tree_data.atlas.style = style
        self.tree_data.atlas.agent_description = agent_description
        self.tree_data.atlas.end_goal = end_goal

        self.tools["forced_text_response"] = ForcedTextResponse()

        # some variables for storing feedback
        self.action_information = []
        self.history = {}
        self.training_updates = []

        # -- Get the root node and construct the tree
        self._get_root()
        self.tree = {}
        self._construct_tree(self.root, self.tree)

        # initialise the returner (for frontend)
        self.returner = TreeReturner(
            user_id=self.user_id,
            conversation_id=self.conversation_id,
        )

        # Print the tree if required
        self.settings.logger.debug(
            "Initialised tree with the following decision nodes:"
        )
        for decision_node in self.decision_nodes.values():
            self.settings.logger.debug(
                f"  - [magenta]{decision_node.id}[/magenta]: {list(decision_node.options.keys())}"
            )

    @property
    def base_lm(self) -> dspy.LM:
        if self.low_memory:
            return load_base_lm(self.settings)
        else:
            if self._base_lm is None:
                self._base_lm = load_base_lm(self.settings)
            return self._base_lm

    @property
    def complex_lm(self) -> dspy.LM:
        if self.low_memory:
            return load_complex_lm(self.settings)
        else:
            if self._complex_lm is None:
                self._complex_lm = load_complex_lm(self.settings)
            return self._complex_lm

    def multi_branch_init(self) -> None:
        self.add_branch(
            root=True,
            branch_id="base",
            instruction="""
            Choose a base-level task based on the user's prompt and available information.
            You can search, which includes aggregating or querying information - this should be used if the user needs (more) information.
            You can end the conversation by choosing text response, or summarise some retrieved information.
            Base your decision on what information is available and what the user is asking for - you can search multiple times if needed,
            but you should not search if you have already found all the information you need.
            """,
            status="Choosing a base-level task...",
        )
        self.add_tool(branch_id="base", tool=CitedSummarizer)
        self.add_tool(branch_id="base", tool=FakeTextResponse)

        self.add_branch(
            root=False,
            branch_id="search",
            from_branch_id="base",
            instruction="""
            Choose between querying the knowledge base via semantic/keyword search, or aggregating information by performing operations, on the knowledge base.
            Querying is when the user is looking for specific information related to the content of the dataset, requiring a specific search query. This is for retrieving specific information via a _query_, similar to a search engine.
            Aggregating is when the user is looking for a specific operations on the dataset, such as summary statistics of the quantity of some items. Aggregation can also include grouping information by some property and returning statistics about the groups.
            """,
            description=f"""
            Search the knowledge base. This should be used when the user is lacking information for this particular prompt. This retrieves information only and provides no output to the user except the information.
            Choose to query (semantic or keyword search on a knowledge base), or aggregate information (calculate properties/summary statistics/averages and operations on the knowledge bases).
            """,
            status="Searching the knowledge base...",
        )
        self.add_tool(branch_id="search", tool=Query, summariser_in_tree=True)
        self.add_tool(branch_id="search", tool=Aggregate)
        self.add_tool(branch_id="base", tool=Visualise)
        self.add_tool(SummariseItems, branch_id="search", from_tool_ids=["query"])

    def one_branch_init(self) -> None:
        self.add_branch(
            root=True,
            branch_id="base",
            instruction="""
            Choose a base-level task based on the user's prompt and available information.
            Decide based on the tools you have available as well as their descriptions.
            Read them thoroughly and match the actions to the user prompt.
            """,
            status="Choosing a base-level task...",
        )
        self.add_tool(branch_id="base", tool=CitedSummarizer)
        self.add_tool(branch_id="base", tool=FakeTextResponse)
        self.add_tool(branch_id="base", tool=Aggregate)
        self.add_tool(branch_id="base", tool=Query, summariser_in_tree=True)
        self.add_tool(branch_id="base", tool=Visualise)
        self.add_tool(SummariseItems, branch_id="base", from_tool_ids=["query"])

    def empty_init(self) -> None:
        self.add_branch(
            root=True,
            branch_id="base",
            instruction="""
            Choose a base-level task based on the user's prompt and available information.
            Decide based on the tools you have available as well as their descriptions.
            Read them thoroughly and match the actions to the user prompt.
            """,
            status="Choosing a base-level task...",
        )

    def clear_tree(self) -> None:
        self.decision_nodes = {}
        self.root = None

    def set_branch_initialisation(self, initialisation: str | None) -> None:
        self.clear_tree()

        if (
            initialisation is None
            or initialisation == ""
            or initialisation == "one_branch"
            or initialisation == "default"
        ):
            self.one_branch_init()
        elif initialisation == "multi_branch":
            self.multi_branch_init()
        elif initialisation == "empty":
            self.empty_init()
        else:
            raise ValueError(f"Invalid branch initialisation: {initialisation}")

        self.branch_initialisation = initialisation

    def smart_setup(self) -> None:
        """
        Configures the `settings` object of the tree with the `Settings.smart_setup()` method.
        """

        self.settings = deepcopy(self.settings)
        self.settings.SETTINGS_ID = str(uuid.uuid4())
        self._config_modified = True
        self.settings.smart_setup()

    def configure(self, **kwargs) -> None:
        """
        Configure the tree with new settings.
        Wrapper for the settings.configure() method.
        Will not affect any settings preceding this (e.g. in TreeManager).
        """
        self.settings = deepcopy(self.settings)
        self.settings.SETTINGS_ID = str(uuid.uuid4())
        self._config_modified = True
        self.tree_data.settings = self.settings
        self.settings.configure(**kwargs)

    def change_style(self, style: str) -> None:
        self.tree_data.atlas.style = style
        self._config_modified = True

    def change_agent_description(self, agent_description: str) -> None:
        self.tree_data.atlas.agent_description = agent_description
        self._config_modified = True

    def change_end_goal(self, end_goal: str) -> None:
        self.tree_data.atlas.end_goal = end_goal
        self._config_modified = True

    def _get_root(self) -> None:
        for decision_node in self.decision_nodes.values():
            if decision_node.root:
                if self.root is not None and self.root != decision_node.id:
                    raise ValueError("Multiple root decision nodes found")

                self.root = decision_node.id

        if self.root is None:
            raise ValueError("No root decision node found")

    def _construct_tree(
        self, node_id: str | None, tree: dict, branch: bool = True
    ) -> dict:
        if node_id is None:
            raise ValueError("Node ID is None")

        decision_node = self.decision_nodes[node_id]

        # Ensure the order of the keys in each option is the same
        key_order = [
            "name",
            "id",
            "description",
            "instruction",
            "reasoning",
            "branch",
            "options",
        ]

        # Set the base node information
        tree["name"] = node_id.capitalize().replace("_", " ")
        tree["id"] = node_id
        if node_id == self.root:
            tree["description"] = ""
        tree["instruction"] = remove_whitespace(
            decision_node.instruction.replace("\n", "")
        )
        tree["reasoning"] = ""
        tree["branch"] = branch
        tree["options"] = {}

        # Order the top-level dictionary
        tree = {key: tree[key] for key in key_order if key in tree}

        # Initialize all options first with ordered dictionaries
        for option in decision_node.options:
            tree["options"][option] = {
                "description": remove_whitespace(
                    str(decision_node.options[option]["description"]).replace("\n", "")
                )
            }

        # Then handle the recursive cases
        for option in decision_node.options:
            next_node: DecisionNode | None = decision_node.options[option]["next"]  # type: ignore
            if (
                decision_node.options[option]["action"] is not None
                and next_node is None
            ):
                tree["options"][option]["name"] = option.capitalize().replace("_", " ")
                tree["options"][option]["id"] = option
                tree["options"][option]["instruction"] = ""
                tree["options"][option]["reasoning"] = ""
                tree["options"][option]["branch"] = False
                tree["options"][option]["options"] = {}

            elif next_node is not None:
                tree["options"][option] = self._construct_tree(
                    next_node.id,
                    tree["options"][option],
                    branch=decision_node.options[option]["action"] is None,
                )
            else:
                tree["options"][option]["name"] = option.capitalize().replace("_", " ")
                tree["options"][option]["id"] = option
                tree["options"][option]["instruction"] = ""
                tree["options"][option]["reasoning"] = ""
                tree["options"][option]["branch"] = True
                tree["options"][option]["options"] = {}

            # Order each option's dictionary
            tree["options"][option] = {
                key: tree["options"][option][key]
                for key in key_order
                if key in tree["options"][option]
            }

        return tree

    async def set_collection_names(
        self,
        collection_names: list[str],
        client_manager: ClientManager,
    ) -> None:
        self.settings.logger.debug(
            f"Using the following collection names: {collection_names}"
        )

        collection_names = await self.tree_data.set_collection_names(
            collection_names, client_manager
        )

    def _remove_empty_branches(self) -> list[str]:
        empty_branches = []
        for branch_id, branch in self.decision_nodes.items():
            if len(branch.options) == 0:
                empty_branches.append(branch_id)

        for branch_id in self.decision_nodes:
            for empty_branch in empty_branches:
                self.decision_nodes[branch_id].remove_option(empty_branch)

        for empty_branch in empty_branches:
            if empty_branch != self.root:
                self.settings.logger.warning(
                    f"Removing empty branch: {empty_branch} "
                    "No tools are attached to this branch, so it has been removed. "
                    f"To add a tool to this branch, use .add_tool(tool_name, branch_id={empty_branch})"
                )
                del self.decision_nodes[empty_branch]

        return empty_branches

    def _get_function_inputs(self, tool_name: str, inputs: dict) -> dict:
        if tool_name in self.tools:
            # any non-provided inputs are set to the default
            default_inputs = self.tools[tool_name].get_default_inputs()
            for default_input_name in default_inputs:
                if default_input_name not in inputs:
                    inputs[default_input_name] = default_inputs[default_input_name]

            # if the inputs match the 'schema' of keys: description, type, default, value, then take the value
            for input_name in inputs:
                if (
                    isinstance(inputs[input_name], dict)
                    and "value" in inputs[input_name]
                ):
                    inputs[input_name] = inputs[input_name]["value"]

            return inputs
        else:
            return {}

    async def _check_rules(
        self, branch_id: str, client_manager: ClientManager
    ) -> tuple[list[str], dict]:
        branch = self.decision_nodes[branch_id]
        nodes_with_rules_met = []
        rule_tool_inputs = {}
        for function_name, option in branch.options.items():
            if function_name not in self.tools:
                pass
            elif "run_if_true" in dir(self.tools[function_name]):
                rule_met, rule_tool_inputs = await self.tools[
                    function_name
                ].run_if_true(
                    tree_data=self.tree_data,
                    client_manager=client_manager,
                    base_lm=self.base_lm,
                    complex_lm=self.complex_lm,
                )
                if rule_met:
                    nodes_with_rules_met.append(function_name)
                    if rule_tool_inputs is None or rule_tool_inputs == {}:
                        rule_tool_inputs[function_name] = self.tools[
                            function_name
                        ].get_default_inputs()
                    else:
                        rule_tool_inputs[function_name] = rule_tool_inputs

        return nodes_with_rules_met, rule_tool_inputs

    def set_conversation_id(self, conversation_id: str) -> None:
        self.conversation_id = conversation_id
        self.returner.conversation_id = conversation_id

    def set_user_id(self, user_id: str) -> None:
        self.user_id = user_id
        self.returner.user_id = user_id

    def soft_reset(self) -> None:
        # conversation history is not reset
        # environment is not reset
        if self.low_memory:
            self.history = {}

        self.recursion_counter = 0
        self.tree_data.num_trees_completed = 0
        self.decision_history = [[]]
        self.training_updates = []
        self.tree_data.soft_reset()
        self.action_information = []
        self.tree_index += 1
        self.retrieved_objects = []
        self.returner.set_tree_index(self.tree_index)

    def save_history(self, query_id: str, time_taken_seconds: float) -> None:
        """
        What the tree did, results for saving feedback.
        """
        training_update = deepcopy(
            [update.to_json() for update in self.training_updates]
        )

        self.history[query_id] = {
            "num_trees_completed": self.tree_data.num_trees_completed,
            "tree_data": deepcopy(self.tree_data),
            "action_information": deepcopy(self.action_information),
            "decision_history": [
                item for sublist in deepcopy(self.decision_history) for item in sublist
            ],
            "base_lm_used": self.settings.BASE_MODEL,
            "complex_lm_used": self.settings.COMPLEX_MODEL,
            "time_taken_seconds": time_taken_seconds,
            "training_updates": training_update,
            "initialisation": f"{self.branch_initialisation}",
        }
        # can reset training updates now
        self.training_updates = []

    def set_start_time(self) -> None:
        self.start_time = time.time()

    def add_tool(
        self,
        tool,
        branch_id: str | None = None,
        from_tool_ids: list[str] = [],
        root: bool = False,
        **kwargs,
    ) -> None:
        """
        Add a Tool to a branch or on top of an existing tool.
        The tool needs to be an instance of the Tool class.

        Args:
            tool (Tool): The tool to add
            branch_id (str): The id of the branch to add the tool to
                If not specified, the tool will be added to the root branch
            from_tool_ids (list[str]): The ids of the tools to add the new tool after
                If not specified, the tool will be added to the base of the branch
            root (bool): Whether the tool is the root tool
                If not specified, the tool will be added to the root branch
            kwargs (any): Additional keyword arguments to pass to the initialisation of the tool

        Example 1:
            To add a tool, `Query`, to a branch called 'search', you can do this:
            ```python
            tree.add_tool(Query, branch_id="search")
            ```
            This will add the `Query` tool to the branch 'search'.
            If the branch 'search' doesn't exist, it will raise an error.
            To add a branch, use the `.add_branch()` method.


        Example 2:
            Assume your tree has a "search" branch with two tools: 'query' and 'aggregate'.
            You can add a tool, `CheckResult`, after the 'query' tool like this:
            ```python
            tree.add_tool(CheckResult, branch_id="search", from_tool_ids=["query"])
            ```
            This will add the `CheckResult` tool to the "search" branch, after the 'query' tool.
            So the "search" branch will still only have two options: 'query' and 'aggregate'.
            But after 'query', there will be a new option for the `CheckResult` tool.

        Example 3:
            You can add a tool, `SendEmail`, after the `CheckResult` (from Example 2) tool like this:
            ```python
            tree.add_tool(SendEmail, from_tool_ids=["query", "check_result"], root=True)
            ```
            It will add an additional option to the root branch, after the 'query' and 'check_result' tools.
        """

        if (
            inspect.getfullargspec(tool.__init__).varkw is None
            or inspect.getfullargspec(tool.__call__).varkw is None
        ):
            raise TypeError("tool __init__ and __call__ must accept **kwargs")

        if not inspect.isasyncgenfunction(tool.__call__):
            raise TypeError(
                "__call__ must be an async generator function. "
                "I.e. it must yield objects."
            )

        if isinstance(tool, Tool):
            tool_instance = tool
        else:
            tool_instance = tool(
                logger=self.settings.logger,
                **kwargs,
            )

        if not isinstance(tool_instance, Tool):
            raise TypeError("tool must be an instance of the Tool class")

        if "__call__" not in dir(tool_instance):
            raise TypeError("tool must be callable (have a __call__ method)")

        if "__init__" not in dir(tool_instance):
            raise TypeError("tool must have an __init__ method")

        if hasattr(tool_instance, "is_tool_available"):
            if not inspect.iscoroutinefunction(tool_instance.is_tool_available):
                raise TypeError(
                    "is_tool_available must be an async function that returns a single boolean value"
                )

        if hasattr(tool_instance, "run_if_true"):
            if not inspect.iscoroutinefunction(tool_instance.run_if_true):
                raise TypeError(
                    "run_if_true must be an async function that returns a single boolean value"
                )

        if root:
            if branch_id is not None:
                self.settings.logger.warning(
                    f"In .add_tool(), `root` is True, so `branch_id` ('{branch_id}') will be ignored. "
                    f"Tool: '{tool_instance.name}' will be added to the root branch ('{self.root}')."
                )
            branch_id = self.root

        if branch_id is None:
            branch_id = self.root

        if branch_id not in self.decision_nodes:
            raise ValueError(
                f"Branch '{branch_id}' not found. Use .add_branch() to add a branch before adding a tool. "
                f"Or, set `root=True` to add the tool to the root branch ('{self.root}')."
            )

        current_decision_node = self.decision_nodes[branch_id]
        for from_tool_id in from_tool_ids:
            if isinstance(current_decision_node, DecisionNode):
                if from_tool_id not in current_decision_node.options:
                    raise ValueError(
                        f"Tool '{from_tool_id}' not found in branch '{branch_id}'. "
                        f"Available options are: {list(current_decision_node.options.keys())}"
                    )

                current_decision_node = current_decision_node.options[from_tool_id][
                    "next"
                ]

        self.tools[tool_instance.name] = tool_instance

        if from_tool_ids == []:
            self.decision_nodes[branch_id].add_option(
                id=tool_instance.name,
                description=tool_instance.description,
                inputs=tool_instance.inputs,
                action=self.tools[tool_instance.name],
                end=tool_instance.end,
                status=tool_instance.status,
            )
        else:

            new_branch_id = branch_id
            for from_tool_id in from_tool_ids:
                new_branch_id += f".{from_tool_id}"

            # only create a new decision node if one doesn't exist here
            if new_branch_id not in self.decision_nodes:
                decision_node = DecisionNode(
                    id=new_branch_id,
                    instruction=f"Choose one of the actions based on their descriptions and the user prompt.",
                    options={},
                    root=False,
                    logger=self.settings.logger,
                    use_elysia_collections=self.use_elysia_collections,
                )
                self.decision_nodes[new_branch_id] = decision_node

                prev_branch_id = branch_id
                for from_tool_id in from_tool_ids[:-1]:
                    prev_branch_id += f".{from_tool_id}"

                self.decision_nodes[prev_branch_id].options[from_tool_ids[-1]][
                    "next"
                ] = self.decision_nodes[new_branch_id]

            # add the tool to the new decision node
            self.decision_nodes[new_branch_id].add_option(
                id=tool_instance.name,
                description=tool_instance.description,
                inputs=tool_instance.inputs,
                action=self.tools[tool_instance.name],
                end=tool_instance.end,
                status=tool_instance.status,
            )

        self.tracker.add_tracker(tool_instance.name)

        # reconstruct tree
        self._get_root()
        self.tree = {}
        self._construct_tree(self.root, self.tree)

    def remove_tool(
        self,
        tool_name: str,
        branch_id: str | None = None,
        from_tool_ids: list[str] = [],
        root: bool = False,
    ) -> None:
        """
        Remove a Tool from a branch.

        Args:
            tool_name (str): The name of the tool to remove.
            branch_id (str): The id of the branch to remove the tool from,
                if not specified, the tool will be removed from the root branch.
            from_tool_ids (list[str]): The ids of the tools to which precedes the tool to remove.
            root (bool): Whether the branch the tool is in is the root branch.
        """
        if root:
            if branch_id is not None:
                self.settings.logger.warning(
                    f"In .add_tool(), `root` is True, so `branch_id` ('{branch_id}') will be ignored. "
                    f"Tool: '{tool_name}' will be removed from the root branch ('{self.root}')."
                )
            branch_id = self.root

        if branch_id is None:
            branch_id = self.root

        if branch_id not in self.decision_nodes:
            raise ValueError(f"Branch {branch_id} not found.")

        if (
            tool_name not in self.decision_nodes[branch_id].options
            and from_tool_ids == []
        ):
            raise ValueError(f"Tool {tool_name} not found in branch {branch_id}.")

        current_decision_node = self.decision_nodes[branch_id]
        for from_tool_id in from_tool_ids:
            if isinstance(current_decision_node, DecisionNode):
                if from_tool_id not in current_decision_node.options:
                    raise ValueError(
                        f"Tool '{from_tool_id}' not found in branch '{current_decision_node.id}'. "
                        f"Available options are: {list(current_decision_node.options.keys())}"
                    )
                current_decision_node = current_decision_node.options[from_tool_id][
                    "next"
                ]

        if (
            isinstance(current_decision_node, DecisionNode)
            and tool_name not in current_decision_node.options
        ):
            raise ValueError(
                f"Tool '{tool_name}' not found in branch '{current_decision_node.id}'. "
                f"Available options are: {list(current_decision_node.options.keys())}"
            )

        if from_tool_ids == []:
            self.decision_nodes[branch_id].remove_option(tool_name)
        else:
            tool_branch_id = branch_id
            for from_tool_id in from_tool_ids:
                tool_branch_id += f".{from_tool_id}"
            tool_branch_id += f".{tool_name}"

            prev_branch_id = branch_id
            for from_tool_id in from_tool_ids:
                prev_branch_id += f".{from_tool_id}"

            self.decision_nodes[prev_branch_id].remove_option(tool_name)
            if self.decision_nodes[prev_branch_id].options == {}:
                del self.decision_nodes[prev_branch_id]
                stem_branch_id = prev_branch_id[: prev_branch_id.rfind(".")]
                for stem_branch_option in self.decision_nodes[
                    stem_branch_id
                ].options.values():
                    if (
                        stem_branch_option["next"] is not None
                        and isinstance(stem_branch_option["next"], DecisionNode)
                        and stem_branch_option["next"].id == prev_branch_id
                    ):
                        stem_branch_option["next"] = None

            if (
                tool_branch_id in self.decision_nodes
                and self.decision_nodes[tool_branch_id].options != {}
            ):
                self.settings.logger.warning(
                    f"The following tools stem from '{tool_branch_id}', "
                    f"and have also been removed: {list(self.decision_nodes[tool_branch_id].options.keys())}"
                )

            # find any decision nodes that stem from this
            nodes_to_remove = []
            for decision_node_id in self.decision_nodes:
                if decision_node_id.startswith(tool_branch_id):
                    if decision_node_id != tool_branch_id:
                        self.settings.logger.warning(
                            f"Decision node '{decision_node_id}' stems from '{tool_branch_id}'. "
                            f"Removing tool '{tool_name}' has also removed '{decision_node_id}'."
                        )
                    nodes_to_remove.append(decision_node_id)

            for decision_node_id in nodes_to_remove:
                del self.decision_nodes[decision_node_id]

        del self.tools[tool_name]
        self.tracker.remove_tracker(tool_name)

        # reconstruct tree
        self._get_root()
        self.tree = {}
        self._construct_tree(self.root, self.tree)

    def add_branch(
        self,
        branch_id: str,
        instruction: str,
        description: str = "",
        root: bool = False,
        from_branch_id: str = "",
        from_tool_ids: list[str] = [],
        status: str = "",
    ) -> None:
        """
        Add a branch to the tree.

        args:
            branch_id (str): The id of the branch being added.
            instruction (str): The general instruction for the branch, what is this branch containing?
                What kind of tools or actions are being decided on this branch?
                Only displayed to the decision maker when this branch is chosen.
            description (str): A description of the branch, if it is to be chosen from a previous branch.
                How does the model know whether to choose this branch or not?
            root (bool): Whether this is the root branch, i.e. the beginning of the tree.
            from_branch_id (str): The id of the branch that this branch is stemming from.
            from_tool_ids (list[str]): The ids of the tools that precede this branch being added (after the `from_branch_id` branch).
            status (str): The status message to be displayed when this branch is chosen.
        """
        if not root and description == "":
            raise ValueError("Description is required for non-root branches.")
        if not root and from_branch_id == "":
            raise ValueError(
                "`from_branch_id` is required for non-root branches. "
                "Set `root=True` to create a root branch or choose where this branch stems from."
            )
        if root and description != "":
            self.settings.logger.warning(f"Description is not used for root branches. ")
            description = ""

        if root and from_branch_id != "":
            self.settings.logger.warning(
                "`from_branch_id` is not used for root branches. "
                "(As this is the root branch, it does not stem from any other branch.)"
                "If you wish this to be stemming from a previous branch, set `root=False`."
            )
            from_branch_id = ""

        if status == "":
            status = f"Running {branch_id}..."

        decision_node = DecisionNode(
            id=branch_id,
            instruction=instruction,
            options={},
            root=root,
            logger=self.settings.logger,
            use_elysia_collections=self.use_elysia_collections,
        )
        self.decision_nodes[branch_id] = decision_node

        if not root:

            if from_tool_ids == []:
                self.decision_nodes[from_branch_id].add_option(
                    id=branch_id,
                    description=description,
                    inputs={},
                    action=None,
                    end=False,
                    status=status,
                    next=self.decision_nodes[branch_id],
                )

            else:

                current_decision_node = self.decision_nodes[from_branch_id]
                for from_tool_id in from_tool_ids:
                    if isinstance(current_decision_node, DecisionNode):
                        if from_tool_id not in current_decision_node.options:
                            raise ValueError(
                                f"Tool '{from_tool_id}' not found in branch '{from_branch_id}'. "
                                f"Available options are: {list(current_decision_node.options.keys())}"
                            )
                        current_decision_node = current_decision_node.options[
                            from_tool_id
                        ]["next"]

                new_branch_id = from_branch_id
                for from_tool_id in from_tool_ids:
                    new_branch_id += f".{from_tool_id}"

                # only create a new decision node if one doesn't exist here
                if new_branch_id not in self.decision_nodes:
                    decision_node = DecisionNode(
                        id=new_branch_id,
                        instruction=f"Choose one of the actions based on their descriptions and the user prompt.",
                        options={},
                        root=False,
                        logger=self.settings.logger,
                        use_elysia_collections=self.use_elysia_collections,
                    )
                    self.decision_nodes[new_branch_id] = decision_node

                    prev_branch_id = branch_id
                    for from_tool_id in from_tool_ids[:-1]:
                        prev_branch_id += f".{from_tool_id}"

                    self.decision_nodes[prev_branch_id].options[from_tool_ids[-1]][
                        "next"
                    ] = self.decision_nodes[new_branch_id]

                # add the tool to the new decision node
                self.decision_nodes[new_branch_id].add_option(
                    id=branch_id,
                    description=description,
                    inputs={},
                    action=None,
                    end=False,
                    status=status,
                    next=self.decision_nodes[branch_id],
                )

        if root and (self.root is not None):
            # replace root branch with this one
            self.decision_nodes[self.root] = decision_node
            self.settings.logger.debug(
                f"Replacing root branch '{self.root}' with '{branch_id}'."
            )
            old_root = self.root
            self.root = branch_id
            self.remove_branch(old_root)

        # reconstruct tree
        self._get_root()
        self.tree = {}
        self._construct_tree(self.root, self.tree)

    def remove_branch(self, branch_id: str) -> None:
        """
        Remove a branch from the tree.

        Args:
            branch_id (str): The id of the branch to remove
        """
        # Validate branch exists
        if branch_id not in self.decision_nodes:
            self.settings.logger.warning(
                f"Branch {branch_id} not found, nothing to remove."
            )
            return

        # Special handling for root node
        if (
            branch_id == self.root
            and sum(1 for node in self.decision_nodes.values() if node.root) == 1
        ):
            self.settings.logger.error(
                "Cannot remove root branch if there is only one root branch."
            )
            raise ValueError(
                "Cannot remove the root branch when there is only one root branch. "
                "Create a new root branch via .add_branch(..., root=True) first. "
                "(You could be trying to replace a root branch with the same ID as the one you are trying to remove. "
                "Try a different name for the new root branch.)"
            )

        for decision_node_id in self.decision_nodes:
            self.decision_nodes[decision_node_id].remove_option(branch_id)

        if branch_id in self.decision_nodes:
            del self.decision_nodes[branch_id]

        # reconstruct tree
        self._get_root()
        self.tree = {}
        self._construct_tree(self.root, self.tree)

    def view(
        self,
        indent: int = 0,
        prefix: str = "",
        max_width: int = 80,
        tree_dict: dict | None = None,
    ):
        """
        Format a tree dictionary into a nice hierarchical text representation.

        Args:
            tree_dict: The tree dictionary to format
            indent: Current indentation level
            prefix: Prefix for the current line (for tree structure visualization)
            max_width: Maximum width for text wrapping

        Returns:
            str: Formatted tree string
        """
        if tree_dict is None:
            tree_dict = self.tree

        result = []

        name = tree_dict.get("name", "Unknown")
        node_id = tree_dict.get("id", "")
        description = tree_dict.get("description", "")
        is_branch = tree_dict.get("branch", False)

        indent_str = "  " * indent
        node_line = (
            f"{indent_str}{prefix}📁 {name}"
            if is_branch
            else f"{indent_str}{prefix}🔧 {name}"
        )

        result.append(node_line)

        if description:
            desc_indent = len(indent_str) + 4  # Extra space for description
            available_width = max_width - desc_indent

            wrapped_desc = textwrap.fill(
                description,
                width=available_width,
                initial_indent="",
                subsequent_indent="",
            )

            for i, line in enumerate(wrapped_desc.split("\n")):
                if i == 0:
                    result.append(f"{indent_str}    💬 {line}")
                else:
                    result.append(f"{indent_str}       {line}")

            result.append("")

        options = tree_dict.get("options", {})
        if options:
            option_items = list(options.items())
            for i, (key, option) in enumerate(option_items):
                is_last = i == len(option_items) - 1
                child_prefix = "└── " if is_last else "├── "
                child_result = self.view(
                    indent + 1, child_prefix, max_width, tree_dict=option
                )
                result.append(child_result)

                if indent == 0 and not is_last:
                    result.append("")

        return "\n".join(result)

    @property
    def conversation_history(self):
        return self.tree_data.conversation_history

    @property
    def environment(self):
        return self.tree_data.environment

    async def create_conversation_title_async(self) -> str:
        """
        Create a title for the tree (async) using the base LM.
        Also assigns the `conversation_title` attribute to the tree.

        Returns:
            (str): The title for the tree.
        """
        with ElysiaKeyManager(self.settings):
            self.conversation_title = await create_conversation_title(
                self.tree_data.conversation_history, self.base_lm
            )
        return self.conversation_title

    def create_conversation_title(self) -> str:
        """
        Create a title for the tree using the base LM.
        Also assigns the `conversation_title` attribute to the tree.

        Returns:
            (str): The title for the tree.
        """
        return asyncio_run(self.create_conversation_title_async())

    async def get_follow_up_suggestions_async(
        self, context: str | None = None, num_suggestions: int = 2
    ) -> list[str]:
        """
        Get follow-up suggestions for the current user prompt via a base model LLM call.

        E.g., if the user asks "What was the most recent Github Issue?",
            and the results show a message from 'Jane Doe',
            the follow-up suggestions might be "What other issues did Jane Doe work on?"

        Args:
            context (str | None): A description of the type of follow-up questions to suggest
            num_suggestions (int): The number of follow-up suggestions to return (length of the list output)

        Returns:
            (list[str]): A list of follow-up suggestions
        """
        with ElysiaKeyManager(self.settings):
            suggestions = await get_follow_up_suggestions(
                self.tree_data,
                self.suggestions,
                self.base_lm,
                context=context,
                num_suggestions=num_suggestions,
            )
        if suggestions != []:
            self.settings.logger.debug(f"Follow-up suggestions: {suggestions}")
        else:
            self.settings.logger.error("No follow-up suggestions found.")

        self.suggestions.extend(suggestions)
        return suggestions

    def get_follow_up_suggestions(
        self,
        context: str | None = None,
        num_suggestions: int = 2,
    ) -> list[str]:
        """
        Get follow-up suggestions for the current user prompt via a base model LLM call (sync wrapper for get_follow_up_suggestions_async).

        E.g., if the user asks "What was the most recent Github Issue?",
            and the results show a message from 'Jane Doe',
            the follow-up suggestions might be "What other issues did Jane Doe work on?"

        Args:
            context (str | None): A description of the type of follow-up questions to suggest
            num_suggestions (int): The number of follow-up suggestions to return (length of the list output)

        Returns:
            (list[str]): A list of follow-up suggestions
        """
        return asyncio_run(
            self.get_follow_up_suggestions_async(context, num_suggestions)
        )

    def _update_conversation_history(self, role: str, message: str) -> None:
        if message != "":
            # If the first message, create a new message
            if len(self.tree_data.conversation_history) == 0:
                self.tree_data.update_list(
                    "conversation_history", {"role": role, "content": message}
                )
            # If the last message is from the same role, append to the content
            elif self.tree_data.conversation_history[-1]["role"] == role:
                if self.tree_data.conversation_history[-1]["content"].endswith(" "):
                    self.tree_data.conversation_history[-1]["content"] += message
                else:
                    self.tree_data.conversation_history[-1]["content"] += " " + message
            # Otherwise, create a new message
            else:
                self.tree_data.update_list(
                    "conversation_history", {"role": role, "content": message}
                )

    def _update_actions_called(self, result: Result, decision: Decision) -> None:
        if self.user_prompt not in self.actions_called:
            self.actions_called[self.user_prompt] = []
            self.actions_called[self.user_prompt].append(
                {
                    "name": decision.function_name,
                    "inputs": decision.function_inputs,
                    "reasoning": decision.reasoning,
                    "output": None,
                }
            )
        if not self.low_memory:
            self.actions_called[self.user_prompt][-1]["output"] = result.objects
        else:
            self.actions_called[self.user_prompt][-1]["output"] = []

    def _add_refs(self, objects: list[dict], tool_name: str, name: str) -> None:

        if (
            tool_name not in self.tree_data.environment.environment
            or name not in self.tree_data.environment.environment[tool_name]
        ):
            len_objects = 0
        else:
            len_objects = len(self.tree_data.environment.environment[tool_name][name])

        for i, obj in enumerate(objects):
            if "_REF_ID" not in obj:
                _REF_ID = f"{tool_name}_{name}_{len_objects}_{i}"
                obj["_REF_ID"] = _REF_ID

    def _update_environment(self, result: Result, decision: Decision) -> None:
        """
        Given a yielded result from an action or otherwise, update the environment.
        I.e. the items within the LLM knowledge base/prompt for future decisions/actions
        All Result subclasses have their .to_json() method added to the environment.
        As well, all Result subclasses have their llm_parse() method added to the tasks_completed.
        """

        # add to environment (store of retrieved/called objects)
        self.tree_data.environment.add(decision.function_name, result)

        # make note of which objects were retrieved _this session_ (for returning)
        if self.store_retrieved_objects:
            self.retrieved_objects.append(result.to_json(mapping=False))

        # add to log of actions called
        self.action_information.append(
            {
                "action_name": decision.function_name,
                **{key: value for key, value in result.metadata.items()},
            }
        )

        # add to tasks completed (parsed info / train of thought for LLM)
        self.tree_data.update_tasks_completed(
            prompt=self.user_prompt,
            task=decision.function_name,
            num_trees_completed=self.tree_data.num_trees_completed,
            reasoning=decision.reasoning,
            inputs=decision.function_inputs,
            parsed_info=result.llm_parse(),
            action=True,
        )

        # add to log of actions called
        self._update_actions_called(result, decision)

    def _add_error(self, function_name: str, error: Error) -> None:
        if function_name not in self.tree_data.errors:
            self.tree_data.errors[function_name] = []

        self.tree_data.update_tasks_completed(
            prompt=self.user_prompt,
            task=function_name,
            num_trees_completed=self.tree_data.num_trees_completed,
            error=True,
        )

        if error.feedback != "An unknown issue occurred.":
            self.tree_data.errors[function_name].append(
                "Avoidable error: "
                f"{error.feedback} "
                "(this error is likely to be solved by incorporating the feedback in a future tool call)"
            )
        else:
            self.tree_data.errors[function_name].append(
                "Unknown error: "
                f"{error.error_message} "
                "(this error is likely outside of your capacity to be solved - "
                "judge the error message based on other information and if it seems fixable, call this tool again "
                "if it is repeated, you may need to try something else or inform the user of the issue)"
            )

    async def _evaluate_result(
        self,
        result: Result | TreeUpdate | Error | TrainingUpdate | Text | Update,
        decision: Decision,
    ) -> tuple[dict | None, bool]:
        error = False

        if isinstance(result, Result):
            self._add_refs(result.objects, decision.function_name, result.name)
            self._update_environment(result, decision)

        if isinstance(result, TrainingUpdate):
            self.training_updates.append(result)
            return None, error

        if isinstance(result, Error):
            self._add_error(decision.function_name, result)
            if self.settings.LOGGING_LEVEL_INT <= 20:
                print(
                    Panel.fit(
                        (
                            result.error_message
                            if result.feedback == "An unknown issue occurred."
                            else result.feedback
                        ),
                        title="Error",
                        border_style="red",
                        padding=(1, 1),
                    )
                )
            error = True

        if isinstance(result, Text):
            self._update_conversation_history("assistant", result.text)
            if self.settings.LOGGING_LEVEL_INT <= 20:
                print(
                    Panel.fit(
                        result.text,
                        title="Assistant response",
                        border_style="cyan",
                        padding=(1, 1),
                    )
                )

        return (
            await self.returner(
                result,
                self.prompt_to_query_id[self.user_prompt],
            ),
            error,
        )

    async def _get_available_tools(
        self, current_decision_node: DecisionNode, client_manager: ClientManager
    ) -> tuple[list[str], list[tuple[str, str]]]:
        available_tools = []
        unavailable_tools = []
        for tool in current_decision_node.options.keys():
            if current_decision_node.options[tool]["action"] is None:
                available_tools.append(tool)
            elif "is_tool_available" in dir(self.tools[tool]) and await self.tools[
                tool
            ].is_tool_available(
                tree_data=self.tree_data,
                base_lm=self.base_lm,
                complex_lm=self.complex_lm,
                client_manager=client_manager,
            ):
                available_tools.append(tool)
            else:
                is_tool_available_doc = (
                    self.tools[tool].is_tool_available.__doc__.strip()
                    if self.tools[tool].is_tool_available.__doc__
                    else ""
                )
                unavailable_tools.append((tool, is_tool_available_doc))
        return available_tools, unavailable_tools

    def _get_successive_actions(
        self, successive_actions: dict, current_options: dict
    ) -> dict:

        for branch in current_options:
            successive_actions[branch] = {}
            if current_options[branch]["options"] != {}:
                successive_actions[branch] = self._get_successive_actions(
                    successive_actions[branch], current_options[branch]["options"]
                )
        return successive_actions

    def log_token_usage(self) -> None:
        if not self.low_memory:
            avg_input_base = self.tracker.get_average_input_tokens("base_lm")
            avg_output_base = self.tracker.get_average_output_tokens("base_lm")
            total_input_base = self.tracker.get_total_input_tokens("base_lm")
            total_output_base = self.tracker.get_total_output_tokens("base_lm")
            avg_input_complex = self.tracker.get_average_input_tokens("complex_lm")
            avg_output_complex = self.tracker.get_average_output_tokens("complex_lm")
            total_input_complex = self.tracker.get_total_input_tokens("complex_lm")
            total_output_complex = self.tracker.get_total_output_tokens("complex_lm")
            total_cost_base = self.tracker.get_total_cost("base_lm")
            total_cost_complex = self.tracker.get_total_cost("complex_lm")
            avg_cost_base = self.tracker.get_average_cost("base_lm")
            avg_cost_complex = self.tracker.get_average_cost("complex_lm")
            num_calls_base = self.tracker.get_num_calls("base_lm")
            num_calls_complex = self.tracker.get_num_calls("complex_lm")

            if num_calls_base > 0:
                self.settings.logger.debug(
                    f"Base Model Usage: \n"
                    f"  - Calls: [magenta]{num_calls_base}[/magenta]\n"
                    f"  - Input Tokens: [magenta]{total_input_base}[/magenta] (Avg. [magenta]{int(avg_input_base)}[/magenta] per call)\n"
                    f"  - Output Tokens: [cyan]{total_output_base}[/cyan] (Avg. [cyan]{int(avg_output_base)}[/cyan] per call)\n"
                    f"  - Total Cost: [yellow]${total_cost_base:.4f}[/yellow] (Avg. [yellow]${avg_cost_base:.4f}[/yellow] per call)\n"
                )
            else:
                self.settings.logger.debug(
                    f"Base Model Usage: [magenta]0[/magenta] calls"
                )
            if num_calls_complex > 0:
                self.settings.logger.debug(
                    f"Complex Model Usage: \n"
                    f"  - Calls: [magenta]{num_calls_complex}[/magenta]\n"
                    f"  - Input Tokens: [magenta]{total_input_complex}[/magenta] (Avg. [magenta]{int(avg_input_complex)}[/magenta] per call)\n"
                    f"  - Output Tokens: [cyan]{total_output_complex}[/cyan] (Avg. [cyan]{int(avg_output_complex)}[/cyan] per call)\n"
                    f"  - Total Cost: [yellow]${total_cost_complex:.4f}[/yellow] (Avg. [yellow]${avg_cost_complex:.4f}[/yellow] per call)\n"
                )
            else:
                self.settings.logger.debug(
                    f"Complex Model Usage: [magenta]0[/magenta] calls"
                )

    async def async_run(
        self,
        user_prompt: str,
        collection_names: list[str] = [],
        client_manager: ClientManager | None = None,
        training_route: str = "",
        query_id: str | None = None,
        close_clients_after_completion: bool = True,
        _first_run: bool = True,
        **kwargs,
    ) -> AsyncGenerator[dict | None, None]:
        """
        Async version of .run() for running Elysia in an async environment.
        See .run() for full documentation.
        """

        if client_manager is None:
            client_manager = ClientManager(
                wcd_url=self.settings.WCD_URL,
                wcd_api_key=self.settings.WCD_API_KEY,
                logger=self.settings.logger,
                client_timeout=None,
                **self.settings.API_KEYS,
            )

        # If training route is provided, split it into a list
        if training_route != "":
            route_list = training_route.split("/")
        else:
            route_list = []

        # Some initial steps if this is the first run (no recursion yet)
        if _first_run:

            self.settings.logger.debug(f"Style: {self.tree_data.atlas.style}")
            self.settings.logger.debug(
                f"Agent description: {self.tree_data.atlas.agent_description}"
            )
            self.settings.logger.debug(f"End goal: {self.tree_data.atlas.end_goal}")

            if query_id is None:
                query_id = str(uuid.uuid4())

            self.returner.add_prompt(user_prompt, query_id)

            # Reset the tree (clear temporary data specific to the last user prompt)
            self.soft_reset()

            check_base_lm_settings(self.settings)
            check_complex_lm_settings(self.settings)

            # Initialise some objects
            self.set_start_time()
            self.query_id_to_prompt[query_id] = user_prompt
            self.prompt_to_query_id[user_prompt] = query_id
            self.tree_data.set_property("user_prompt", user_prompt)
            self._update_conversation_history("user", user_prompt)
            self.user_prompt = user_prompt

            # check and start clients if not already started
            if client_manager.is_client:
                await client_manager.start_clients()

                # Initialise the collections
                if self.use_elysia_collections:
                    if collection_names == []:
                        async with client_manager.connect_to_async_client() as client:
                            collection_names = await retrieve_all_collection_names(
                                client
                            )
                    await self.set_collection_names(
                        collection_names,
                        client_manager,
                    )

            # If there are any empty branches, remove them (no tools attached to them)
            self._remove_empty_branches()

            if self.settings.LOGGING_LEVEL_INT <= 20:
                print(
                    Panel.fit(
                        user_prompt,
                        title="User prompt",
                        border_style="yellow",
                        padding=(1, 1),
                    )
                )

        # Start the tree at the root node
        if self.root is not None:
            current_decision_node: DecisionNode = self.decision_nodes[self.root]
        else:
            raise ValueError("No root node found!")

        # Loop through the tree until the end is reached
        while True:

            available_tools, unavailable_tools = await self._get_available_tools(
                current_decision_node, client_manager
            )

            if len(available_tools) == 0:
                self.settings.logger.error("No tools available to use!")
                raise ValueError(
                    "No tools available to use! "
                    "Check the tool definitions and the `is_tool_available` methods."
                )

            init_options = deepcopy(self.tree["options"])
            successive_actions = self._get_successive_actions(
                successive_actions={},
                current_options=init_options,
            )

            # Evaluate any tools which have hardcoded rules that have been met
            nodes_with_rules_met, rule_tool_inputs = await self._check_rules(
                current_decision_node.id, client_manager
            )

            if len(nodes_with_rules_met) > 0:
                for rule in nodes_with_rules_met:
                    rule_decision = Decision(rule, {}, "", False, False)
                    with ElysiaKeyManager(self.settings):
                        async for result in self.tools[rule](
                            tree_data=self.tree_data,
                            inputs=rule_tool_inputs[rule],
                            base_lm=self.base_lm,
                            complex_lm=self.complex_lm,
                            client_manager=client_manager,
                        ):
                            action_result, _ = await self._evaluate_result(
                                result, rule_decision
                            )
                            if action_result is not None:
                                yield action_result

            # If training route is provided, decide from the training route
            if len(route_list) > 0:
                self.settings.logger.debug(f"Route that will be used: {route_list}")

                (
                    self.current_decision,
                    training_route,
                ) = current_decision_node.decide_from_route(route_list)

                force_text_response = (
                    self.current_decision.function_name == "text_response"
                )

            # Under normal circumstances decide from the decision node
            else:
                self.tracker.start_tracking("decision_node")
                self.tree_data.set_current_task("elysia_decision_node")
                with ElysiaKeyManager(self.settings):
                    self.current_decision, results = await current_decision_node(
                        tree_data=self.tree_data,
                        base_lm=self.base_lm,
                        complex_lm=self.complex_lm,
                        available_tools=available_tools,
                        unavailable_tools=unavailable_tools,
                        successive_actions=successive_actions,
                        client_manager=client_manager,
                    )

                for result in results:
                    action_result, _ = await self._evaluate_result(
                        result, self.current_decision
                    )
                    if action_result is not None:
                        yield action_result

                self.tracker.end_tracking(
                    "decision_node",
                    "Decision Node",
                    self.base_lm if not self.low_memory else None,
                    self.complex_lm if not self.low_memory else None,
                )

                # Force text response (later) if model chooses end actions
                # but no response will be generated from the node, set flag now
                force_text_response = (
                    not current_decision_node.options[
                        self.current_decision.function_name
                    ]["end"]
                    and self.current_decision.end_actions
                )

            # Set default values for the function inputs for current call
            self.current_decision.function_inputs = self._get_function_inputs(
                self.current_decision.function_name,
                self.current_decision.function_inputs,
            )

            # end criteria, task picked is "text_response" or model chooses to end conversation
            completed = (
                self.current_decision.function_name == "text_response"
                or self.current_decision.end_actions
                or self.current_decision.impossible
                or self.tree_data.num_trees_completed > self.tree_data.recursion_limit
            )

            # assign action function
            action_fn: Tool | None = current_decision_node.options[
                self.current_decision.function_name
            ][
                "action"
            ]  # type: ignore

            # update the decision history
            self.decision_history[-1].append(self.current_decision.function_name)

            # print the current node information
            if self.settings.LOGGING_LEVEL_INT <= 20:
                print(
                    Panel.fit(
                        f"[bold]Node:[/bold] [magenta]{current_decision_node.id}[/magenta]\n"
                        f"[bold]Decision:[/bold] [green]{self.current_decision.function_name}[/green]\n"
                        f"[bold]Reasoning:[/bold] {self.current_decision.reasoning}\n",
                        title="Current Decision",
                        border_style="magenta",
                        padding=(1, 1),
                    )
                )

            self.tree_data.update_tasks_completed(
                prompt=self.user_prompt,
                task=self.current_decision.function_name,
                num_trees_completed=self.tree_data.num_trees_completed,
                reasoning=self.current_decision.reasoning,
                action=action_fn is not None,
            )

            # evaluate the action if this is not a branch
            if action_fn is not None:
                self.tracker.start_tracking(self.current_decision.function_name)
                self.tree_data.set_current_task(self.current_decision.function_name)
                successful_action = True
                with ElysiaKeyManager(self.settings):
                    async for result in action_fn(
                        tree_data=self.tree_data,
                        inputs=self.current_decision.function_inputs,
                        base_lm=self.base_lm,
                        complex_lm=self.complex_lm,
                        client_manager=client_manager,
                        **kwargs,
                    ):
                        action_result, error = await self._evaluate_result(
                            result, self.current_decision
                        )

                        if action_result is not None:
                            yield action_result

                        successful_action = not error and successful_action

                if not successful_action:
                    completed = (
                        False
                        or self.tree_data.num_trees_completed
                        > self.tree_data.recursion_limit
                    )

                if successful_action:
                    self.tree_data.clear_error(self.current_decision.function_name)

                self.tracker.end_tracking(
                    self.current_decision.function_name,
                    self.current_decision.function_name,
                    self.base_lm if not self.low_memory else None,
                    self.complex_lm if not self.low_memory else None,
                )

            yield (
                await self._evaluate_result(
                    TreeUpdate(
                        from_node=current_decision_node.id,
                        to_node=self.current_decision.function_name,
                        reasoning=(
                            self.current_decision.reasoning
                            if self.settings.BASE_USE_REASONING
                            else ""
                        ),
                        reset_tree=current_decision_node.options[
                            self.current_decision.function_name
                        ]["next"]
                        is None
                        and (not completed),
                    ),
                    self.current_decision,
                )
            )[0]

            # check if the current node is the end of the tree
            if (
                current_decision_node.options[self.current_decision.function_name][
                    "next"
                ]
                is None
                or completed
            ):
                break
            else:
                current_decision_node = current_decision_node.options[
                    self.current_decision.function_name
                ][
                    "next"
                ]  # type: ignore

        self.tree_data.num_trees_completed += 1

        # end of all trees
        if completed:

            # firstly, if we reached the end of a tree at a node that shouldn't be the end, call text response tool here to respond
            if (
                not current_decision_node.options[self.current_decision.function_name][
                    "end"
                ]
                or force_text_response
            ):
                with ElysiaKeyManager(self.settings):
                    async for result in self.tools["forced_text_response"](
                        tree_data=self.tree_data,
                        inputs={},
                        base_lm=self.base_lm,
                        complex_lm=self.complex_lm,
                    ):
                        action_result, _ = await self._evaluate_result(
                            result, self.current_decision
                        )
                        if action_result is not None:
                            yield action_result

            self.save_history(
                query_id=self.prompt_to_query_id[user_prompt],
                time_taken_seconds=time.time() - self.start_time,
            )

            yield await self.returner(
                Completed(), query_id=self.prompt_to_query_id[user_prompt]
            )

            self.settings.logger.debug(
                f"[bold green]Model identified overall goal as completed![/bold green]"
            )
            self.settings.logger.debug(
                f"Total time taken for decision tree: {time.time() - self.start_time:.2f} seconds"
            )
            self.settings.logger.debug(
                f"Decision Node Avg. Time: {self.tracker.get_average_time('decision_node'):.2f} seconds"
            )
            self.log_token_usage()

            avg_times = []
            for i, iteration in enumerate(self.decision_history):
                if iteration != []:
                    avg_times = [
                        (
                            f"  - {task} ([magenta]Avg. {self.tracker.get_average_time(task):.2f} seconds[/magenta])\n"
                            if task in self.tracker.trackers
                            else ""
                        )
                        for task in iteration
                    ]
                    self.settings.logger.debug(
                        f"Tasks completed (iteration {i+1}):\n" + "".join(avg_times)
                    )

            if close_clients_after_completion and client_manager.is_client:
                await client_manager.close_clients()

        # otherwise, end of the tree for this iteration, and recursively call process() to restart the tree
        else:
            self.settings.logger.debug(
                f"Model did [bold red]not[/bold red] yet complete overall goal! "
            )
            self.settings.logger.debug(
                f"Restarting tree (Recursion: {self.tree_data.num_trees_completed+1}/{self.tree_data.recursion_limit})..."
            )

            # recursive call to restart the tree since the goal was not completed
            self.decision_history.append([])
            async for result in self.async_run(
                user_prompt,
                collection_names,
                client_manager,
                training_route=training_route,
                query_id=query_id,
                _first_run=False,
            ):
                yield result

    def run(
        self,
        user_prompt: str,
        collection_names: list[str] = [],
        client_manager: ClientManager | None = None,
        training_route: str = "",
        query_id: str | None = None,
        close_clients_after_completion: bool = True,
    ) -> tuple[str, list[dict]]:
        """
        Run the Elysia decision tree.

        Args:
            user_prompt (str): The input from the user.
            collection_names (list[str]): The names of the collections to use.
                If not provided, Elysia will attempt to retrieve all collection names from the client.
            client_manager (ClientManager): The client manager to use.
                If not provided, a new ClientManager will be created.
            training_route (str): The route to use for training.
                Separate tools/branches you want to use with a "/".
                e.g. "query/text_response" will only use the "query" tool and the "text_response" tool, and end the tree there.
            query_id (str): The id of the query.
                Only necessary if you are hosting Elysia on a server with multiple users.
                If not provided, a new query id will be generated.
            close_clients_after_completion (bool): Whether to close the clients after the tree is completed.
                Leave as True for most use cases, but if you don't want to close the clients for the ClientManager, set to False.
                For example, if you are managing your own clients (e.g. in an app), you may want to set this to False.

        Returns:
            (str): The concatenation of all the responses from the tree.
            (list[dict]): The retrieved objects from the tree.
        """

        self.store_retrieved_objects = True

        async def run_process():
            async for result in self.async_run(
                user_prompt,
                collection_names,
                client_manager,
                training_route,
                query_id,
                close_clients_after_completion,
            ):
                pass
            return self.retrieved_objects

        async def run_with_live():
            console = Console()

            with console.status("[bold indigo]Thinking...") as status:
                async for result in self.async_run(
                    user_prompt,
                    collection_names,
                    client_manager,
                    training_route,
                    query_id,
                    close_clients_after_completion,
                ):
                    if (
                        result is not None
                        and "type" in result
                        and result["type"] == "status"
                        and isinstance(result["payload"], dict)
                        and "text" in result["payload"]
                    ):
                        payload: dict = result["payload"]  # type: ignore
                        status.update(f"[bold indigo]{payload['text']}")

            return self.retrieved_objects

        if self.settings.LOGGING_LEVEL_INT <= 20:
            yielded_results = asyncio_run(run_with_live())
        else:
            yielded_results = asyncio_run(run_process())

        text = self.tree_data.conversation_history[-1]["content"]

        return text, yielded_results

    def detailed_memory_usage(self) -> dict:
        """
        Returns a detailed breakdown of memory usage for all major objects in the Tree class.

        Returns:
            dict: Dictionary containing memory sizes (in bytes) for each major component
        """
        memory_usage = {}

        # Core data structures
        memory_usage["tree_data"] = asizeof.asizeof(self.tree_data)
        memory_usage["tree_data"] = asizeof.asizeof(self.tree_data)
        memory_usage["tree_data"] = asizeof.asizeof(self.tree_data)
        memory_usage["collection_data"] = asizeof.asizeof(
            self.tree_data.collection_data
        )

        # Decision nodes and tools
        memory_usage["decision_nodes"] = {
            node_id: asizeof.asizeof(node)
            for node_id, node in self.decision_nodes.items()
        }
        memory_usage["tools"] = {
            tool_name: asizeof.asizeof(tool) for tool_name, tool in self.tools.items()
        }

        # History and tracking
        memory_usage["decision_history"] = asizeof.asizeof(self.decision_history)
        memory_usage["history"] = asizeof.asizeof(self.history)
        memory_usage["training_updates"] = asizeof.asizeof(self.training_updates)
        memory_usage["action_information"] = asizeof.asizeof(self.action_information)

        # Mapping dictionaries
        memory_usage["query_mappings"] = {
            "query_id_to_prompt": asizeof.asizeof(self.query_id_to_prompt),
            "prompt_to_query_id": asizeof.asizeof(self.prompt_to_query_id),
        }

        # Calculate total
        memory_usage["total"] = sum(
            (v if isinstance(v, int) else sum(v.values()) if isinstance(v, dict) else 0)
            for v in memory_usage.values()
        )

        return memory_usage

    def export_to_json(self) -> dict:
        """
        Export the tree to a JSON object, to be used for loading the tree via import_from_json().

        Returns:
            (dict): The JSON object.
        """
        try:
            return {
                "user_id": self.user_id,
                "conversation_id": self.conversation_id,
                "conversation_title": self.conversation_title,
                "branch_initialisation": self.branch_initialisation,
                "use_elysia_collections": self.use_elysia_collections,
                "tree_index": self.tree_index,
                "store_retrieved_objects": self.store_retrieved_objects,
                "low_memory": self.low_memory,
                "tree_data": self.tree_data.to_json(remove_unserialisable=True),
                "settings": self.settings.to_json(),
                "tool_names": list(self.tools.keys()),
                "frontend_rebuild": self.returner.store,
            }
        except Exception as e:
            self.settings.logger.error(f"Error exporting tree to JSON: {str(e)}")
            raise e

    async def export_to_weaviate(
        self, collection_name: str, client_manager: ClientManager | None = None
    ) -> None:
        """
        Export the tree to a Weaviate collection.

        Args:
            collection_name (str): The name of the collection to export to.
            client_manager (ClientManager): The client manager to use.
                If not provided, a new ClientManager will be created from environment variables.
        """
        if client_manager is None:
            client_manager = ClientManager()
            close_after_use = True
        else:
            close_after_use = False

        async with client_manager.connect_to_async_client() as client:

            if not await client.collections.exists(collection_name):
                await client.collections.create(
                    collection_name,
                    vectorizer_config=wc.Configure.Vectorizer.none(),
                    inverted_index_config=wc.Configure.inverted_index(
                        index_timestamps=True
                    ),
                    properties=[
                        wc.Property(
                            name="user_id",
                            data_type=wc.DataType.TEXT,
                        ),
                        wc.Property(
                            name="conversation_id",
                            data_type=wc.DataType.TEXT,
                        ),
                        wc.Property(
                            name="tree",
                            data_type=wc.DataType.TEXT,
                        ),
                        wc.Property(
                            name="title",
                            data_type=wc.DataType.TEXT,
                        ),
                    ],
                )

            collection = client.collections.get(collection_name)

            json_data_str = json.dumps(self.export_to_json())

            uuid = generate_uuid5(self.conversation_id)

            if await collection.data.exists(uuid):
                await collection.data.update(
                    uuid=uuid,
                    properties={
                        "user_id": self.user_id,
                        "conversation_id": self.conversation_id,
                        "tree": json_data_str,
                        "title": self.conversation_title,
                    },
                )
                self.settings.logger.info(
                    f"Successfully updated existing tree in collection '{collection_name}' with id '{self.conversation_id}'"
                )
            else:
                await collection.data.insert(
                    uuid=uuid,
                    properties={
                        "user_id": self.user_id,
                        "conversation_id": self.conversation_id,
                        "tree": json_data_str,
                        "title": self.conversation_title,
                    },
                )
                self.settings.logger.info(
                    f"Successfully inserted new tree in collection '{collection_name}' with id '{self.conversation_id}'"
                )

        if close_after_use:
            await client_manager.close_clients()

    @classmethod
    def import_from_json(cls, json_data: dict) -> "Tree":
        """
        Import a tree from a JSON object, outputted by the export_to_json() method.

        Args:
            json_data (dict): The JSON object to import the tree from.

        Returns:
            (Tree): The new tree instance loaded from the JSON object.
        """
        settings = Settings.from_json(json_data["settings"])
        logger = settings.logger
        tree = cls(
            user_id=json_data["user_id"],
            conversation_id=json_data["conversation_id"],
            branch_initialisation=json_data["branch_initialisation"],
            style=json_data["tree_data"]["atlas"]["style"],
            agent_description=json_data["tree_data"]["atlas"]["agent_description"],
            end_goal=json_data["tree_data"]["atlas"]["end_goal"],
            low_memory=json_data["low_memory"],
            use_elysia_collections=json_data["use_elysia_collections"],
            settings=settings,
        )

        tree.returner.store = json_data["frontend_rebuild"]
        tree.tree_data = TreeData.from_json(json_data["tree_data"])
        tree.set_branch_initialisation(json_data["branch_initialisation"])

        # check tools
        for tool_name in json_data["tool_names"]:
            if tool_name not in tree.tools:
                logger.warning(
                    f"In saved tree, custom tool '{tool_name}' found. "
                    "This will not be loaded in the new tree. "
                    "You will need to add it to the tree manually."
                )

        return tree

    @classmethod
    async def import_from_weaviate(
        cls,
        collection_name: str,
        conversation_id: str,
        client_manager: ClientManager | None = None,
    ) -> "Tree":
        """
        Import a tree from a Weaviate collection.

        Args:
            collection_name (str): The name of the collection to import from.
            conversation_id (str): The id of the conversation to import.
            client_manager (ClientManager): The client manager to use.
                If not provided, a new ClientManager will be created from environment variables.

        Returns:
            (Tree): The tree object.
        """

        if client_manager is None:
            client_manager = ClientManager()
            close_after_use = True
        else:
            close_after_use = False

        async with client_manager.connect_to_async_client() as client:

            if not await client.collections.exists(collection_name):
                raise ValueError(
                    f"Collection '{collection_name}' does not exist in this Weaviate instance."
                )

            collection = client.collections.get(collection_name)
            uuid = generate_uuid5(conversation_id)
            # if not await collection.data.exists(uuid):
            #     raise ValueError(
            #         f"No tree found for conversation id '{conversation_id}' in collection '{collection_name}'."
            #     )

            response = await collection.query.fetch_object_by_id(uuid)

        if close_after_use:
            await client_manager.close_clients()

        if response is None:
            raise ValueError(
                f"No tree found for conversation id '{conversation_id}' in collection '{collection_name}'."
            )

        json_data_str = response.properties["tree"]
        json_data = json.loads(json_data_str)  # type: ignore

        return cls.import_from_json(json_data)

    def __call__(self, *args, **kwargs) -> tuple[str, list[dict]]:
        return self.run(*args, **kwargs)
