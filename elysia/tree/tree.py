import inspect
import json
import time
import textwrap
from copy import deepcopy
from typing import AsyncGenerator, Literal, Any

import dspy
from pydantic.v1.typing import NoneType
from rich import print
from rich.console import Console
from rich.panel import Panel

import uuid
from datetime import datetime

# Weaviate
import weaviate.classes.config as wc
from weaviate.util import generate_uuid5

# Elysia
from elysia.config import Settings, ElysiaKeyManager
from elysia.config import settings as environment_settings
from elysia.tree.util import (
    Node,
    TreeReturner,
    ToolOption,
    ToolInput,
    Decision,
    get_follow_up_suggestions,
    create_conversation_title,
)
from elysia.objects import (
    Completed,
    Result,
    StreamedReturn,
    Update,
    Text,
    Tool,
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
from elysia.util.streaming import StreamedParserFactory
from elysia.config import (
    check_base_lm_settings,
    check_complex_lm_settings,
    load_base_lm,
    load_complex_lm,
)
from elysia.util.objects import (
    Tracker,
    TrainingUpdate,
    EdgeUpdate,
    ViewEnvironment,
    GraphUpdate,
)
from elysia.util.collection import retrieve_all_collection_names
from elysia.util.feedback import create_feedback_collection
from elysia.util.parsing import format_datetime


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
        preset_id: str | None = None,
        low_memory: bool = False,
        use_weaviate_collections: bool = True,
        streaming: bool = False,
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
            use_weaviate_collections (bool): Whether to use weaviate collections as processed by Elysia.
                If False, the tree will not use the processed collections.
            settings (Settings): The settings for the tree, an object of elysia.Settings.
                This is automatically set to the environment settings if not provided.
        """
        # Define base variables of the tree
        self.user_id = user_id if user_id is not None else str(uuid.uuid4())
        self.conversation_id = (
            conversation_id if conversation_id is not None else str(uuid.uuid4())
        )
        self.preset_id = preset_id

        if settings is not None and not isinstance(settings, Settings):
            raise TypeError("settings must be an instance of Settings")
        self.settings = settings if settings is not None else environment_settings

        # Initialise some tree variables
        self.nodes: dict[str, Node] = {}
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
        self.streamer = StreamedParserFactory()

        # Define the inputs to prompts
        self.tree_data = TreeData(
            user_id=self.user_id,
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
            use_weaviate_collections=use_weaviate_collections,
            streaming=streaming,
        )

        # initialise the timers
        self.tracker = Tracker(
            tracker_names=["decision_node"],
            logger=self.settings.logger,
        )

        # console and status for rich output (set during run_with_live)
        self._console: Console | None = None
        self._status = None

        # Set the initialisations
        self.tools: dict[str, Tool] = {}
        self.set_branch_initialisation(branch_initialisation)
        self.tree_data.atlas.style = style
        self.tree_data.atlas.agent_description = agent_description
        self.tree_data.atlas.end_goal = end_goal

        self.tools["forced_text_response"] = ForcedTextResponse()

        # some variables for storing feedback
        self.action_information = []
        self.history = {}
        self.training_updates = []

        # Get the root node
        self._get_root()

        # initialise the returner (for frontend)
        self.returner = TreeReturner(
            user_id=self.user_id,
            conversation_id=self.conversation_id,
        )

        # Print the tree if required
        self.settings.logger.debug(
            "Initialised tree with the following decision nodes:"
        )
        for node in self.nodes.values():
            if len(node.options) > 0:
                self.settings.logger.debug(
                    f"  - [magenta]{node.name}[/magenta]: {[self.nodes[o].name for o in node.options]}"
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

    _BASE_INSTRUCTION_SIMPLE = """
        Choose a base-level task based on the user's prompt and available information.
        Decide based on the tools you have available as well as their descriptions.
        Read them thoroughly and match the actions to the user prompt.
    """

    _BASE_INSTRUCTION_MULTI = """
        Choose a base-level task based on the user's prompt and available information.
        You can search, which includes aggregating or querying information - this should be used if the user needs (more) information.
        You can end the conversation by choosing text response, or summarise some retrieved information.
        Base your decision on what information is available and what the user is asking for - you can search multiple times if needed,
        but you should not search if you have already found all the information you need.
    """

    _SEARCH_INSTRUCTION = """
        Choose between querying the knowledge base via semantic/keyword search, or aggregating information by performing operations, on the knowledge base.
        Querying is when the user is looking for specific information related to the content of the dataset, requiring a specific search query. This is for retrieving specific information via a _query_, similar to a search engine.
        Aggregating is when the user is looking for a specific operations on the dataset, such as summary statistics of the quantity of some items. Aggregation can also include grouping information by some property and returning statistics about the groups.
    """

    _SEARCH_DESCRIPTION = """
        Search the knowledge base. This should be used when the user is lacking information for this particular prompt. This retrieves information only and provides no output to the user except the information.
        Choose to query (semantic or keyword search on a knowledge base), or aggregate information (calculate properties/summary statistics/averages and operations on the knowledge bases).
    """

    def _add_base_tools(
        self, base_id: str, include_search_tools: bool = True
    ) -> str | None:
        self.add_tool(from_node_id=base_id, tool=CitedSummarizer, end=True)
        self.add_tool(from_node_id=base_id, tool=FakeTextResponse, end=True)

        if not include_search_tools:
            return None

        self.add_tool(from_node_id=base_id, tool=Aggregate, end=False)
        query_id = self.add_tool(
            from_node_id=base_id, tool=Query, summariser_in_tree=True, end=False
        )
        self.add_tool(from_node_id=base_id, tool=Visualise, end=False)
        return query_id

    def multi_branch_init(self) -> None:
        base_id = self.add_branch(
            name="base",
            instruction=self._BASE_INSTRUCTION_MULTI,
            status="Choosing a base-level task...",
        )
        self.add_tool(from_node_id=base_id, tool=CitedSummarizer, end=True)
        self.add_tool(from_node_id=base_id, tool=FakeTextResponse, end=True)

        search_id = self.add_branch(
            name="search",
            from_node_id=base_id,
            instruction=self._SEARCH_INSTRUCTION,
            description=self._SEARCH_DESCRIPTION,
            status="Searching the knowledge base...",
        )
        query_id = self.add_tool(
            from_node_id=search_id, tool=Query, summariser_in_tree=True, end=False
        )
        self.add_tool(from_node_id=search_id, tool=Aggregate, end=False)
        self.add_tool(from_node_id=base_id, tool=Visualise, end=False)
        self.add_tool(SummariseItems, from_node_id=query_id)

    def one_branch_init(self) -> None:
        base_id = self.add_branch(
            name="base",
            instruction=self._BASE_INSTRUCTION_SIMPLE,
            status="Choosing a base-level task...",
        )
        query_id = self._add_base_tools(base_id, include_search_tools=True)
        if query_id:
            self.add_tool(SummariseItems, from_node_id=query_id)

    def empty_init(self) -> None:
        self.add_branch(
            name="base",
            instruction=self._BASE_INSTRUCTION_SIMPLE,
            status="Choosing a base-level task...",
        )

    def clear_tree(self) -> None:
        self.nodes = {}
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
        self.root = None
        for node_id, node in self.nodes.items():
            if node.root:
                if self.root is not None and self.root != node_id:
                    raise ValueError("Multiple root nodes found")

                self.root = node_id

        if self.root is None:
            raise ValueError("No root node found")

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

    async def _check_rules(
        self, node: Node, client_manager: ClientManager
    ) -> tuple[list[str], dict]:

        nodes_with_rules_met = []
        rule_tool_inputs = {}
        for node_id in node.options:
            function_name = self.nodes[node_id].name

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
        tool: type[Tool] | Tool | str,
        from_node_id: str | None = None,
        node_id: str | None = None,
        description: str | None = None,
        **kwargs,
    ) -> str | None:
        """
        Add a Tool to a branch or on top of an existing tool.
        Creates a new node in the tree corresponding to that tool call.
        The tool needs to be an instance of the Tool class.

        Args:
            tool (Tool | type[Tool] | str): The tool to add. If a string, then the tool should already have been registered in the tree.
                (i.e. via a separate `.add_tool()` call with a Tool class).
            from_node_id (str): The ID of the node to add the tool to. If not specified, the tool will be added to the root node.
            node_id (str): The ID of the node created. If not specified, a new random node ID will be generated and returned.
            description (str): The description of the tool. If not specified, the `tool.description` attribute will be used.
            kwargs (any): Additional keyword arguments to pass to the initialisation of the tool

        Returns:
            (str): The node ID of the tool node created.

        Example:
            To add a tool without any other setup, you can just call it directly.
            ```python
            tree.add_tool(Query)
            ```
            This will add the `Query` tool to the root node.

        Example:
            A more complicated setup involves adding tools to branches (see the .add_branch() method for more details).
            ```python
            search_branch_id = tree.add_branch("search")
            query_node_id = tree.add_tool(Query, from_node_id=search_branch_id)
            aggregate_node_id = tree.add_tool(Aggregate, from_node_id=search_branch_id)
            ```
            This adds two tools (Query, Aggregate) to the "search" branch.
            You can add a tool, `CheckResult`, after the 'query' tool like this:
            ```python
            tree.add_tool(CheckResult, from_node_id=query_node_id)
            ```
            This will add the `CheckResult` tool to the "search" branch, after the 'query' tool.
            This essentially makes the Query tool a branch with a single option.
            To add the same CheckResult tool to the Aggregate tool, you can do this:
            ```python
            tree.add_tool("check_result", from_node_id=aggregate_node_id)
            ```
            Where we assume "check_result" is the name property of the CheckResult tool.
            You can add via
            ```python
            tree.add_tool(CheckResult, from_node_id=aggregate_node_id)
            ```
            which will do the same thing.
        """

        if not isinstance(tool, str):
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

        if isinstance(tool, str):
            if tool not in self.tools:
                raise ValueError(
                    f"Tool specified as string, but '{tool}' not found in tree (self.tools)."
                )
            tool_instance = self.tools[tool]
        elif isinstance(tool, Tool):
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

        if node_id is None:
            node_id = str(uuid.uuid4())

        if from_node_id and from_node_id not in self.nodes:
            raise ValueError(f"Node with ID '{from_node_id}' not found.")

        if not isinstance(tool, str):
            if tool_instance.name in self.tools:
                self.settings.logger.warning(
                    f"Tool '{tool_instance.name}' already exists in tree (self.tools). Overwriting."
                )
                del self.tools[tool_instance.name]

            self.tools[tool_instance.name] = tool_instance

            self.nodes[node_id] = Node(
                id=node_id,
                name=tool_instance.name,
                description=description if description else tool_instance.description,
                branch=False,
                root=False,
                options=[],
                end=tool_instance.end,
                status=tool_instance.status,
            )

            self.tracker.add_tracker(tracker_name=tool_instance.name)
            self._get_root()
        else:
            node_id = next(node.id for node in self.nodes.values() if node.name == tool)

        if from_node_id:
            self.nodes[from_node_id].options.append(node_id)
        elif self.root:
            self.nodes[self.root].options.append(node_id)
        else:
            raise ValueError("No root node found and no from_node_id provided.")

        return None if isinstance(tool, str) else node_id

    def remove_node(self, node_id: str) -> None:
        """
        Remove a single node from the tree based on its ID.

        Args:
            node_id (str): The ID of the node to remove.

        You can retrieve the node ID from either the returning of the `add_tool()` or `add_branch()` methods, or by cycling through the `self.nodes` dictionary.

        Example:
            ```python
            node_id = tree.add_tool(Query)
            tree.remove_node(node_id)
            ```

            ```python
            for node in tree.nodes.values():
                if node.name == "Query":
                    tree.remove_node(node.id)
            ```
        """

        if node_id not in self.nodes:
            raise ValueError(f"Node with ID '{node_id}' not found.")

        del self.nodes[node_id]
        for node in self.nodes.values():
            if node_id in node.options:
                node.options.remove(node_id)

    def remove_tool(self, tool_name: str) -> None:
        """
        Remove a Tool from a completely from the tree. Purges all instances of nodes with this tool across the tree.

        Args:
            tool_name (str): The name of the tool to remove.
        """

        if tool_name not in self.tools:
            raise ValueError(f"Tool '{tool_name}' not found in tree (self.tools).")

        nodes_to_remove = []
        for node in self.nodes.values():
            if node.name == tool_name:
                nodes_to_remove.append(node.id)

        for node_id in nodes_to_remove:
            for node in self.nodes.values():
                if node_id in node.options:
                    node.options.remove(node_id)
            del self.nodes[node_id]

        del self.tools[tool_name]
        self.tracker.remove_tracker(tool_name)

    def add_branch(
        self,
        name: str,
        instruction: str = "",
        description: str = "",
        node_id: str | None = None,
        from_node_id: str | None = None,
        status: str = "",
    ) -> str:
        """
        Add a branch to the tree. Creates a new node in the tree corresponding to that branch.
        Branches are a decision point in the tree, but they are not associated with a specific tool call.
        They can be seen as a categorisation of different tools.
        So if you have two similar tools that perform a 'search functionality for example, you could create a branch called 'Search' and add both tools to it.
        The decision model will first choose a branch (giving it the same weight as tools), then in a separate call, choose a tool (or another branch) under that branch.
        Tool nodes themselves _can_ also have multiple options afterwards, but they are associated with a function call when selected.
        Branches are specifically _only_ used as categorisation.

        Args:
            name (str): The name of the branch.
            instruction (str): The general instruction for the branch, what is this branch containing?
                What kind of tools or actions are being decided on this branch?
                Only displayed to the decision maker when this branch is chosen.
            description (str): A description of the branch, if it is to be chosen from a previous branch.
                How does the model know whether to choose this branch or not?
            node_id (str): The id of the branch node being added. If not specified, a new random node ID will be generated and returned.
            from_node_id (str): The id of the preceding node to add the branch to. If not specified, the branch will be the root node.
            status (str): The status message to be displayed when this branch is chosen.
        """
        if from_node_id is not None and description == "":
            raise ValueError("Description is required for non-root branches.")

        if from_node_id is None and description != "":
            self.settings.logger.warning("Description is not used for root branches.")

        if node_id is None:
            node_id = str(uuid.uuid4())

        self.nodes[node_id] = Node(
            id=node_id,
            name=name,
            branch=True,
            root=from_node_id is None,
            options=[],
            end=False,
            status=status,
            instruction=instruction,
            description=description,
        )

        if from_node_id:
            self.nodes[from_node_id].options.append(node_id)

        self._get_root()

        return node_id

    @property
    def edges(self):
        edges = []
        for node in self.nodes.values():
            for option in node.options:
                edges.append((node.id, option))
        return edges

    def _view(
        self,
        node_id: str | None = None,
        indent: int = 0,
        prefix: str = "",
        max_width: int = 80,
    ) -> str:
        """
        Format the tree into a nice hierarchical text representation.

        Args:
            node_id (str | None): The ID of the node to start from. If None, starts from root.
            indent (int): Current indentation level
            prefix (str): Prefix for the current line (for tree structure visualization)
            max_width (int): Maximum width for text wrapping

        Returns:
            (str): Formatted tree string
        """
        if node_id is None:
            if self.root is None:
                return "No root node found"
            node_id = self.root

        if node_id not in self.nodes:
            return f"Node {node_id} not found"

        node = self.nodes[node_id]
        result = []

        # Format the node name
        indent_str = "  " * indent
        icon = "📁" if node.branch else "🔧"
        node_line = f"{indent_str}{prefix}{icon} {node.name}"

        # Add end indicator if applicable
        if node.end:
            node_line += " [END]"

        result.append(node_line)

        # Add description if present
        if node.description:
            desc_indent = len(indent_str) + 4
            available_width = max_width - desc_indent

            desc = node.description
            desc = " ".join(desc.split())
            desc = desc.strip()

            wrapped_desc = textwrap.fill(
                desc,
                width=available_width,
                initial_indent="",
                subsequent_indent="",
            )

            for i, line in enumerate(wrapped_desc.split("\n")):
                if i == 0:
                    result.append(f"{indent_str}    📝 {line}")
                else:
                    result.append(f"{indent_str}      {line}")

        # Add instruction if present (for branches)
        if node.instruction and node.branch:
            inst_indent = len(indent_str) + 4
            available_width = max_width - inst_indent

            inst = " ".join(node.instruction.split())
            inst = inst.strip()

            wrapped_inst = textwrap.fill(
                inst,
                width=available_width,
                initial_indent="",
                subsequent_indent="",
            )

            for i, line in enumerate(wrapped_inst.split("\n")):
                if i == 0:
                    result.append(f"{indent_str}    💬 {line}")
                else:
                    result.append(f"{indent_str}      {line}")

        # Add spacing after descriptions/instructions
        if node.description or (node.instruction and node.branch):
            result.append("")

        # Recursively process child options
        if node.options:
            for i, option_id in enumerate(node.options):
                is_last = i == len(node.options) - 1
                child_prefix = "└── " if is_last else "├── "
                child_result = self._view(
                    option_id, indent + 1, child_prefix, max_width
                )
                result.append(child_result)

                # Add spacing between top-level children
                if indent == 0 and not is_last:
                    result.append("")

        return "\n".join(result)

    def view(self):
        result = self._view()
        print(result)

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
        Sync wrapper for create_conversation_title_async().
        Create a title for the tree using the base LM.

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
            if len(self.tree_data.conversation_history) == 0:
                self.tree_data.update_list(
                    "conversation_history", {"role": role, "content": message}
                )
            elif self.tree_data.conversation_history[-1]["role"] == role:
                if self.tree_data.conversation_history[-1]["content"].endswith(" "):
                    self.tree_data.conversation_history[-1]["content"] += message
                else:
                    self.tree_data.conversation_history[-1]["content"] += " " + message
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

    def _add_refs(self, objects: list[dict], tool_name: str) -> None:

        if tool_name not in self.environment.environment:
            len_results = 0
        else:
            len_results = len(self.environment.environment[tool_name])

        for i, obj in enumerate(objects):
            if "_REF_ID" not in obj:
                _REF_ID = f"{tool_name}_{len_results}_{i}"
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
            num_items=len(result.objects),
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

        is_avoidable = error.feedback != "An unknown issue occurred."
        error_msg = (
            f"Avoidable error: {error.feedback} "
            "(this error is likely to be solved by incorporating the feedback in a future tool call)"
            if is_avoidable
            else f"Unknown error: {error.error_message} "
            "(this error is likely outside of your capacity to be solved - "
            "judge the error message based on other information and if it seems fixable, call this tool again "
            "if it is repeated, you may need to try something else or inform the user of the issue)"
        )
        self.tree_data.errors[function_name].append(error_msg)

    def _reset_view_environment_vars(self):
        self.tree_data.view_env_vars = None

    def _set_view_environment_vars(self, result):
        self.tree_data.view_env_vars = result

    def _print_panel(self, content: str, title: str, style: str) -> None:
        """Print a panel, pausing the status spinner if active to avoid overlap."""
        if self.settings.LOGGING_LEVEL_INT <= 20:
            panel = Panel.fit(content, title=title, border_style=style, padding=(1, 1))
            if self._status is not None:
                self._status.stop()
                if self._console is not None:
                    self._console.print(panel)
                else:
                    print(panel)
                self._status.start()
            else:
                print(panel)

    def _print_decision_panel(self, node_name: str, decision: Decision) -> None:
        self._print_panel(
            f"[bold]Node:[/bold] [magenta]{node_name}[/magenta]\n"
            f"[bold]Decision:[/bold] [green]{decision.function_name}[/green]\n"
            f"[bold]Reasoning:[/bold] {decision.reasoning}\n",
            "Current Decision",
            "magenta",
        )

    def _log_completion_stats(self) -> None:
        self.settings.logger.debug(
            "[bold green]Model identified overall goal as completed![/bold green]"
        )
        self.settings.logger.debug(
            f"Total time taken for decision tree: {time.time() - self.start_time:.2f} seconds"
        )
        self.settings.logger.debug(
            f"Decision Node Avg. Time: {self.tracker.get_average_time('decision_node'):.2f} seconds"
        )
        self.log_token_usage()

        for i, iteration in enumerate(self.decision_history):
            if iteration:
                avg_times = [
                    f"  - {task} ([magenta]Avg. {self.tracker.get_average_time(task):.2f} seconds[/magenta])\n"
                    for task in iteration
                    if task in self.tracker.trackers
                ]
                if avg_times:
                    self.settings.logger.debug(
                        f"Tasks completed (iteration {i+1}):\n" + "".join(avg_times)
                    )

    async def _execute_rule_tools(
        self,
        node: Node,
        client_manager: ClientManager,
    ) -> AsyncGenerator[dict | None, None]:
        nodes_with_rules_met, rule_tool_inputs = await self._check_rules(
            node, client_manager
        )

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
                    if isinstance(result, StreamedReturn):
                        parser, stream_id = self.streamer.get_parser(
                            result.output_type, result.field_name
                        )
                        streamed_payloads = parser.feed(result.chunk)
                        for i, streamed_payload in enumerate(streamed_payloads):
                            yield await self.returner.send_streamed(
                                streamed_payload,
                                self.prompt_to_query_id[self.user_prompt],
                                stream_id,
                            )
                    else:
                        action_result, _ = await self._evaluate_result(
                            result, rule_decision
                        )
                        if action_result is not None:
                            yield action_result

    async def _run_decision_node(
        self,
        current_node: Node,
        available_options: list[str],
        unavailable_options: list[tuple[str, str]],
        user_prompt: str,
        client_manager: ClientManager,
    ) -> AsyncGenerator[dict | None, None]:

        self.tracker.start_tracking("decision_node")
        self.tree_data.set_current_task("elysia_decision_node")
        options = self._build_tool_options(available_options, unavailable_options)

        results = []
        with ElysiaKeyManager(self.settings):
            async for result in current_node.decide(
                tree_data=self.tree_data,
                base_lm=self.base_lm,
                complex_lm=self.complex_lm,
                options=options,
                client_manager=client_manager,
            ):
                if isinstance(result, StreamedReturn):
                    parser, stream_id = self.streamer.get_parser(
                        result.output_type, result.field_name
                    )
                    streamed_payloads = parser.feed(result.chunk)
                    for i, streamed_payload in enumerate(streamed_payloads):
                        yield await self.returner.send_streamed(
                            streamed_payload,
                            self.prompt_to_query_id[user_prompt],
                            stream_id,
                        )
                elif isinstance(result, ViewEnvironment):
                    yield await self.returner.send(
                        result, self.prompt_to_query_id[user_prompt]
                    )
                    self._set_view_environment_vars(result)
                elif isinstance(result, Decision):
                    self.current_decision = result
                else:
                    results.append(result)

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
        self.streamer.reset()

    async def _execute_tool_action(
        self,
        tool_name: str,
        decision: Decision,
        client_manager: ClientManager,
        **kwargs,
    ) -> AsyncGenerator[tuple[dict | None, bool], None]:
        self.tracker.start_tracking(decision.function_name)
        self.tree_data.set_current_task(decision.function_name)
        successful_action = True

        with ElysiaKeyManager(self.settings):
            async for result in self.tools[tool_name](
                tree_data=self.tree_data,
                inputs=decision.function_inputs,
                base_lm=self.base_lm,
                complex_lm=self.complex_lm,
                client_manager=client_manager,
                **kwargs,
            ):
                if isinstance(result, StreamedReturn):
                    parser, stream_id = self.streamer.get_parser(
                        result.output_type, result.field_name
                    )
                    streamed_payloads = parser.feed(result.chunk)
                    for i, streamed_payload in enumerate(streamed_payloads):
                        yield (
                            await self.returner.send_streamed(
                                streamed_payload,
                                self.prompt_to_query_id[self.user_prompt],
                                stream_id,
                            ),
                            False,
                        )
                else:
                    action_result, error = await self._evaluate_result(result, decision)
                    if action_result is not None:
                        yield action_result, error
                    successful_action = not error and successful_action

        self.tracker.end_tracking(
            decision.function_name,
            decision.function_name,
            self.base_lm if not self.low_memory else None,
            self.complex_lm if not self.low_memory else None,
        )

        yield None, not successful_action

    async def _evaluate_result(
        self,
        result: Result | EdgeUpdate | Error | TrainingUpdate | Text | Update,
        decision: Decision,
    ) -> tuple[dict | None, bool]:
        error = False

        if isinstance(result, Result):
            self._add_refs(result.objects, decision.function_name)
            self._update_environment(result, decision)

        if isinstance(result, TrainingUpdate):
            self.training_updates.append(result)
            return None, error

        if isinstance(result, Error):
            self._add_error(decision.function_name, result)
            error_text = (
                result.feedback
                if result.feedback != "An unknown issue occurred."
                else result.error_message
            )
            self._print_panel(error_text, "Error", "red")
            error = True

        if isinstance(result, Text):
            self._update_conversation_history("assistant", result.text)
            self._print_panel(result.text, "Assistant response", "cyan")

        return (
            await self.returner.send(result, self.prompt_to_query_id[self.user_prompt]),
            error,
        )

    async def _get_available_options(
        self, current_node: Node, client_manager: ClientManager
    ) -> tuple[list[str], list[tuple[str, str]]]:
        available_options = []
        unavailable_options = []

        for option in current_node.options:
            # if the option is a branch and it has no options itself, it is invisibly unavailable
            if self.nodes[option].branch and len(self.nodes[option].options) == 0:
                pass

            # if it is a tool, and it has the is_tool_available method, and it returns True, it is available
            elif (
                not self.nodes[option].branch
                and self.nodes[option].name in self.tools
                and "is_tool_available" in dir(self.tools[self.nodes[option].name])
                and not (
                    await self.tools[self.nodes[option].name].is_tool_available(
                        tree_data=self.tree_data,
                        base_lm=self.base_lm,
                        complex_lm=self.complex_lm,
                        client_manager=client_manager,
                    )
                )
            ):

                doc = self.tools[self.nodes[option].name].is_tool_available.__doc__
                is_tool_available_doc = doc.strip() if isinstance(doc, str) else ""
                unavailable_options.append((option, is_tool_available_doc))
            else:
                available_options.append(option)

        return available_options, unavailable_options

    def _create_client_manager(self) -> ClientManager:
        return ClientManager(settings=self.settings)

    async def _initialize_run(
        self,
        user_prompt: str,
        query_id: str | None,
        collection_names: list[str],
        client_manager: ClientManager,
    ) -> str:
        self.settings.logger.debug(f"Style: {self.tree_data.atlas.style}")
        self.settings.logger.debug(
            f"Agent description: {self.tree_data.atlas.agent_description}"
        )
        self.settings.logger.debug(f"End goal: {self.tree_data.atlas.end_goal}")

        query_id = query_id if query_id is not None else str(uuid.uuid4())
        self.returner.add_prompt(user_prompt, query_id)

        # Reset the tree (clear temporary data specific to the last user prompt)
        self.soft_reset()

        check_base_lm_settings(self.settings)
        check_complex_lm_settings(self.settings)

        self.set_start_time()
        self.query_id_to_prompt[query_id] = user_prompt
        self.prompt_to_query_id[user_prompt] = query_id
        self.tree_data.set_property("user_prompt", user_prompt)
        self._update_conversation_history("user", user_prompt)
        self.user_prompt = user_prompt

        # Check and start clients if not already started
        if client_manager.is_client:
            await client_manager.start_clients()

            if self.tree_data.use_weaviate_collections:
                if not collection_names:
                    async with client_manager.connect_to_async_client() as client:
                        collection_names = await retrieve_all_collection_names(client)
                await self.set_collection_names(collection_names, client_manager)

        self._print_panel(user_prompt, "User Prompt", "yellow")

        return query_id

    def _build_tool_options(
        self, available_options: list[str], unavailable_options: list[tuple[str, str]]
    ) -> list[ToolOption]:
        options = []
        for option in available_options:
            node = self.nodes[option]
            if node.branch:
                options.append(
                    ToolOption(
                        name=node.name,
                        available=True,
                        description=node.description,
                        inputs=[],
                    )
                )
            else:
                options.append(
                    ToolOption(
                        name=node.name,
                        available=True,
                        description=node.description,
                        inputs=[
                            ToolInput(
                                name=input_name,
                                description=input_def.get("description", ""),
                                type=input_def.get("type", Any),
                                default=input_def.get("default", None),
                                required=input_def.get("required", False),
                            )
                            for input_name, input_def in self.tools[
                                node.name
                            ].inputs.items()
                        ],
                    )
                )

        # Add unavailable options
        options.extend(
            ToolOption(
                name=self.nodes[opt_id].name,
                available=False,
                description="",
                unavailable_reason=reason,
                inputs=[],
            )
            for opt_id, reason in unavailable_options
        )
        return options

    def _format_model_usage(self, model_type: str) -> str:
        num_calls = self.tracker.get_num_calls(model_type)
        if num_calls == 0:
            return f"{model_type.replace('_', ' ').title()} Usage: [magenta]0[/magenta] calls"

        avg_input = self.tracker.get_average_input_tokens(model_type)
        avg_output = self.tracker.get_average_output_tokens(model_type)
        total_input = self.tracker.get_total_input_tokens(model_type)
        total_output = self.tracker.get_total_output_tokens(model_type)
        total_cost = self.tracker.get_total_cost(model_type)
        avg_cost = self.tracker.get_average_cost(model_type)

        return (
            f"{model_type.replace('_', ' ').title()} Usage: \n"
            f"  - Calls: [magenta]{num_calls}[/magenta]\n"
            f"  - Input Tokens: [magenta]{total_input}[/magenta] (Avg. [magenta]{int(avg_input)}[/magenta] per call)\n"
            f"  - Output Tokens: [cyan]{total_output}[/cyan] (Avg. [cyan]{int(avg_output)}[/cyan] per call)\n"
            f"  - Total Cost: [yellow]${total_cost:.4f}[/yellow] (Avg. [yellow]${avg_cost:.4f}[/yellow] per call)\n"
        )

    def log_token_usage(self) -> None:
        if self.low_memory:
            return

        self.settings.logger.debug(self._format_model_usage("base_lm"))
        self.settings.logger.debug(self._format_model_usage("complex_lm"))

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
            client_manager = self._create_client_manager()

        if _first_run:
            query_id = await self._initialize_run(
                user_prompt, query_id, collection_names, client_manager
            )
            yield await self.returner.send(
                GraphUpdate(
                    nodes=[node.to_tree_node() for node in self.nodes.values()],
                    edges=self.edges,
                ),
                query_id,
            )

        if self.root is None:
            raise ValueError("No root node found!")
        current_decision_node: Node = self.nodes[self.root]

        while True:
            available_options, unavailable_options = await self._get_available_options(
                current_decision_node, client_manager
            )
            self._reset_view_environment_vars()

            if not available_options:
                self.settings.logger.error("No options available to use!")
                raise ValueError(
                    f"No tools or branches available to be decided from (on branch [magenta]{current_decision_node.name}[/magenta])! "
                    "Check the tool definitions and the `is_tool_available` methods."
                )

            async for action_result in self._execute_rule_tools(
                current_decision_node, client_manager
            ):
                yield action_result

            async for action_result in self._run_decision_node(
                current_decision_node,
                available_options,
                unavailable_options,
                user_prompt,
                client_manager,
            ):
                yield action_result

            next_node_id = next(
                opt
                for opt in available_options
                if self.nodes[opt].name == self.current_decision.function_name
            )
            next_node = self.nodes[next_node_id]

            # If model picks to end but node isn't an end, then force a text answer for completeness
            force_text_response = (
                not next_node.end and self.current_decision.end_actions
            )

            # Completed when either text response is picked or model chooses to end, or recursion limit reached
            completed = (
                self.current_decision.function_name == "text_response"
                or self.current_decision.end_actions
                or self.tree_data.num_trees_completed > self.tree_data.recursion_limit
            )

            self.decision_history[-1].append(self.current_decision.function_name)
            self._print_decision_panel(
                current_decision_node.name, self.current_decision
            )

            yield await self.returner.send(
                EdgeUpdate(
                    from_node=current_decision_node.id,
                    to_node=next_node.id,
                    reasoning=(
                        self.current_decision.reasoning
                        if self.settings.BASE_USE_REASONING
                        else ""
                    ),
                    reset_tree=not completed and len(next_node.options) == 0,
                    tree_index=self.tree_data.num_trees_completed,
                ),
                self.prompt_to_query_id[user_prompt],
            )

            # If action is a tool (not branch) then yield out results/record outputs
            if not next_node.branch:
                successful_action = True
                async for action_result, had_error in self._execute_tool_action(
                    next_node.name, self.current_decision, client_manager, **kwargs
                ):
                    if action_result is not None:
                        yield action_result
                    if had_error:
                        successful_action = False

                if not successful_action:
                    completed = (
                        self.tree_data.num_trees_completed
                        > self.tree_data.recursion_limit
                    )
                else:
                    self.tree_data.clear_error(self.current_decision.function_name)

            self.tree_data.update_tasks_completed(
                prompt=self.user_prompt,
                task=self.current_decision.function_name,
                num_trees_completed=self.tree_data.num_trees_completed,
                reasoning=self.current_decision.reasoning,
                action=not next_node.branch,
                inputs=self.current_decision.function_inputs,
            )
            # Check termination conditions
            if completed or not next_node.options:
                break

            current_decision_node = next_node

        self.tree_data.num_trees_completed += 1

        # Handle completion
        if completed:
            # Generate forced text response if needed
            if not next_node.end or force_text_response:
                async for action_result in self._execute_forced_text_response(
                    client_manager
                ):
                    yield action_result

            self.save_history(
                query_id=self.prompt_to_query_id[user_prompt],
                time_taken_seconds=time.time() - self.start_time,
            )
            yield await self.returner.send(
                Completed(), query_id=self.prompt_to_query_id[user_prompt]
            )
            self._log_completion_stats()

            if close_clients_after_completion and client_manager.is_client:
                await client_manager.close_clients()
        else:
            # Recursive restart for incomplete goal
            self.settings.logger.debug(
                "Model did [bold red]not[/bold red] yet complete overall goal!"
            )
            self.settings.logger.debug(
                f"Restarting tree (Recursion: {self.tree_data.num_trees_completed+1}/{self.tree_data.recursion_limit})..."
            )
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

    async def _execute_forced_text_response(
        self, client_manager: ClientManager
    ) -> AsyncGenerator[dict | None, None]:
        with ElysiaKeyManager(self.settings):
            async for result in self.tools["forced_text_response"](
                tree_data=self.tree_data,
                inputs={},
                base_lm=self.base_lm,
                complex_lm=self.complex_lm,
                client_manager=client_manager,
            ):
                action_result, _ = await self._evaluate_result(
                    result, self.current_decision
                )
                if action_result is not None:
                    yield action_result

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
            self._console = Console()

            with self._console.status(
                "[bold indigo]Thinking...[/bold indigo]"
            ) as status:
                self._status = status
                try:
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
                            payload: dict = result["payload"]
                            status.update(
                                f"[bold indigo]{payload['text']}[/bold indigo]"
                            )
                finally:
                    self._status = None
                    self._console = None

            return self.retrieved_objects

        if self.settings.LOGGING_LEVEL_INT <= 20:
            yielded_results = asyncio_run(run_with_live())
        else:
            yielded_results = asyncio_run(run_process())

        text = self.tree_data.conversation_history[-1]["content"]

        return text, yielded_results

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
                "preset_id": self.preset_id,
                "conversation_title": self.conversation_title,
                "branch_initialisation": self.branch_initialisation,
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
            preset_id (str): The id of the tool preset used in the tree.
                If not provided, set to None.
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
                    vector_config=wc.Configure.Vectors.self_provided(),
                    inverted_index_config=wc.Configure.inverted_index(
                        index_timestamps=True,
                        index_null_state=True,
                    ),
                    multi_tenancy_config=wc.Configure.multi_tenancy(
                        enabled=True,
                        auto_tenant_creation=True,
                        auto_tenant_activation=True,
                    ),
                    properties=[
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
            if not await collection.tenants.exists(self.user_id):
                await collection.tenants.create(self.user_id)
            user_collection = collection.with_tenant(self.user_id)

            json_data_str = json.dumps(self.export_to_json())

            uuid = generate_uuid5(self.conversation_id)

            if await user_collection.data.exists(uuid):
                await user_collection.data.update(
                    uuid=uuid,
                    properties={
                        "conversation_id": self.conversation_id,
                        "tree": json_data_str,
                        "title": self.conversation_title,
                    },
                )
                self.settings.logger.info(
                    f"Successfully updated existing tree in collection '{collection_name}' with id '{self.conversation_id}'"
                )
            else:
                await user_collection.data.insert(
                    uuid=uuid,
                    properties={
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
            preset_id=json_data["preset_id"],
            branch_initialisation=json_data["branch_initialisation"],
            style=json_data["tree_data"]["atlas"]["style"],
            agent_description=json_data["tree_data"]["atlas"]["agent_description"],
            end_goal=json_data["tree_data"]["atlas"]["end_goal"],
            low_memory=json_data["low_memory"],
            use_weaviate_collections=json_data["tree_data"]["use_weaviate_collections"],
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
        user_id: str,
        conversation_id: str,
        client_manager: ClientManager | None = None,
    ) -> "Tree":
        """
        Import a tree from a Weaviate collection.

        Args:
            collection_name (str): The name of the collection to import from.
            user_id (str): The user ID to import the tree from.
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
            if not await collection.tenants.exists(user_id):
                raise ValueError(
                    f"User '{user_id}' does not have any saved trees in collection '{collection_name}'."
                )
            user_collection = collection.with_tenant(user_id)
            uuid = generate_uuid5(conversation_id)
            if not await user_collection.data.exists(uuid):
                raise ValueError(
                    f"No tree found for conversation id '{conversation_id}' in collection '{collection_name}'."
                )

            response = await user_collection.query.fetch_object_by_id(uuid)

        if close_after_use:
            await client_manager.close_clients()

        json_data = json.loads(response.properties["tree"])  # type: ignore

        return cls.import_from_json(json_data)

    async def feedback(
        self,
        feedback_level: Literal[0, 1, 2],
        collection_name: str = "ELYSIA_FEEDBACK__",
        client_manager: ClientManager | None = None,
        prompt: str | None = None,
        query_id: str | None = None,
    ):
        """
        Leave feedback on the an interaction with Elysia for this decision tree.
        The feedback saves all inputs and outputs used for the decision agent and any supported tools to a Weaviate collection.

        Args:
            feedback_level (Literal[0, 1, 2]): the rating for the interaction. 0 = negative, 1=positive, 2=very positive
            collection_name (str): the Weaviate collection name used to save feedback. Defaults to ELYSIA_FEEDBACK__.
            prompt (str | None): The prompt to use to save the feedback for. If not provided, defaults to the last interaction.
            query_id (str | None): The query ID to use to save the feedback for, as an alternative to the prompt.
                If not provided, defaults to the last interaction.
                Cannot use both `query_id` and `prompt`.
        """
        if client_manager is None:
            client_manager = ClientManager(settings=self.settings)

        if prompt and query_id and self.prompt_to_query_id[prompt] != query_id:
            raise ValueError(
                "Provided both `prompt` and `query_id` to `tree.feedback`, but they referenced different interactions"
            )

        if prompt:
            query_id = self.prompt_to_query_id[prompt]

        if not query_id and not prompt:
            query_id = list(self.query_id_to_prompt.keys())[0]

        async with client_manager.connect_to_async_client() as client:

            if not await client.collections.exists(collection_name):
                await create_feedback_collection(client)

            base_feedback_collection = client.collections.get(collection_name)

            if not await base_feedback_collection.tenants.exists(self.user_id):
                await base_feedback_collection.tenants.create(self.user_id)

            feedback_collection = base_feedback_collection.with_tenant(self.user_id)

            history = self.history[query_id]

            # ensure "action" is a bool in tasks_completed
            for task_prompt in history["tree_data"].tasks_completed:
                for task in task_prompt["task"]:
                    task["action"] = bool(task["action"])

            date_now = datetime.now()

            properties = {
                "user_id": self.user_id,
                "conversation_id": self.conversation_id,
                "query_id": query_id,
                "feedback": int(feedback_level),
                "modules_used": list(
                    set([h["module_name"] for h in history["training_updates"]])
                ),
                "user_prompt": history["tree_data"].user_prompt,
                "conversation_history": history["tree_data"].conversation_history,
                "tasks_completed": history["tree_data"].tasks_completed,
                "route": history["decision_history"],
                "action_information": json.dumps(history["action_information"]),
                "time_taken_seconds": history["time_taken_seconds"],
                "decision_time": self.tracker.get_average_time("decision_node"),
                "base_lm_used": self.base_lm.model,
                "complex_lm_used": self.complex_lm.model,
                "feedback_datetime": format_datetime(date_now),
                "feedback_date": format_datetime(
                    date_now.replace(hour=0, minute=0, second=0, microsecond=0)
                ),
                "training_updates": json.dumps(history["training_updates"]),
                "initialisation": history["initialisation"],
            }

            # uuid is generated based on the user_id, conversation_id, query_id ONLY
            # so if the user re-selects the same feedback, it will be updated instead of added
            session_uuid = generate_uuid5(
                {
                    "user_id": self.user_id,
                    "conversation_id": self.conversation_id,
                    "query_id": query_id,
                }
            )

            if await feedback_collection.data.exists(session_uuid):
                await feedback_collection.data.update(
                    properties=properties, uuid=session_uuid
                )
            else:
                await feedback_collection.data.insert(
                    properties=properties, uuid=session_uuid
                )
            return session_uuid

    def __call__(self, *args, **kwargs) -> tuple[str, list[dict]]:
        return self.run(*args, **kwargs)
