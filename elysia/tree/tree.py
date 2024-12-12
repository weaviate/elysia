import json
import os
import ast
import inspect
import time
import dspy
from typing import Callable, List, Any, Dict
from rich import print
from rich.panel import Panel

import asyncio
import nest_asyncio
import datetime

# Actions
from elysia.text.text import Summarizer
from elysia.querying.query import AgenticQuery
from elysia.aggregating.aggregate import AgenticAggregate

#  Decision Prompt executors
from elysia.tree.prompt_executors import DecisionExecutor, InputExecutor

# Util
from elysia.util.parsing import remove_whitespace, update_current_message, format_datetime
from elysia.util.logging import backend_print

# Objects
from elysia.tree.objects import Returns, Objects, TreeData, ActionData, DecisionData
from elysia.api.objects import Status, TreeUpdate, Error, Warning, Completed
from elysia.text.objects import Text, Response, Summary
from elysia.querying.objects import Retrieval
from elysia.aggregating.objects import Aggregation

# globals
from elysia.globals.weaviate_client import client
from elysia.globals.return_types import all_return_types

# training
from elysia.training.prompt_executors import TrainingDecisionExecutor

# dspy requires a 'base' LM but this should not be used
import dspy
# from dspy import LM
from elysia.dspy.cached_lm import CachingLM as LM

from elysia.dspy.cached_claude import CachingClaude

fallback_lm = LM("claude-3-5-haiku-20241022")
dspy.settings.configure(lm=fallback_lm)

class RecursionLimitException(Exception):
    pass

class DecisionNode:

    def __init__(self, 
            id: str, 
            instruction: str, 
            options: list[dict[str, str]], 
            root: bool = False, 
            dspy_model: str = None,
            base_lm: LM = None,
            complex_lm: LM = None
        ):
        
        self.id = id
        self.instruction = instruction
        self.options = options
        self.root = root
        self.base_lm = base_lm
        self.complex_lm = complex_lm

        # Define the decision executor
        self.decision_executor = DecisionExecutor(list(options.keys())).activate_assertions(max_backtracks=4) 

        # Load the model if it exists
        if dspy_model is not None:
            self.decision_executor.load(os.path.join("elysia", "training", "dspy_models", "decision", dspy_model + ".json"))

    def decide_with_training(
            self, 
            tree_data: TreeData, 
            decision_data: DecisionData, 
            action_data: ActionData,
            task: str = ""
        ):
        # If training, the task is given and we are only interested in the metadata, otherwise business as usual
        self.decision_executor = TrainingDecisionExecutor(list(self.options.keys()))
        return self.decide(tree_data, decision_data, action_data, training=True, task=task)

    def _options_to_json(self):
        out = {}
        for node in self.options:
            out[node] = self.options[node]["description"]
        return out
    
    def _options_to_future(self):
        out = {}
        for node in self.options:
            out[node] = self.options[node]["future"]
        return out

    def decide(
            self, 
            tree_data: TreeData, 
            decision_data: DecisionData, 
            action_data: ActionData, 
            training: bool = False,
            model: str = "base",
            **kwargs
        ):
        
        with dspy.context(lm=self.base_lm if model == "base" else self.complex_lm):
            output, self.completed = self.decision_executor(
                user_prompt=tree_data.user_prompt,
                conversation_history=tree_data.conversation_history,
                previous_reasoning=tree_data.previous_reasoning,
                current_message=tree_data.current_message,
                data_queried=tree_data.data_queried_string(),
                instruction=decision_data.instruction,
                collection_information=action_data.collection_information,
                tree_count=decision_data.tree_count_string(),
                available_tasks=decision_data.available_tasks,
                available_information=decision_data.available_information.to_json(),
                future_information=decision_data.future_information
            )

        # if training, the task is given input
        if training:
            self.decision = kwargs.get("task", "")
        else:
            self.decision = output.task

        # Return action function if it exists
        if self.options[self.decision]['action'] is not None:
            action_fn = self.options[self.decision]['action']
        else:
            action_fn = None

        # Save reasoning
        self.reasoning = output.reasoning

        return output, action_fn, self.completed

    def __call__(
            self, 
            tree_data: TreeData,
            decision_data: DecisionData, 
            action_data: ActionData,
            **kwargs
        ):
        return self.decide(tree_data, decision_data, action_data, **kwargs)
    
class TreeReturner:
    """
    Class to parse the output of the tree to the frontend.
    """
    def __init__(self, conversation_id: str, tree_index: int = 0, mappings: dict = None):
        self.conversation_id = conversation_id
        self.tree_index = tree_index
        self.mappings = mappings

    def set_tree_index(self, tree_index: int):
        self.tree_index = tree_index

    def _parse_error(self, error: Error, query_id: str):
        return error.to_frontend(self.conversation_id, query_id)
    
    def _parse_warning(self, warning: Warning, query_id: str):
        return warning.to_frontend(self.conversation_id, query_id)

    def _parse_status(self, status: Status, query_id: str):
        return status.to_frontend(self.conversation_id, query_id)
    
    def _parse_tree_update(self, tree_update: TreeUpdate, query_id: str, reset: bool = False):
        return tree_update.to_frontend(self.tree_index, self.conversation_id, query_id, reset)
    
    def _parse_completed(self, query_id: str):
        return Completed().to_frontend(self.conversation_id, query_id)
    
    def _parse_result(self, result: Objects, query_id: str):
        if result.type != "aggregation":
            mapping = self.mappings[result.metadata["collection_name"]][result.type]
        else:
            mapping = None

        return result.to_frontend(
            self.conversation_id, 
            query_id, 
            mapping,
        )
    
    def _parse_text(self, text: Text, query_id: str):
        return text.to_frontend(self.conversation_id, query_id)
    
    def __call__(self, result: Any, **kwargs):

        completed = kwargs.get("completed", False)
        query_id = kwargs.get("query_id", "")

        if isinstance(result, Status):
            return self._parse_status(result, query_id = query_id)

        if isinstance(result, TreeUpdate):
            return self._parse_tree_update(
                tree_update=result, 
                query_id = query_id,
                reset = result.last and not completed,
            )

        if isinstance(result, Objects):
            if len(result.objects) > 0:
                return self._parse_result(
                    result, 
                    query_id = query_id
                )

        if isinstance(result, Text):
            return self._parse_text(result, query_id = query_id)

        if isinstance(result, Error):
            return self._parse_error(result, query_id = query_id)

        if isinstance(result, Warning):
            return self._parse_warning(result, query_id = query_id)
        
class BranchVisitor(ast.NodeVisitor):
    def __init__(self):
        self.branches = []
        
    def _evaluate_constant(self, node):
        """Convert AST Constant node to its actual value."""
        return node.value
    
    def _evaluate_dict(self, node):
        """Convert AST Dict node to a regular Python dictionary."""
        if isinstance(node, ast.Dict):
            return {
                self._evaluate_constant(key): self._evaluate_constant(value)
                for key, value in zip(node.keys, node.values)
            }
        return None
        
    def visit_Call(self, node):
        # Check if the call is creating a Branch object
        if isinstance(node.func, ast.Name) and node.func.id == 'Branch':
            # Extract the updates dictionary from the Branch constructor
            if node.args:
                dict_node = node.args[0]
                evaluated_dict = self._evaluate_dict(dict_node)
                if evaluated_dict:
                    self.branches.append(evaluated_dict)
        self.generic_visit(node)

class Tree:

    def __init__(self, 
            conversation_id: str = "1", 
            collection_names: list[str] = [],
            verbosity: int = 1, 
            break_down_instructions: bool = False,
            dspy_model: str = None
        ):
        
        # Define base variables of the tree
        self.conversation_id = conversation_id
        self.verbosity = verbosity
        self.break_down_instructions = break_down_instructions
        self.dspy_model = dspy_model
        self.collection_names = collection_names

        # set up LLMs in dspy
        # self.base_lm = LM(model="gpt-4o-mini", max_tokens=6000)
        # self.complex_lm = LM(model="gpt-4o", max_tokens=6000)
        # self.complex_lm = LM(model="claude-3-5-haiku-20241022", max_tokens=6000    )
        self.complex_lm = LM(model="claude-3-5-sonnet-20241022", max_tokens=6000)
        # self.complex_lm = LM(model="bedrock/us.anthropic.claude-3-5-haiku-20241022-v1:0", max_tokens=6000)
        # self.complex_lm = LM(model="bedrock/us.anthropic.claude-3-5-sonnet-20241022-v2:0", max_tokens=6000)
        
        # self.base_lm = LM(model="gemini-1.5-flash", max_tokens=6000)
        # self.complex_lm = LM(model="gemini-1.5-flash", max_tokens=6000)
        self.base_lm = LM(model="openrouter/google/gemini-flash-1.5", max_tokens=6000)
        # self.base_lm = LM(model="openrouter/google/gemini-2.0-flash-exp", max_tokens=6000)
        # self.complex_lm = LM(model="openrouter/meta-llama/llama-3.3-70b-instruct", max_tokens=6000)

        # keep track of the number of trees completed
        self.num_trees_completed = 0
        self.max_recursions = 5

        # Initialise some tree variables
        self.decision_nodes = {}
        self.decision_history = []
        self.tree_index = -1

        # Get collection information and initialise error message for collection based errors
        self.initialise_error_message = ""
        self.collection_information, self.removed_collections = self._get_collection_information()
        if len(self.removed_collections) > 0:
            self.initialise_error_message = f"The following collections have not been processed yet and have been removed: {self.removed_collections}"

        # get return types for each collection
        self.collection_return_types = self._get_collection_mappings()

        # initialise the returner (for frontend)
        self.returner = TreeReturner(
            conversation_id=self.conversation_id, 
            mappings={key: self.collection_information[key]["mappings"] for key in self.collection_information}
        )

        # mapping between query ids and prompts
        self.query_id_to_prompt = {}
        self.prompt_to_query_id = {}

        # Parse the instructions if required (use an LLM to break the prompt into a manageable number of instructions)
        if self.break_down_instructions:
            self.input_executor = InputExecutor()

        # Define the inputs to prompts
        self.tree_data = TreeData()
        self.action_data = ActionData(
            collection_information=self.collection_information,
            collection_return_types=self.collection_return_types
        )
        self.decision_data = DecisionData(
            recursion_limit=self.max_recursions,
            available_information=Returns(
                retrieved = [], 
                aggregation = [],
                text = Text(objects=[], metadata={})
            )
        )

        # Define the action agents
        self.querier = AgenticQuery(
            collection_names=self.collection_names, 
            collection_return_types=self.collection_return_types,
            base_lm=self.base_lm,
            complex_lm=self.complex_lm,
            verbosity=verbosity
        )

        self.aggregator = AgenticAggregate(
            base_lm=self.base_lm,
            complex_lm=self.complex_lm,
            collection_names=self.collection_names
        )

        # Summariser
        self.summariser = Summarizer(
            base_lm=self.base_lm,
            complex_lm=self.complex_lm
        )

        # -- Initialise the tree
        # default initialisations -
        self.add_decision_node(
            id = "base",
            instruction = """
            Choose a base-level task based on the user's prompt and available information. 
            You can search, which includes aggregating or querying information - this should be used if the user needs (more) information.
            You can end the conversation by choosing text response, or summarise some retrieved information.
            Base your decision on what information is available and what the user is asking for - you can search multiple times if needed,
            but you should not search if you have already found all the information you need.
            """,
            options = {
                "search": {
                    "description": "Search the knowledge base. This should be used when the user is lacking information for this particular prompt. This retrieves information only and provides no output to the user except the information.",
                    "future": "Choose to query (semantic or keyword search on a knowledge base), or aggregate information (calculate properties/summary statistics/averages and operations on the knowledge bases). Collections that can be queried are " + ", ".join(self.collection_names) + ". Return types that are available are: " + ", ".join(all_return_types.values()),
                    "action": None,
                    "next": "search",
                    "status": "Searching..."
                },
                "summarize": {
                    "description": "Summarize some already required information. This should be used when the user wants a high-level overview of some retrieved information or a generic response based on the information.  This is usually the last decision to make, so you should set all_actions_completed to True if you choose this.",
                    "future": "",
                    "action": self.summariser,
                    "next": None,
                    "status": "Summarising..."
                },
                "text_response": {
                    "description": "End the conversation. This should be used when the user has finished their query, or you have nothing more to do.",
                    "future": "",
                    "action": None,
                    "next": None,
                    "status": "Writing response..."
                }
            },
            root = True
        )

        self.add_decision_node(
            id = "search",
            instruction = """
            Choose between querying the knowledge base via semantic/keyword search, or aggregating information by performing operations, on the knowledge base.
            Querying is when the user is looking for specific information related to the content of the dataset, requiring a specific search query. This is for retrieving specific information via a _query_, similar to a search engine.
            Aggregating is when the user is looking for a specific operations on the dataset, such as summary statistics of the quantity of some items. Aggregation can also include grouping information by some property and returning statistics about the groups. This is similar to SQL style queries (grouping/filtering/performing operations on the data).
            """,
            options = {
                "query": {
                    "description": "Query the knowledge base based on searching with a specific query.",
                    "future": "You will be given information about the collection, such as the fields and types of the data. Then, query with semantic search, keyword search, or a combination of both.",
                    "action": self.querier,
                    "next": None,
                    "status": "Querying..."
                },
                "aggregate": {
                    "description": "Perform functions such as counting, averaging, summing, etc. on the data, grouping by some property and returning metrics/statistics about different categories. Or providing a high level overview of the dataset.",
                    "future": "You will be given information about the collection, such as the fields and types of the data. Then, aggregate over different categories of the collection, with operations: top_occurences, count, sum, average, min, max, median, mode, group_by.",
                    "action": self.aggregator,
                    "next": None,     # next=None represents the end of the tree
                    "status": "Aggregating..."
                }
            },
            root = False
        )

        # -- Get the root node and construct the tree
        self._get_root()
        self.tree = {}
        self._construct_tree(self.root, self.tree)        

        # Print the tree if required
        if verbosity > 1:
            backend_print("Initialised tree with the following decision nodes:")
            for decision_node in self.decision_nodes.values():
                backend_print(f"  - [magenta]{decision_node.id}[/magenta]: {list(decision_node.options.keys())}")

    def _get_collection_information(self):
        collection_information = {}
        removed_collections = []
        for collection_name in self.collection_names:
            metadata_name = f"ELYSIA_METADATA_{collection_name}__"
            if client.collections.exists(metadata_name):
                metadata = client.collections.get(metadata_name).query.fetch_objects(limit=1)
                properties = {}

                def format_datetime_in_dict(d):
                    for key, value in d.items():
                        if isinstance(value, datetime.datetime):
                            d[key] = format_datetime(value)
                        elif isinstance(value, dict):
                            format_datetime_in_dict(value)
                        elif isinstance(value, list):
                            for i, item in enumerate(value):
                                if isinstance(item, dict):
                                    format_datetime_in_dict(item)
                                elif isinstance(item, datetime.datetime):
                                    d[key][i] = format_datetime(item)

                format_datetime_in_dict(metadata.objects[0].properties)
                properties.update(metadata.objects[0].properties)
                collection_information[collection_name] = properties
            
            else:
                # remove the collection from the list if metadata does not exist
                # TODO: could process at this point but takes lots of time
                removed_collections.append(collection_name)
                self.collection_names.remove(collection_name)

        return collection_information, removed_collections

    def _get_collection_mappings(self):
        collection_mappings = {}
        for collection_name in self.collection_names:
            collection_mappings[collection_name] = list(self.collection_information[collection_name]["mappings"].keys())
        return collection_mappings

    def _get_root(self):
        for decision_node in self.decision_nodes.values():            
            if decision_node.root:

                if "root" in dir(self):
                    raise ValueError("Multiple root decision nodes found")

                self.root = decision_node.id

        if "root" not in dir(self):
            raise ValueError("No root decision node found")

    def _get_function_branches(self, func: Any) -> List[Dict]:
        """Analyzes a function for Branch objects without executing it."""
        source = inspect.getsource(func)
        tree = ast.parse(source.strip())
        visitor = BranchVisitor()
        visitor.visit(tree)
        return visitor.branches
    
    def _update_conversation_history(self, role: str, message: str, append_to_previous: bool = False):
        if message != "":
            if append_to_previous and self.tree_data.conversation_history[-1]["role"] == role:
                self.tree_data.conversation_history[-1]["content"] += message
            else:
                self.tree_data.update_list("conversation_history", {
                    "role": role,
                    "content": message
                })

    def _construct_tree(self, node_id: str, tree: dict):
        decision_node = self.decision_nodes[node_id]
        
        # Define desired key order
        key_order = ["name", "id", "description", "instruction", "reasoning", "options"]
        
        # Set the base node information
        tree["name"] = node_id.capitalize().replace("_", " ")
        tree["id"] = node_id
        if node_id == self.root:
            tree["description"] = ""
        tree["instruction"] = remove_whitespace(decision_node.instruction.replace("\n", ""))
        tree["reasoning"] = ""
        tree["options"] = {}

        # Order the top-level dictionary
        tree = {key: tree[key] for key in key_order if key in tree}

        # Initialize all options first with ordered dictionaries
        for option in decision_node.options:
            tree["options"][option] = {
                "description": remove_whitespace(decision_node.options[option]["description"].replace("\n", ""))
            }
        
        # Then handle the recursive cases
        for option in decision_node.options:
            if decision_node.options[option]["action"] is not None:
                func = decision_node.options[option]["action"].__call__
                branches = self._get_function_branches(func)
                if branches:
                    tree["options"][option]["name"] = option.capitalize().replace("_", " ")
                    tree["options"][option]["id"] = option
                    tree["options"][option]["instruction"] = ""
                    sub_tree = tree["options"][option]
                    for branch in branches:
                        branch_name = branch["name"].lower().replace(" ", "_")
                        if "options" not in sub_tree:
                            sub_tree["options"] = {}
                        sub_tree["options"][branch_name] = branch
                        sub_tree["options"][branch_name]["name"] = branch["name"]
                        sub_tree["options"][branch_name]["id"] = branch_name
                        sub_tree["options"][branch_name]["description"] = branch["description"]
                        sub_tree["options"][branch_name]["instruction"] = ""
                        sub_tree["options"][branch_name]["reasoning"] = ""
                        sub_tree = sub_tree["options"][branch_name]
                    sub_tree["options"] = {}
                
                else:
                    tree["options"][option]["name"] = option.capitalize().replace("_", " ")
                    tree["options"][option]["id"] = option
                    tree["options"][option]["instruction"] = ""
                    tree["options"][option]["reasoning"] = ""
                    tree["options"][option]["options"] = {}

            elif decision_node.options[option]["next"] is not None:
                tree["options"][option] = self._construct_tree(
                    decision_node.options[option]["next"], 
                    tree["options"][option]
                )
            else:
                tree["options"][option]["name"] = option.capitalize().replace("_", " ")
                tree["options"][option]["id"] = option
                tree["options"][option]["instruction"] = ""
                tree["options"][option]["reasoning"] = ""
                tree["options"][option]["options"] = {}
            
            # Order each option's dictionary
            tree["options"][option] = {key: tree["options"][option][key] 
                                     for key in key_order 
                                     if key in tree["options"][option]}
        
        return tree

    def _update_returns(self, action_result: Retrieval, user_prompt: str):

        if isinstance(action_result, Retrieval): 
            self.decision_data.available_information.add_retrieval(objects=action_result)

            dict_to_update = {
                "type": "retrieval",
                "count": len(action_result.objects),
                "prompt": user_prompt
            }
            if "return_type" in action_result.metadata:
                dict_to_update["return_type"] = action_result.metadata["return_type"]
            if "output_type" in action_result.metadata:
                dict_to_update["output_type"] = action_result.metadata["output_type"]
            if "impossible_prompts" in action_result.metadata and action_result.metadata["impossible_prompts"][-1] == user_prompt:
                dict_to_update["impossible_prompt"] = True

            self.tree_data.update_data_queried(action_result.metadata["collection_name"], user_prompt, dict_to_update)

        if isinstance(action_result, Aggregation):
            self.decision_data.available_information.add_aggregation(objects=action_result)
            dict_to_update = {
                "type": "aggregation",
                "count": len(action_result.objects),
                "prompt": user_prompt
            }
            if "impossible_prompts" in action_result.metadata and action_result.metadata["impossible_prompts"][-1] == user_prompt:
                dict_to_update["impossible_prompt"] = True

            self.tree_data.update_data_queried(action_result.metadata["collection_name"], user_prompt, dict_to_update)
    
    async def _evaluate_action(self, action_fn: Callable, user_prompt: str, completed: bool, **kwargs):
        """
        Run the action function/agent, and check for each result
        """

        async for result in action_fn(
            tree_data=self.tree_data,
            action_data=self.action_data,
            decision_data=self.decision_data
        ):
            yield self.returner(result, query_id = self.prompt_to_query_id[user_prompt], completed = completed)
            
            if isinstance(result, Objects):
                self._update_returns(result, user_prompt)

            if isinstance(result, Response): 
                self.tree_data.current_message, message_update = update_current_message(self.tree_data.current_message, result.objects[0]["text"])
                self._update_conversation_history("assistant", message_update.strip(), append_to_previous=True)
                    
            if isinstance(result, Summary):
                self._update_conversation_history("assistant", result.objects[0]["text"], append_to_previous=False)

                if self.verbosity > 1:
                    print(Panel.fit(result.objects[0]["text"], title="Summary", padding=(1,1), border_style="green"))

    def _remove_collection_from_data(self, collection_name: str):
        self.tree_data.delete_from_dict("data_queried", collection_name)
        
        for object in self.decision_data.available_information.retrieved:
            if object.metadata["collection_name"] == collection_name:
                self.decision_data.available_information.retrieved.remove(object)
        
    def set_collection_names(self, collection_names: list[str], remove_data: bool = False):
        collection_names_to_remove = [name for name in self.collection_names if name not in collection_names]

        if self.verbosity > 1:
            backend_print(f"Setting collection names to: {collection_names}")
            backend_print(f"Collection names to remove: {collection_names_to_remove}")

        self.collection_names = collection_names
        self.querier.set_collection_names(collection_names)
        self.aggregator.set_collection_names(collection_names)
        self.action_data.set_collection_names(self.collection_information, collection_names)
        
        if remove_data:
            for collection_name in collection_names_to_remove:
                self._remove_collection_from_data(collection_name)
        
        if self.verbosity >= 1:
            backend_print(f"Update collection names: {self.collection_names}")

    def _decide_from_route(self, training_route: list[str], node_id: str):
        node = self.decision_nodes[node_id]
        possible_nodes = node.options.keys()

        next_route = training_route[0]
        if next_route not in possible_nodes:
            raise Exception(f"Next node in training route ({next_route}) not in possible nodes ({possible_nodes})")
        
        training_route = training_route[1:]
        completed = len(training_route) == 0
        
        return next_route, node.options[next_route]["action"], completed, training_route
    
    def set_conversation_id(self, conversation_id: str):
        self.conversation_id = conversation_id
        self.returner.conversation_id = conversation_id

    def hard_reset(self):
        self = Tree(verbosity=self.verbosity)

    def soft_reset(self):
        # conversation history is not reset
        # available information is not reset (returns)
        self.decision_history = []
        self.num_trees_completed = 0
        self.tree_data.soft_reset()
        self.tree_index += 1
        self.returner.set_tree_index(self.tree_index)

    def add_decision_node(self, id: str, instruction: str, options: dict[str, dict[str, str]], root: bool = False):
        decision_node = DecisionNode(
            id, 
            instruction, 
            options, 
            root, 
            dspy_model = self.dspy_model,
            base_lm=self.base_lm,
            complex_lm=self.complex_lm
        )
        self.decision_nodes[id] = decision_node
        return decision_node

    async def process(
            self, 
            user_prompt: str, 
            query_id: str = "1", 
            recursion_counter: int = 0, 
            first_run: bool = True, 
            training_route: str = "",
            training_mimick_model: bool = False,
            **kwargs
        ):

        self.tree_data.update_dict("previous_reasoning", f"tree_{self.num_trees_completed+1}", {})

        if first_run:

            self.num_trees_completed = 0

            if training_route == "" or training_route is None:
                training_route = None
            else:
                training_route = training_route.split("/")

            self.tree_data.set_property("user_prompt", user_prompt)

            self.query_id_to_prompt[query_id] = user_prompt
            self.prompt_to_query_id[user_prompt] = query_id

            if self.break_down_instructions:
                user_prompt = self.input_executor(user_prompt)

            self._update_conversation_history("user", user_prompt)

            if self.verbosity > 1:
                print(Panel.fit(user_prompt, title="User prompt", padding=(1,1), border_style="yellow"))

        current_decision_node = self.decision_nodes[self.root]
        training_completed = False # flag to check if the training route has been completed, if True, halts execution
        
        while True:

            # If training, decide from the training route
            if training_route is not None:

                task, action_fn, completed, training_route = self._decide_from_route(training_route, current_decision_node.id)

                # But if we need to output info from the decisions, we need to run the decision anyway
                if training_mimick_model:
                    training_completed = completed

            # Under normal circumstances (training route is None) or mimicking model output, decide from the decision node
            if training_mimick_model or training_route is None:
                
                # update decision data with current node options
                self.decision_data.set_property("available_tasks", current_decision_node._options_to_json())
                self.decision_data.set_property("future_information", current_decision_node._options_to_future())
                self.decision_data.set_property("instruction", current_decision_node.instruction)

                # run the decision agent
                if training_mimick_model:
                    t = time.time()
                    decision, action_fn, model_completed = current_decision_node.decide_with_training(
                        tree_data=self.tree_data,
                        decision_data=self.decision_data,
                        action_data=self.action_data,
                        task=task
                    )
                    print(f"Time taken for decision: {time.time() - t:.2f} seconds")
                else:
                    t = time.time()
                    decision, action_fn, model_completed = current_decision_node(
                        tree_data=self.tree_data,
                        decision_data=self.decision_data,
                        action_data=self.action_data,
                        model = "base"
                    )
                    print(f"Time taken for decision (base): {time.time() - t:.2f} seconds")

                    if decision.confidence_score < 0.5:
                        yield self.returner._parse_status(
                            Status(f"Deferring to larger model..."), 
                            query_id = self.prompt_to_query_id[user_prompt]
                        )

                        if self.verbosity > 1:
                            backend_print(f"Confidence: {decision.confidence_score:.2f}. Deferring to larger model...")

                        t = time.time()
                        decision, action_fn, model_completed = current_decision_node(
                            tree_data=self.tree_data,
                            decision_data=self.decision_data,
                            action_data=self.action_data,
                            model = "complex"
                        )
                        print(f"Time taken for decision (complex): {time.time() - t:.2f} seconds")

                # extract task
                if training_route is None:
                    task = decision.task

                # additional end criteria, task picked is "text_response" - the model knows this
                if task == "text_response":
                    model_completed = True

                # additional end criteria, recursion limit reached
                if recursion_counter > self.decision_data.recursion_limit:
                    backend_print(f"Warning: [bold red]Recursion limit reached! ({recursion_counter})[/bold red]")
                    yield self.returner._parse_warning(
                        Warning("Recursion limit reached! Forcing text response."), 
                        query_id = self.prompt_to_query_id[user_prompt]
                    )
                    model_completed = True

                # set current variables (if not in training mode)
                if training_route is None:
                    task = task
                    completed = model_completed
                
                # update the current message
                self.tree_data.current_message, message_update = update_current_message(
                    self.tree_data.current_message, decision.reasoning_update_message
                )

                if message_update != "" and not completed:
                    yield self.returner._parse_text(Response([{"text": message_update}], {}), query_id = self.prompt_to_query_id[user_prompt])

                    if self.verbosity > 1:
                        print(Panel.fit(
                            message_update, 
                            title="Reasoning update", 
                            padding=(1, 1), 
                            border_style="cyan"
                        ))

                # update the tree update
                yield self.returner._parse_tree_update(
                    tree_update=TreeUpdate(
                        from_node=current_decision_node.id,
                        to_node=task,
                        reasoning=decision.reasoning,
                        last=False
                    ),
                    query_id = self.prompt_to_query_id[user_prompt],
                    reset = False
                )

                yield self.returner._parse_status(
                    Status(current_decision_node.options[task]["status"]), 
                    query_id = self.prompt_to_query_id[user_prompt]
                )
                
                # update the previous reasoning
                self.tree_data.update_dict("previous_reasoning", f"tree_{recursion_counter+1}", {current_decision_node.id: decision.reasoning})

            # update the decision history
            self.decision_history.append(task)
                        
            # print the current node information
            if self.verbosity > 1:
                backend_print(f"Node: [magenta]{current_decision_node.id}[/magenta]")
                backend_print(f"Instruction: [italic]{current_decision_node.instruction.strip()}[/italic]")
                backend_print(f"Decision: [green]{task}[/green]\n")

            # evaluate the action
            if action_fn is not None and not training_completed:
                async for result in self._evaluate_action(action_fn, user_prompt, completed, **kwargs):
                    yield result

            # check if the current node is the end of the tree
            if current_decision_node.options[task]["next"] is None or training_completed or completed:
                break
            else:
                current_decision_node = self.decision_nodes[current_decision_node.options[task]["next"]]
    
        # end of all trees
        if completed:

            self.decision_data.num_trees_completed += 1

            if (
                training_route is None or training_mimick_model
            ) and (
                decision.full_chat_response != "" and task != "summarize"
            ):
                self._update_conversation_history("assistant", decision.full_chat_response, append_to_previous=True)
                yield self.returner._parse_text(Response([{"text": decision.full_chat_response}], {}), query_id = self.prompt_to_query_id[user_prompt])

                if self.verbosity > 1:
                    print(Panel.fit(
                        decision.full_chat_response, 
                        title="Full chat response", 
                        padding=(1, 1), 
                        border_style="cyan"
                    ))

            yield self.returner._parse_completed(query_id = self.prompt_to_query_id[user_prompt])

            if self.verbosity >= 1:
                backend_print(f"[bold green]Model identified overall goal as completed![/bold green]")

        # end of the tree for this iteration
        else:
            if self.verbosity > 1:
                backend_print(f"Model did [bold red]not[/bold red] yet complete overall goal! Restarting tree (Recursion: {recursion_counter+1}/{self.decision_data.recursion_limit})...")

            # recursive call to restart the tree since the goal was not completed
            self.decision_data.num_trees_completed = recursion_counter
            async for result in self.process(user_prompt, query_id, recursion_counter + 1, first_run=False, training_route=training_route, training_mimick_model=training_mimick_model, **kwargs):
                yield result

    def process_sync(self, user_prompt: str, query_id: str = "1", recursion_counter: int = 0, first_run: bool = True, **kwargs):
        """Synchronous version of process() for testing purposes"""
        
        nest_asyncio.apply()

        async def run_process():
            results = []
            async for result in self.process(user_prompt, query_id, recursion_counter, first_run, **kwargs):
                results.append(result)
            return results
            
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(run_process())


