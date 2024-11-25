import json
import os
import ast
import inspect
from typing import Callable, List, Any, Dict
from rich import print

import asyncio
import nest_asyncio

# from elysia.text.prompt_executors import SummarizingExecutor, TextResponseExecutor
from elysia.text.text import Summarizer, TextResponse
from elysia.querying.agentic_query import AgenticQuery
from elysia.aggregating.aggregate import AgenticAggregate
from elysia.tree.prompt_executors import DecisionExecutor, InputExecutor
from elysia.util.parsing import remove_whitespace, update_current_message
from elysia.util.logging import backend_print
from elysia.util.api import (
    parse_result, 
    parse_finished, 
    parse_error, 
    parse_warning,
    parse_text,
    parse_tree_update
)
from elysia.tree import base_lm, complex_lm
from elysia.tree.objects import Returns, Objects, Status, Branch, TreeUpdate, Error, Warning
from elysia.text.objects import Text, Response, Summary, Code
from elysia.querying.objects import Retrieval
from elysia.aggregating.objects import Aggregation
from elysia.globals.weaviate_client import client
from elysia.training.prompt_executors import TrainingDecisionExecutor

class RecursionLimitException(Exception):
    pass

class DecisionNode:

    def __init__(self, 
            id: str, 
            instruction: str, 
            options: list[dict[str, str]], 
            root: bool = False, 
            dspy_model: str = None,
            training: bool = False
        ):
        
        self.id = id
        self.instruction = instruction
        self.options = options
        self.root = root
        self.training = training
        if training:
            self.decision_executor = TrainingDecisionExecutor(list(options.keys()))
        else:
            self.decision_executor = DecisionExecutor(list(options.keys())).activate_assertions(max_backtracks=4) 

        if dspy_model is not None:
            self.decision_executor.load(os.path.join("elysia", "training", "dspy_models", "decision", dspy_model + ".json"))

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

    def decide(self, 
            user_prompt: str, 
            completed_tasks: list[dict], 
            available_information: Returns, 
            conversation_history: list[dict], 
            decision_tree: dict, 
            previous_reasoning: dict,
            idx: int,
            data_queried: dict,
            collection_names: list[str],
            current_message: str = "",
            tree_count: str = "",
            **kwargs
        ):
        
        output, self.completed = self.decision_executor(
            user_prompt = user_prompt, 
            instruction = self.instruction, 
            available_tasks = self._options_to_json(), 
            available_information = available_information.to_llm_str(),
            completed_tasks = completed_tasks,
            conversation_history = conversation_history,
            data_queried = data_queried,
            decision_tree = decision_tree,
            previous_reasoning = previous_reasoning,
            future_tasks = self._options_to_future(),
            collection_names = collection_names,
            current_message = current_message,
            tree_count = tree_count,
            idx = idx,
            **kwargs
        )

        if self.training:
            self.decision = kwargs.get("task", "")
        else:
            self.decision = output.task
        self.reasoning = output.reasoning

        if self.options[self.decision]['action'] is not None:
            action_fn = self.options[self.decision]['action']
        else:
            action_fn = None

        return output, action_fn, self.completed

    def __call__(self, user_prompt: str, completed_tasks: list[dict], available_information: list[dict], conversation_history: list[dict], **kwargs):
        return self.decide(user_prompt, completed_tasks, available_information, conversation_history, **kwargs)

    def construct_as_previous_info(self):
        return {
            "id": self.id,
            "options": {
                key: option["description"] for key, option in self.options.items()
            },
            "decision": self.decision,
            "instruction": self.instruction
        }
    
class TreeReturner:
    def __init__(self, conversation_id: str, tree_index: int = 0):
        self.conversation_id = conversation_id
        self.tree_index = tree_index

    def _parse_error(self, error: str, query_id: str):
        return parse_error(error, self.conversation_id, query_id)
    
    def _parse_warning(self, warning: str, query_id: str):
        return parse_warning(warning, self.conversation_id, query_id)

    def _parse_status(self, status: Status, query_id: str):
        return status.to_json(self.conversation_id, query_id)
    
    def _parse_tree_update(self, node_id: str, decision: str, reasoning: str, reset: bool, query_id: str):
        return parse_tree_update(node_id, self.tree_index, decision, reasoning, self.conversation_id, reset, query_id)
    
    def _parse_result(self, result: Objects, code: str, query_id: str):
        return parse_result(result, code, self.conversation_id, query_id)
    
    def _parse_finished(self, query_id: str):
        return parse_finished(self.conversation_id, query_id)
    
    def _parse_text(self, text: Text, query_id: str):
        return parse_text(text, self.conversation_id, query_id)

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
            training_route: str = None,
            training_decision_output: bool = False,
            dspy_model: str = None
        ):
        
        self.conversation_id = conversation_id
        self.verbosity = verbosity
        self.break_down_instructions = break_down_instructions
        self.dspy_model = dspy_model
        self.collection_names = collection_names
        self.querier = AgenticQuery(
            collection_names=collection_names, 
            # TODO: make this adaptive based on the tree.objects file
            return_types={
                "conversation": "retrieve full conversations, including all messages and message authors, with timestamps and context of other messages in the conversation.",
                "message": "retrieve individual messages, only including the author of each individual message and timestamp, without surrounding context of other messages by different people.",
                "ticket": "retrieve individual tickets, including all fields of the ticket.",
                "ecommerce": "retrieve individual products, including all fields of the product.",
                "generic": "retrieve any other type of information that does not fit into the other categories."
            },
            verbosity=verbosity
        )

        self.aggregator = AgenticAggregate(
            collection_names=collection_names
        )

        self.current_message = ""
        self.query_id_to_prompt = {}
        self.prompt_to_query_id = {}

        # for training purposes, we may want to run the tree until a certain node
        # training route is e.g. "search/query/text_response"
        if training_route is not None:
            self.training_route = training_route.split("/")
        else:
            self.training_route = None

        # whether to output model reasoning for the decisions even if there is a route
        self.training_decision_output = training_decision_output

        self.num_trees_completed = 0
        self.max_recursions = 5

        self.decision_nodes = {}

        self.data_queried = {}
        self.data_queried_str = ""
        self.previous_info = []
        self.decision_history = []
        self.conversation_history = []
        self.previous_reasoning = {}
        self.tree_index = -1

        self.returns = Returns(
            retrieved = {}, 
            aggregation = {},
            text = Text(objects=[], metadata={})
        )

        self.returner = TreeReturner(conversation_id=self.conversation_id)

        if self.break_down_instructions:
            self.input_executor = InputExecutor()

        # default initialisations ---
        self.add_decision_node(
            id = "base",
            instruction = """
            Choose a task based on the user's prompt and the available information. 
            Use all information available to you to make the decision.
            If you _have searched already_ (based on completed_tasks), and there is no information (based on available_information), 
            you should assume that the task is impossible, hence choose text_response to reply this to the user.
            Otherwise, if you haven't searched yet, you should search the knowledge base.
            If you don't need to search, i.e. the user is talking to you, or you have already searched and there is no new information, choose text_response.
            If you have searched, and there is available information, you should choose summarize to reply this to the user.
            If you choose summarize, you should set all_actions_completed to True, since this is the last decision to make.
            """,
            options = {
                "search": {
                    "description": "Search the knowledge base. This should be used when the user is lacking information about a specific issue. This retrieves information only and provides no output to the user except the information.",
                    "future": "Choose to query, or aggregate information. Collections that can be queried are " + ", ".join(self.collection_names),
                    "action": None,
                    "next": "search",
                },
                "summarize": {
                    "description": "Summarize some already required information. This should be used when the user wants a high-level overview of some retrieved information or a generic response based on the information.  This is usually the last decision to make, so you should set all_actions_completed to True if you choose this.",
                    "future": "",
                    "action": Summarizer(),
                    "next": None    # next=None represents the end of the tree
                },
                "text_response": {
                    "description": "Respond to the user's prompt. This should be used when the user wants a response that is explicitly not any of the other options. These responses are informal, polite, and assistant-like. This is usually the last decision to make, so you should set all_actions_completed to True if you choose this.",
                    "future": "",
                    "action": TextResponse(),
                    "next": None
                }
            },
            root = True
        )

        self.add_decision_node(
            id = "search",
            instruction = """
            Choose between querying the knowledge base via semantic search, or aggregating information from the knowledge base.
            Querying is when the user is looking for specific information related to the content of the dataset, requiring a specific search query.
            Aggregating is when the user is looking for a high-level overview of the dataset, such as summary statistics of the quantity of some items. Aggregation can also include grouping information by some property and returning statistics about the groups.
            """,
            options = {
                "query": {
                    "description": "Query the knowledge base based on searching with a specific query.",
                    "future": "You will be given information about the collection, such as the fields and types of the data. Then, query with semantic search, keyword search, or a combination of both.",
                    "action": self.querier,
                    "next": None,
                },
                "aggregate": {
                    "description": "Perform functions such as counting, averaging, summing, etc. on the data, grouping by some property and returning metrics/statistics about different categories. Or providing a high level overview of the dataset.",
                    "future": "You will be given information about the collection, such as the fields and types of the data. Then, aggregate over different categories of the collection, with operations: top_occurences, count, sum, average, min, max, median, mode, group_by.",
                    "action": self.aggregator,
                    "next": None    # next=None represents the end of the tree
                }
            },
            root = False
        )

        self._get_root()
        self.tree = {}
        self._construct_tree(self.root, self.tree)

        # Get collection metadata
        self.collection_information = []
        for collection_name in self.collection_names:
            metadata_name = f"ELYSIA_METADATA_{collection_name}__"
            if client.collections.exists(metadata_name):
                metadata = client.collections.get(metadata_name).query.fetch_objects(limit=1)
                self.collection_information.append(metadata.objects[0].properties)
        
        if verbosity > 1:
            backend_print("Initialised tree with the following decision nodes:")
            for decision_node in self.decision_nodes.values():
                backend_print(f"  - [magenta]{decision_node.id}[/magenta]: {list(decision_node.options.keys())}")

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
        if append_to_previous and self.conversation_history[-1]["role"] != "user":
            self.conversation_history[-1]["content"] += message
        else:
            self.conversation_history.append({
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
            self.returns.add_retrieval(collection_name=action_result.metadata["collection_name"], objects=action_result)

            if action_result.metadata["collection_name"] not in self.data_queried:
                self.data_queried[action_result.metadata["collection_name"]] = [{
                    "type": "retrieval",
                    "count": len(action_result.objects),
                    "prompt": user_prompt
                }]
            else:
                self.data_queried[action_result.metadata["collection_name"]].append({
                    "type": "retrieval",
                    "count": len(action_result.objects),
                    "prompt": user_prompt
                })
            self.data_queried_str += f" - For query '{user_prompt}', queried '{action_result.metadata['collection_name']}' and retrieved {len(action_result.objects)} objects\n"
            

        if isinstance(action_result, Aggregation):
            self.returns.add_aggregation(collection_name=action_result.metadata["collection_name"], objects=action_result)

            if action_result.metadata["collection_name"] not in self.data_queried:
                self.data_queried[action_result.metadata["collection_name"]] = [{
                    "type": "aggregation",
                    "count": len(action_result.objects),
                    "prompt": user_prompt
                }]
            else:
                self.data_queried[action_result.metadata["collection_name"]].append({
                    "type": "aggregation",
                    "count": len(action_result.objects),
                    "prompt": user_prompt
                })
            self.data_queried_str += f" - For query '{user_prompt}', aggregated '{action_result.metadata['collection_name']} with description '{action_result.metadata['description'][-1]}'\n"
    
    async def _evaluate_action(self, action_fn: Callable, user_prompt: str, completed: bool, **kwargs):

        async for result in action_fn(
            user_prompt=user_prompt, 
            available_information=self.returns, 
            previous_reasoning=self.previous_reasoning,
            data_queried=self.data_queried_str,
            current_message=self.current_message,
            conversation_history=self.conversation_history,
            collection_information=self.collection_information,
            **kwargs
        ):
            if isinstance(result, Status):
                yield self.returner._parse_status(result, query_id = self.prompt_to_query_id[user_prompt])

            if isinstance(result, TreeUpdate):
                yield self.returner._parse_tree_update(
                    result.from_node, 
                    result.to_node, 
                    result.reasoning, 
                    result.last and not completed,
                    query_id = self.prompt_to_query_id[user_prompt]
                )

            if isinstance(result, Objects):
                self._update_returns(result, user_prompt)

                if len(result.objects) > 0:
                    yield self.returner._parse_result(
                        result, 
                        code=result.metadata["last_code"], 
                        query_id = self.prompt_to_query_id[user_prompt]
                    )

            if isinstance(result, Text):
                yield self.returner._parse_text(result, query_id = self.prompt_to_query_id[user_prompt])

            if isinstance(result, Response): 
                self.current_message, message_update = update_current_message(self.current_message, result.objects[0]["text"])
                if message_update != "":
                    self._update_conversation_history("assistant", message_update.strip(), append_to_previous=True)
                backend_print(f"Updated current message: [bold yellow]'{self.current_message}'[/bold yellow]")
                    
            if isinstance(result, Summary):
                self._update_conversation_history("assistant", result.objects[0]["text"], append_to_previous=False)

            if isinstance(result, Error):
                yield self.returner._parse_error(result.text, query_id = self.prompt_to_query_id[user_prompt])
                raise Exception(result.text)

            if isinstance(result, Warning):
                yield self.returner._parse_warning(result.text, query_id = self.prompt_to_query_id[user_prompt])

    def _remove_collection_from_data(self, collection_name: str):
        if collection_name in self.data_queried:
            del self.data_queried[collection_name]
        
        if collection_name in self.returns.retrieved:
            del self.returns.retrieved[collection_name]

    def set_collection_names(self, collection_names: list[str], remove_data: bool = False):
        collection_names_to_remove = [name for name in self.collection_names if name not in collection_names]

        if self.verbosity >= 1:
            backend_print(f"Setting collection names to: {collection_names}")
            backend_print(f"Collection names to remove: {collection_names_to_remove}")

        self.collection_names = collection_names
        self.querier.set_collection_names(collection_names)
        
        if remove_data:
            for collection_name in collection_names_to_remove:
                self._remove_collection_from_data(collection_name)

    def _decide_from_route(self, node_id: str):
        node = self.decision_nodes[node_id]
        possible_nodes = node.options.keys()

        next_route = self.training_route[0]
        if next_route not in possible_nodes:
            raise Exception(f"Next node in training route ({next_route}) not in possible nodes ({possible_nodes})")
        
        self.training_route = self.training_route[1:]
        completed = len(self.training_route) == 0
        
        return next_route, node.options[next_route]["action"], completed
        

    def hard_reset(self):
        self = Tree(verbosity=self.verbosity)

    def soft_reset(self):
        # conversation history is not reset
        # available information is not reset (returns)
        self.previous_info = []
        self.decision_history = []
        self.previous_reasoning = {}
        self.num_trees_completed = 0
        self.data_queried = {}

        self.tree_index += 1
        self.returner = TreeReturner(conversation_id=self.conversation_id, tree_index=self.tree_index)

    def add_decision_node(self, id: str, instruction: str, options: dict[str, dict[str, str]], root: bool = False):
        decision_node = DecisionNode(
            id, 
            instruction, 
            options, 
            root, 
            dspy_model = self.dspy_model, 
            training = self.training_decision_output and self.training_route is not None
        )
        self.decision_nodes[id] = decision_node
        return decision_node

    async def process(self, user_prompt: str, query_id: str = "1", recursion_counter: int = 0, first_run: bool = True, **kwargs):

        if recursion_counter > 5:
            backend_print(f"[bold red]Recursion limit reached! ({recursion_counter})[/bold red]")
            yield self.returner._parse_warning("Recursion limit reached!", query_id = self.prompt_to_query_id[user_prompt])
            raise RecursionLimitException("Recursion limit reached!")  # Force exit from the async function by raising an exception

        self.previous_reasoning[f"tree_{self.num_trees_completed+1}"] = {}

        if first_run:

            self.query_id_to_prompt[query_id] = user_prompt
            self.prompt_to_query_id[user_prompt] = query_id

            if self.break_down_instructions:
                user_prompt = self.input_executor(user_prompt)

            self._update_conversation_history("user", user_prompt)

            if self.verbosity >= 1:
                print(f"[bold yellow]User prompt:[/bold yellow][yellow]\n{user_prompt}[/yellow]")

        current_decision_node = self.decision_nodes[self.root]
        
        while True:

            if self.training_route is not None:
                task, action_fn, completed = self._decide_from_route(current_decision_node.id)

                if self.training_decision_output:
                    decision_kwargs = {"task": task}
                else:
                    decision_kwargs = {}
            else:
                decision_kwargs = {}
            
            if self.training_decision_output or self.training_route is None:

                decision, action_fn, model_completed = current_decision_node(
                    user_prompt=user_prompt, 
                    completed_tasks=self.previous_info,
                    available_information=self.returns,
                    conversation_history=self.conversation_history,
                    data_queried=self.data_queried_str,
                    decision_tree=self.tree,
                    previous_reasoning=self.previous_reasoning,
                    collection_names=self.collection_names,
                    current_message=self.current_message,
                    tree_count=f"{self.num_trees_completed}/{self.max_recursions}",
                    idx=self.num_trees_completed,
                    **decision_kwargs
                )
                if self.training_route is None:
                    task = decision.task
                    completed = model_completed
                    
                self.current_message, message_update = update_current_message(self.current_message, decision.text_return)
                print(f"Decision message: [bold yellow]'{message_update}'[/bold yellow]")

                if message_update != "" and not (task == "text_response" or task == "summarize"):
                    yield self.returner._parse_text(Response([{"text": message_update}], {}), query_id = self.prompt_to_query_id[user_prompt])
                yield self.returner._parse_tree_update(current_decision_node.id, task, decision.reasoning, False, query_id = self.prompt_to_query_id[user_prompt])
                
                self.previous_reasoning[f"tree_{recursion_counter+1}"][current_decision_node.id] = decision.reasoning
                self.previous_info.append(current_decision_node.construct_as_previous_info())

            self.decision_history.append(task)
                        

            if self.verbosity > 1:
                backend_print(f"Node: [magenta]{current_decision_node.id}[/magenta]")
                backend_print(f"Instruction: [italic]{current_decision_node.instruction.strip()}[/italic]")
                backend_print(f"Decision: [green]{task}[/green]\n")

            if action_fn is not None:
                async for result in self._evaluate_action(action_fn, user_prompt, completed, **kwargs):
                    yield result

            if current_decision_node.options[task]["next"] is None:
                break
            else:
                current_decision_node = self.decision_nodes[current_decision_node.options[task]["next"]]

        # end of the tree for this iteration
        if not completed:
            
            if self.verbosity == 2:
                backend_print("Model did [bold red]not[/bold red] complete overall goal! Restarting tree...")

            # recursive call to restart the tree since the goal was not completed
            self.num_trees_completed += 1
            async for result in self.process(user_prompt, query_id, recursion_counter + 1, first_run=False, **kwargs):
                yield result

        # end of all trees
        else:

            self.num_trees_completed += 1
            self.current_message = ""

            yield self.returner._parse_finished(query_id = self.prompt_to_query_id[user_prompt])

            if self.verbosity >= 1:
                backend_print(f"[bold green]Model identified overall goal as completed![/bold green]")
                backend_print(f"History:")

                for i, info in enumerate(self.previous_info):
                    backend_print(f"[Decision {i} ({info['id']})]: [bold]Instruction:[/bold] [cyan italic]{info['instruction']}[/cyan italic] -> [bold]Result:[/bold] [green]{info['decision']}[/green]")

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


