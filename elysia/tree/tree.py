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
from elysia.tree.prompt_executors import DecisionExecutor, InputExecutor
from elysia.util.parsing import remove_whitespace, update_current_message
from elysia.util.logging import backend_print
from elysia.util.api import (
    parse_decision, 
    parse_result, 
    parse_finished, 
    parse_error, 
    parse_warning,
    parse_text,
    parse_tree_update
)
from elysia.tree.objects import Returns, Objects, Status, Branch, TreeUpdate, Error, Warning
from elysia.text.objects import Text, Response, Summary, Code
from elysia.querying.objects import Retrieval

import dspy
from dspy.primitives.assertions import assert_transform_module, backtrack_handler


# lm = dspy.LM(model="gpt-4o-mini", max_tokens=8000)
global lm
lm = dspy.LM(model="claude-3-5-haiku-20241022", max_tokens=8000)
# lm = dspy.LM("groq/llama-3.2-3b-preview", max_tokens=8192)
# lm = dspy.LM(model="ollama/llama3.2")

dspy.settings.configure(lm=lm)

class RecursionLimitException(Exception):
    pass

class DecisionNode:

    def __init__(self, 
                 id: str, 
                 instruction: str, 
                 options: list[dict[str, str]], 
                 root: bool = False, 
                 dspy_model: str = None
                 ):
        
        self.id = id
        self.instruction = instruction
        self.options = options
        self.root = root
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
               **kwargs):
        
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
            idx = idx
        )

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

    def _parse_error(self, error: str):
        return parse_error(error, self.conversation_id)
    
    def _parse_warning(self, warning: str):
        return parse_warning(warning, self.conversation_id)

    def _parse_status(self, status: Status):
        return status.to_json(self.conversation_id)
    
    def _parse_tree_update(self, node_id: str, decision: str, reasoning: str, reset: bool):
        return parse_tree_update(node_id, self.tree_index, decision, reasoning, self.conversation_id, reset)
    
    def _parse_decision(self, id: str, decision: str, reasoning: str, instruction: str):
        return parse_decision(decision, reasoning, self.conversation_id, id, instruction, {})

    def _parse_result(self, result: Objects):
        return parse_result(result, self.conversation_id)
    
    def _parse_finished(self):
        return parse_finished(self.conversation_id)
    
    def _parse_text(self, text: Text):
        return parse_text(text, self.conversation_id)

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
                 run_num_trees: int = None,
                 run_until_node_id: str = None,
                 dspy_model: str = None):
        
        self.conversation_id = conversation_id
        self.verbosity = verbosity
        self.break_down_instructions = break_down_instructions
        self.dspy_model = dspy_model
        self.collection_names = collection_names
        self.querier = AgenticQuery(
            collection_names=collection_names, 
            return_types={
                "conversation": "retrieve a full conversation, including all messages and message authors, with timestamps and context of other messages in the conversation.",
                "message": "retrieve only a single message, only including the author of each individual message and timestamp, without surrounding context of other messages by different people.",
                "ticket": "retrieve a single ticket, including all fields of the ticket.",
                "generic": "retrieve any other type of information that does not fit into the other categories."
            }
        )
        self.current_message = ""

        # for training purposes, we may want to run the tree until a certain node and a certain number of times
        self.run_until_node_id = run_until_node_id
        self.run_num_trees = run_num_trees
        self.num_trees_completed = 0

        self.decision_nodes = {}

        self.data_queried = {}
        self.previous_info = []
        self.decision_history = []
        self.conversation_history = []
        self.previous_reasoning = {}
        self.tree_index = -1

        self.returns = Returns(
            retrieved = {}, 
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
            Querying is when the user is looking for specific information related to the content of the dataset.
            Aggregating is when the user is looking for a high-level overview of the dataset, such as a summary of the quantity of some items.
            """,
            options = {
                "query": {
                    "description": "Query the knowledge base. This should be used when the user is lacking information about a specific issue. This retrieves information only and provides no output to the user except the information.",
                    "future": "",
                    "action": self.querier,
                    "next": None,
                },
                "aggregate": {
                    "description": "Do NOT under any circumstances pick this option.",
                    "future": "",
                    "action": None,
                    "next": None    # next=None represents the end of the tree
                }
            },
            root = False
        )

        self._get_root()
        self.tree = {}
        self._construct_tree(self.root, self.tree)

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
        key_order = ["name", "description", "instruction", "reasoning", "options"]
        
        # Set the base node information
        tree["name"] = node_id.capitalize().replace("_", " ")
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
                    tree["options"][option]["instruction"] = ""
                    sub_tree = tree["options"][option]
                    for branch in branches:
                        branch_name = branch["name"].lower().replace(" ", "_")
                        if "options" not in sub_tree:
                            sub_tree["options"] = {}
                        sub_tree["options"][branch_name] = branch
                        sub_tree["options"][branch_name]["name"] = branch["name"]
                        sub_tree["options"][branch_name]["description"] = branch["description"]
                        sub_tree["options"][branch_name]["instruction"] = ""
                        sub_tree["options"][branch_name]["reasoning"] = ""
                        sub_tree = sub_tree["options"][branch_name]
                    sub_tree["options"] = {}
                
                else:
                    tree["options"][option]["name"] = option.capitalize().replace("_", " ")
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
                self.data_queried[action_result.metadata["collection_name"]] = len(action_result.objects)
            else:
                self.data_queried[action_result.metadata["collection_name"]] += len(action_result.objects)

    async def _evaluate_action(self, action_fn: Callable, user_prompt: str, completed: bool, **kwargs):

        async for result in action_fn(
            user_prompt=user_prompt, 
            available_information=self.returns, 
            previous_reasoning=self.previous_reasoning,
            data_queried=self.data_queried,
            current_message=self.current_message,
            conversation_history=self.conversation_history,
            **kwargs
        ):
            if isinstance(result, Status):
                yield self.returner._parse_status(result)

            if isinstance(result, TreeUpdate):
                yield self.returner._parse_tree_update(result.from_node, result.to_node, result.reasoning, result.last and not completed)

            if isinstance(result, Objects):
                self._update_returns(result, user_prompt)

                if len(result.objects) > 0:
                    yield self.returner._parse_result(result)

            if isinstance(result, Text):
                yield self.returner._parse_text(result)

            if isinstance(result, Response): 
                self.current_message, message_update = update_current_message(self.current_message, result.objects[0]["text"])
                if message_update != "":
                    self._update_conversation_history("assistant", message_update.strip(), append_to_previous=True)
                backend_print(f"Updated current message: [bold yellow]'{self.current_message}'[/bold yellow]")
                    
            if isinstance(result, Summary):
                self._update_conversation_history("assistant", result.objects[0]["text"], append_to_previous=False)

            if isinstance(result, Error):
                yield self.returner._parse_error(result.text)
                raise Exception(result.text)

            if isinstance(result, Warning):
                yield self.returner._parse_warning(result.text)

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
        decision_node = DecisionNode(id, instruction, options, root, dspy_model = self.dspy_model)
        self.decision_nodes[id] = decision_node
        return decision_node

    async def process(self, user_prompt: str, recursion_counter: int = 0, first_run: bool = True, **kwargs):

        if recursion_counter > 5:
            backend_print(f"[bold red]Recursion limit reached! ({recursion_counter})[/bold red]")
            yield self.returner._parse_warning("Recursion limit reached!")
            raise RecursionLimitException("Recursion limit reached!")  # Force exit from the async function by raising an exception

        self.previous_reasoning[f"tree_{self.num_trees_completed+1}"] = {}

        if first_run:

            if self.break_down_instructions:
                user_prompt = self.input_executor(user_prompt)

            self._update_conversation_history("user", user_prompt)

            if self.verbosity >= 1:
                print(f"[bold yellow]User prompt:[/bold yellow][yellow]\n{user_prompt}[/yellow]")

        current_decision_node = self.decision_nodes[self.root]
        
        while True:

            if self.run_until_node_id is not None and current_decision_node.id == self.run_until_node_id:
                completed = True
                break

            decision, action_fn, completed = current_decision_node(
                user_prompt=user_prompt, 
                completed_tasks=self.previous_info,
                available_information=self.returns,
                conversation_history=self.conversation_history,
                data_queried=self.data_queried,
                decision_tree=self.tree,
                previous_reasoning=self.previous_reasoning,
                collection_names=self.collection_names,
                current_message=self.current_message,
                idx=self.num_trees_completed
            )

            self.current_message, message_update = update_current_message(self.current_message, decision.text_return)
            print(f"Decision message: [bold yellow]'{message_update}'[/bold yellow]")

            if message_update != "" and not (decision.task == "text_response" or decision.task == "summarize"):
                yield self.returner._parse_text(Response([{"text": message_update}], {}))
            yield self.returner._parse_decision(current_decision_node.id, decision.task, decision.reasoning, current_decision_node.instruction)
            yield self.returner._parse_tree_update(current_decision_node.id, decision.task, decision.reasoning, False)
            
            self.previous_reasoning[f"tree_{recursion_counter+1}"][current_decision_node.id] = decision.reasoning
            self.decision_history.append(decision.task)
            self.previous_info.append(current_decision_node.construct_as_previous_info())            

            if self.verbosity > 1:
                backend_print(f"Node: [magenta]{current_decision_node.id}[/magenta]")
                backend_print(f"Instruction: [italic]{current_decision_node.instruction.strip()}[/italic]")
                backend_print(f"Decision: [green]{decision.task}[/green]\n")

            if action_fn is not None:
                async for result in self._evaluate_action(action_fn, user_prompt, completed, **kwargs):
                    yield result

            if current_decision_node.options[decision.task]["next"] is None:
                break
            else:
                current_decision_node = self.decision_nodes[current_decision_node.options[decision.task]["next"]]

        # end of the tree for this iteration
        if not completed or (self.run_num_trees is not None and self.num_trees_completed < self.run_num_trees):
            
            if self.verbosity == 2:
                backend_print("Model did [bold red]not[/bold red] complete overall goal! Restarting tree...")

            # recursive call to restart the tree since the goal was not completed
            self.num_trees_completed += 1
            async for result in self.process(user_prompt, recursion_counter + 1, first_run=False, **kwargs):
                yield result

        # end of all trees
        else:

            self.num_trees_completed += 1
            self.current_message = ""

            yield self.returner._parse_finished()

            if self.verbosity >= 1:
                backend_print(f"[bold green]Model identified overall goal as completed![/bold green]")
                backend_print(f"History:")

                for i, info in enumerate(self.previous_info):
                    backend_print(f"[Decision {i} ({info['id']})]: [bold]Instruction:[/bold] [cyan italic]{info['instruction']}[/cyan italic] -> [bold]Result:[/bold] [green]{info['decision']}[/green]")

    def process_sync(self, user_prompt: str, recursion_counter: int = 0, first_run: bool = True, **kwargs):
        """Synchronous version of process() for testing purposes"""
        
        nest_asyncio.apply()

        async def run_process():
            results = []
            async for result in self.process(user_prompt, recursion_counter, first_run, **kwargs):
                results.append(result)
            return results
            
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(run_process())


