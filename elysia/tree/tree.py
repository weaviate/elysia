import json
import os
from typing import Callable, List
from rich import print

import asyncio
import nest_asyncio

# from elysia.text.prompt_executors import SummarizingExecutor, TextResponseExecutor
from elysia.text.text import Summarizer, TextResponse
from elysia.querying.agentic_query import AgenticQuery
from elysia.tree.prompt_executors import DecisionExecutor, InputExecutor
from elysia.util.parsing import remove_whitespace
from elysia.util.logging import backend_print
from elysia.util.api import parse_decision, parse_result, parse_finished, parse_error, parse_warning
from elysia.tree.objects import Text, Returns, Objects, Status
from elysia.querying.objects import Retrieval

import dspy
from dspy.primitives.assertions import assert_transform_module, backtrack_handler


# lm = dspy.LM(model="gpt-4o-mini", max_tokens=8000)

lm = dspy.LM(model="claude-3-5-haiku-20241022", max_tokens=8000)
# lm = dspy.LM("groq/llama-3.2-3b-preview", max_tokens=8192)

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
            out[node] = {key: value for key, value in self.options[node].items() if key not in ["action", "returns", "next"]}
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
            idx = idx
        )

        self.decision = output.task
        self.reasoning = output.reasoning

        if self.options[self.decision]['action'] is not None:
            action_fn = self.options[self.decision]['action']
        else:
            action_fn = None

        return self.decision, self.reasoning, action_fn, self.completed

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
        self.querier = AgenticQuery(collection_names=collection_names, return_types=["conversation", "message", "ticket", "generic"])

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

        self.returns = Returns(
            retrieved = {}, 
            text = Text(objects=[], metadata={})
        )

        if self.break_down_instructions:
            self.input_executor = InputExecutor()

        # default initialisations ---
        self.add_decision_node(
            id = "base",
            instruction = """
            Choose a task based on the user's prompt and the available information. 
            Use all information available to you to make the decision.
            If you _have queried already_ (based on completed_tasks), and there is no information (based on available_information), 
            you should assume that the task is impossible, hence choose text_response to reply this to the user.
            Otherwise, if you haven't queried yet, you should query the knowledge base.
            If you don't need to query, i.e. the user is talking to you, choose text_response.
            If you have queried, and there is available information, you should choose summarize to reply this to the user.
            If the user is just talking to you and requires no information, choose text_response.
            """,
            options = {
                "query": {
                    "description": "query the knowledge base. This should be used when the user is lacking information about a specific issue. This retrieves information only and provides no output to the user except the information.",
                    "future_options": ["collections that can be queried are " + ", ".join(self.collection_names)],
                    "status": "Deciding which collection to query",
                    "action": self.querier,
                    "returns": None,
                    "next": None,
                },
                "summarize": {
                    "description": "summarize some already required information. This should be used when the user wants a high-level overview of some retrieved information or a generic response based on the information.  This is usually the last decision to make, so you should set all_actions_completed to True if you choose this.",
                    "future_options": [],
                    "status": "Summarizing information",
                    "action": Summarizer(), 
                    "returns": "text",
                    "next": None    # next=None represents the end of the tree
                },
                "text_response": {
                    "description": "respond to the user's prompt. This should be used when the user wants a response that is explicitly not any of the other options. These responses are informal, polite, and assistant-like. This is usually the last decision to make, so you should set all_actions_completed to True if you choose this.",
                    "future_options": [],
                    "status": "Crafting response",
                    "action": TextResponse(),
                    "returns": "text",
                    "next": None
                }
            },
            root = True
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

    def _update_conversation_history(self, role: str, message: str):
        self.conversation_history.append({
            "role": role,
            "content": message
        })

    def _construct_tree(self, node_id: str, tree: dict):
        decision_node = self.decision_nodes[node_id]
        
        # Set the base node information outside the loop
        tree["id"] = node_id
        tree["instruction"] = decision_node.instruction
        tree["options"] = {}
        
        # Initialize all options first
        for option in decision_node.options:
            tree["options"][option] = {}
        
        # Then handle the recursive cases
        for option in decision_node.options:
            if decision_node.options[option]["next"] is not None:
                tree["options"][option] = self._construct_tree(
                    decision_node.options[option]["next"], 
                    tree["options"][option]
                )
        
        return tree

    def _update_returns(self, action_result: str | list, user_prompt: str):

        if isinstance(action_result, Retrieval): 
            self.returns.add_retrieval(collection_name=action_result.metadata["collection_name"], objects=action_result)

            if action_result.metadata["collection_name"] not in self.data_queried:
                self.data_queried[action_result.metadata["collection_name"]] = len(action_result.objects)
            else:
                self.data_queried[action_result.metadata["collection_name"]] += len(action_result.objects)

        if isinstance(action_result, Text):
            self.returns.add_text(objects=action_result)
            self._update_conversation_history("assistant", action_result.objects[0])

    async def _evaluate_action(self, action_fn: Callable, user_prompt: str, **kwargs):

        async for result in action_fn(
            user_prompt, 
            self.returns, 
            self.previous_reasoning,
            data_queried=self.data_queried,
            **kwargs
        ):
            if isinstance(result, Status):
                yield self._parse_status(result)

            if isinstance(result, Objects):
                self._update_returns(result, user_prompt)

                if len(result.objects) > 0:
                    yield self._parse_result(result)
    
    def _parse_error(self, error: str):
        return parse_error(error, self.conversation_id)
    
    def _parse_warning(self, warning: str):
        return parse_warning(warning, self.conversation_id)

    def _parse_status(self, status: Status):
        return status.to_json(self.conversation_id)
    
    def _parse_decision(self, id: str, decision: str, reasoning: str, instruction: str):
        return parse_decision(decision, reasoning, self.conversation_id, id, instruction, {})

    def _parse_result(self, result: Objects):
        return parse_result(result, self.conversation_id)
    
    def _parse_finished(self):
        return parse_finished(self.conversation_id)

    def hard_reset(self):
        self = Tree(verbosity=self.verbosity)

    def soft_reset(self):
        # conversation history is not reset
        # available information is not reset (returns)
        self.previous_info = []
        self.decision_history = []
        self.previous_reasoning = {}
        self.num_trees_completed = 0

    def add_decision_node(self, id: str, instruction: str, options: dict[str, dict[str, str]], root: bool = False):
        decision_node = DecisionNode(id, instruction, options, root, dspy_model = self.dspy_model)
        self.decision_nodes[id] = decision_node
        return decision_node

    async def process(self, user_prompt: str, recursion_counter: int = 0, first_run: bool = True, **kwargs):

        if recursion_counter > 5:
            backend_print(f"[bold red]Recursion limit reached! ({recursion_counter})[/bold red]")
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

            decision, reasoning, action_fn, completed = current_decision_node(
                user_prompt=user_prompt, 
                completed_tasks=self.previous_info,
                available_information=self.returns,
                conversation_history=self.conversation_history,
                data_queried=self.data_queried,
                decision_tree=self.tree,
                previous_reasoning=self.previous_reasoning,
                idx=self.num_trees_completed
            )

            yield self._parse_decision(current_decision_node.id, decision, reasoning, current_decision_node.instruction)
            yield self._parse_status(Status(current_decision_node.options[decision]["status"]))

            self.previous_reasoning[f"tree_{recursion_counter+1}"][current_decision_node.id] = reasoning
            self.decision_history.append(decision)
            self.previous_info.append(current_decision_node.construct_as_previous_info())

            if action_fn is not None:
                async for result in self._evaluate_action(action_fn, user_prompt, **kwargs):
                    yield result

            if self.verbosity > 1:
                backend_print(f"Node: [magenta]{current_decision_node.id}[/magenta]")
                backend_print(f"Instruction: [italic]{current_decision_node.instruction.strip()}[/italic]")
                backend_print(f"Decision: [green]{decision}[/green]\n")

            if current_decision_node.options[decision]["next"] is None:
                break
            else:
                current_decision_node = self.decision_nodes[current_decision_node.options[decision]["next"]]

        if not completed or (self.run_num_trees is not None and self.num_trees_completed < self.run_num_trees):
            
            if self.verbosity == 2:
                backend_print("Model did [bold red]not[/bold red] complete overall goal! Restarting tree...")

            # recursive call to restart the tree since the goal was not completed
            self.num_trees_completed += 1
            async for result in self.process(user_prompt, recursion_counter + 1, first_run=False, **kwargs):
                yield result

        else:

            self.num_trees_completed += 1
            
            yield self._parse_finished()

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


