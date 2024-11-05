import json
import os
from typing import Callable, List
from rich import print

# from backend.text.prompt_executors import SummarizingExecutor, TextResponseExecutor
from backend.text.text import Summarizer, TextResponse
from backend.querying.agentic_query import QueryOptions
from backend.tree.prompt_executors import DecisionExecutor, InputExecutor
from backend.util.logging import backend_print
from backend.tree.objects import Text, Returns, GenericRetrieval

import dspy

lm = dspy.LM(model="gpt-4o", max_tokens=8000)

# lm = dspy.LM(model="claude-3-5-haiku-20241022", max_tokens=8000)
# lm = dspy.LM("groq/llama-3.2-1b-preview", max_tokens=8192)

dspy.settings.configure(lm=lm)

class DecisionNode:

    def __init__(self, id: str, instruction: str, options: list[dict[str, str]], root: bool = False):
        self.id = id
        self.instruction = instruction
        self.options = options
        self.root = root
        self.decision_executor = DecisionExecutor()

    def decide(self, user_prompt: str, completed_tasks: list[dict], available_information: Returns, conversation_history: list[dict], **kwargs):
        
        self.decision, self.completed = self.decision_executor(
            user_prompt = user_prompt, 
            instruction = self.instruction, 
            available_tasks = self.options, 
            available_information = available_information,
            completed_tasks = completed_tasks,
            conversation_history = conversation_history
        )

        if self.options[self.decision]['action'] is not None:
            action_fn = eval(self.options[self.decision]['action'])
        else:
            action_fn = None

        return self.decision, action_fn, self.completed

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

    def __init__(self, verbosity: int = 1, break_down_instructions: bool = False):
        self.previous_info = []
        self.decision_nodes = {}
        self.verbosity = verbosity
        self.break_down_instructions = break_down_instructions

        self.conversation_history = []

        self.returns = Returns(
            retrieved = {}, 
            text = Text(objects=[], metadata={})
        )

        self.input_executor = InputExecutor()

        # default initialisations ---
        self.add_decision_node(
            id = "initial",
            instruction = "choose a task based on the user's prompt and the available information",
            options = {
                "query": {
                    "description": "query the knowledge base. This should be used when the user is lacking information about a specific issue. This retrieves information only and provides no output to the user except the information.",
                    "action": None,
                    "returns": None,
                    "next": "collection",
                },
                "summarize": {
                    "description": "summarize some already required information. This should be used when the user wants a high-level overview of some retrieved information or a generic response based on the information.",
                    "action": 'Summarizer()', 
                    "returns": "text",
                    "next": None    # next=None represents the end of the tree
                },
                "text_response": {
                    "description": "respond to the user's prompt. This should be used when the user wants a response that is explicitly not any of the other options. These responses are informal, polite, and assistant-like.",
                    "action": 'TextResponse()',
                    "returns": "text",
                    "next": None
                }
            },
            root = True
        )

        self.add_decision_node(
            id = "collection",
            instruction = "choose a collection to query",
            options = {
                "example_verba_email_chains": {
                    "description": "email correspondence within the company that is loosely related to verba",
                    "action": 'QueryOptions["message"](collection_name = "example_verba_email_chains")',
                    "returns": "Conversation",
                    "next": None
                },
                "example_verba_slack_conversations": {
                    "description": "slack chats in the company that is loosely related to verba",
                    "action": 'QueryOptions["message"](collection_name = "example_verba_slack_conversations")',
                    "returns": "Conversation",
                    "next": None
                },
                "example_verba_github_issues": {
                    "description": "github issues for the verba app",
                    "action": 'QueryOptions["generic"](collection_name = "example_verba_github_issues")',
                    "returns": "Retrieved",
                    "next": None
                }
            },
            root = False
        )

        self._get_root()

        if verbosity > 1:
            backend_print("Initialised tree with the following decision nodes:")
            for decision_node in self.decision_nodes.values():
                backend_print(f"  - [magenta]{decision_node.id}[/magenta]: [italic]{decision_node.instruction}[/italic]")

    def _get_root(self):
        for decision_node in self.decision_nodes.values():            
            if decision_node.root:

                if "root" in dir(self):
                    raise ValueError("Multiple root decision nodes found")

                self.root = decision_node.id

        if "root" not in dir(self):
            raise ValueError("No root decision node found")

    def _update_conversation_history(self, user_prompt: str, assistant_response: str):
        self.conversation_history.append({
            "user": user_prompt,
            "assistant": assistant_response
        })

    def _update_returns(self, action_result: str | list, user_prompt: str):

        # this is where we should put some yields

        if isinstance(action_result, GenericRetrieval): # for now assume a list = retrieved objects
            self.returns.add_retrieval(collection_name=action_result.metadata["collection_name"], objects=action_result)

        elif isinstance(action_result, Text):
            self.returns.add_text(objects=action_result)
            self._update_conversation_history(user_prompt, action_result.objects[0])

    def _evaluate_actions(self, actions: list[Callable], user_prompt: str, **kwargs):
        for action_fn in actions:
            if action_fn is not None:
                result = action_fn(
                    user_prompt=user_prompt, 
                    available_information=self.returns, 
                    **kwargs
                )
                self._update_returns(result, user_prompt)

    def reset(self):
        self = Tree(verbosity=self.verbosity)

    def add_decision_node(self, id: str, instruction: str, options: dict[str, dict[str, str]], root: bool = False):
        decision_node = DecisionNode(id, instruction, options, root)
        self.decision_nodes[id] = decision_node
        return decision_node

    def process(self, user_prompt: str, recursion_counter: int = 0, first_run: bool = True, **kwargs) -> dict:

        if recursion_counter > 5:
            backend_print(f"[bold red]Recursion limit reached! ({recursion_counter})[/bold red]")
            return self.returns
        
        if first_run:

            if self.break_down_instructions:
                user_prompt = self.input_executor(user_prompt)

            if self.verbosity >= 1:
                print(f"[bold yellow]User prompt:[/bold yellow][yellow]\n{user_prompt}[/yellow]")

        current_decision_node = self.decision_nodes[self.root]
        
        actions = []

        while True:

            if self.verbosity > 1:
                backend_print(f"Node: [magenta]{current_decision_node.id}[/magenta]")
                backend_print(f"Instruction: [italic]{current_decision_node.instruction}[/italic]")

            decision, action_fn, completed = current_decision_node(
                user_prompt=user_prompt, 
                completed_tasks=self.previous_info,
                available_information=self.returns,
                conversation_history=self.conversation_history,
                **kwargs
            )
            actions.append(action_fn) # update list of actions for later evaluation

            # another yield here for decision that was made (for updates to frontend)

            if self.verbosity > 1:
                backend_print(f"Decision: [magenta]{decision}[/magenta]")

            self.previous_info.append(current_decision_node.construct_as_previous_info())

            if current_decision_node.options[decision]["next"] is None:
                break
            else:
                current_decision_node = self.decision_nodes[current_decision_node.options[decision]["next"]]

        # evaluate actions
        self._evaluate_actions(actions, user_prompt, **kwargs)
        # self._update_returns(result, user_prompt)

        if not completed:
            
            if self.verbosity == 2:
                backend_print("Model did [bold red]not[/bold red] complete overall goal! Restarting tree...")

            # recursive call to restart the tree since the goal was not completed
            self.process(user_prompt, recursion_counter + 1, first_run=False, **kwargs)

        else:
            
            if self.verbosity >= 1:
                backend_print(f"[bold green]Model identified overall goal as completed![/bold green]")
                backend_print(f"History:")

                for i, info in enumerate(self.previous_info):
                    backend_print(f"[Decision {i} ({info['id']})]: [bold]Instruction:[/bold] [cyan italic]{info['instruction']}[/cyan italic] -> [bold]Result:[/bold] [green]{info['decision']}[/green]")

        return self.returns

