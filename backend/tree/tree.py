import json
from typing import Callable, List

from backend.summarizing.prompt_executors import SummarizingExecutor
from backend.summarizing.summarize import Summarizer
from backend.querying.query import QueryOptions
from backend.tree.prompt_executors import DecisionExecutor
from backend.util.logging import backend_print
from backend.tree.objects import Text, Returns

import dspy

lm = dspy.LM(model="gpt-4o-mini", max_tokens=8000)
dspy.settings.configure(lm=lm)

class DecisionNode:

    def __init__(self, id: str, instruction: str, options: list[dict[str, str]], depends: str = None):
        self.id = id
        self.instruction = instruction
        self.options = options
        self.depends = depends

    def decide(self, user_prompt: str, completed_tasks: list[dict], available_information: dict, **kwargs):
        
        decision_executor = DecisionExecutor()

        self.decision, self.completed = decision_executor(
            user_prompt = user_prompt, 
            instruction = self.instruction, 
            available_tasks = self.options, 
            available_information = available_information,
            completed_tasks = completed_tasks
        )

        if self.options[self.decision]['action'] is not None:
            action_fn = eval(self.options[self.decision]['action'])
            self.result, metadata = action_fn(user_prompt=user_prompt, completed_tasks=completed_tasks, available_information=available_information, **kwargs)
        else:
            self.result = None
            metadata = {}

        return self.decision, self.result, self.completed, metadata

    def __call__(self, user_prompt: str, completed_tasks: list[dict], available_information: list[dict], **kwargs):
        return self.decide(user_prompt, completed_tasks, available_information, **kwargs)

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

    def __init__(self, verbosity: int = 1):
        self.previous_info = []
        self.decision_nodes = {}
        self.verbosity = verbosity

        self.returns = Returns(
            retrieved = {}, 
            text = Text(objects=[], metadata={})
        )

        # default initialisations ---
        self.add_decision_node(
            id = "initial",
            instruction = "choose a task based on the user's prompt and the available information",
            options = {
                "query": {
                    "description": "query the knowledge base. This should be used when the user is lacking information about a specific issue.",
                    "action": None,
                    "returns": None,
                    "next": "collection",
                },
                "summarize": {
                    "description": "summarize some information. This should be used when the user wants a high-level overview of some retrieved information.",
                    "action": 'Summarizer()', 
                    "returns": "text",
                    "next": None    # next=None represents the end of the tree
                }
            },
            depends = None
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
                    "action": 'QueryOptions["issue"](collection_name = "example_verba_github_issues")',
                    "returns": "Retrieved",
                    "next": None
                }
            },
            depends = "initial"
        )

        self._get_dependencies()

        if verbosity > 1:
            backend_print("Initialised tree with the following decision nodes:")
            for decision_node in self.decision_nodes.values():
                backend_print(f"  - [magenta]{decision_node.id}[/magenta]: [italic]{decision_node.instruction}[/italic]")

    def _get_dependencies(self):
        self.dependencies = {}
        for decision_node in self.decision_nodes.values():
            if decision_node.depends is not None:
                self.dependencies[decision_node.depends] = decision_node.id
            else:
                if "root" in self.dependencies:
                    raise ValueError("Multiple root decisions found")
                self.root = decision_node.id

    def _update_returns(self, action_result: str | list, metadata: dict = {}):

        # this is where we should put some yields

        if isinstance(action_result, list): # for now assume a list = retrieved objects
            self.returns.add_retrieval(collection_name=metadata["collection_name"], objects=action_result, metadata=metadata)

        elif isinstance(action_result, str):
            self.returns.add_text(objects=[action_result], metadata=metadata)

    def add_decision_node(self, id: str, instruction: str, options: dict[str, dict[str, str]], depends: str = None):
        decision_node = DecisionNode(id, instruction, options, depends)
        self.decision_nodes[id] = decision_node
        return decision_node

    def process(self, user_prompt: str, recursion_counter: int = 0, **kwargs) -> dict:

        if recursion_counter > 5:
            backend_print(f"[bold red]Recursion limit reached![/bold red]")
            return self.returns

        current_decision_node = self.decision_nodes[self.root]
        
        while True:

            if self.verbosity > 1:
                backend_print(f"Node: [magenta]{current_decision_node.id}[/magenta]")
                backend_print(f"Instruction: [italic]{current_decision_node.instruction}[/italic]")

            decision, action_result, completed, metadata = current_decision_node(
                user_prompt=user_prompt, 
                completed_tasks=self.previous_info,
                available_information=self.returns,
                **kwargs
            )

            # another yield here for decision that was made (for updates to frontend)

            if self.verbosity > 1:
                backend_print(f"Decision: [magenta]{decision}[/magenta]")

                if completed:
                    backend_print(f"[bold green]Model identified overall goal as completed![/bold green]")
                else:
                    backend_print(f"Model identified overall goal as [red]not[/red] completed ([italic]yet[/italic]).")

            self.previous_info.append(current_decision_node.construct_as_previous_info())

            if current_decision_node.options[decision]["next"] is None:
                break
            else:
                current_decision_node = self.decision_nodes[current_decision_node.options[decision]["next"]]


        if not completed:
            
            if self.verbosity == 2:
                backend_print("Model did [bold red]not[/bold red] complete overall goal! Restarting tree...")

            self._update_returns(action_result, metadata)
            self.process(user_prompt, recursion_counter + 1, **kwargs)
        else:
            
            if self.verbosity >= 1:
                backend_print(f"[bold green]Model identified overall goal as completed![/bold green]")
                backend_print(f"History:")

                for i, info in enumerate(self.previous_info):
                    backend_print(f"[Decision {i} ({info['id']})]: [bold]Instruction:[/bold] [cyan italic]{info['instruction']}[/cyan italic] -> [bold]Result:[/bold] [green]{info['decision']}[/green]")

            self._update_returns(action_result, metadata)

        return self.returns

