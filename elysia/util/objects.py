import time
import uuid
from copy import deepcopy
import dspy
from pydantic import BaseModel
from typing import Any
from elysia.util.parsing import format_dict_to_serialisable
from logging import Logger
from elysia.objects import Update


class Tracker:
    """
    Simple class to track:
    - the average time taken for an LLM call
    - number of calls made
    - number of input/output tokens used
    """

    def __init__(self, tracker_names: list[str], logger: Logger):
        self.trackers = {
            name: {
                "timer": {
                    "calls": 0,
                    "avg_time": 0,
                    "total_time": 0,
                    "start_time": None,
                    "end_time": None,
                }
            }
            for name in tracker_names
        }
        self.trackers = {
            **self.trackers,
            "models": {
                "base_lm": {
                    "calls": 0,
                    "cost": None,
                    "avg_cost": None,
                    "input_tokens": None,
                    "output_tokens": None,
                    "avg_input_tokens": None,
                    "avg_output_tokens": None,
                },
                "complex_lm": {
                    "calls": 0,
                    "cost": None,
                    "avg_cost": None,
                    "input_tokens": None,
                    "output_tokens": None,
                    "avg_input_tokens": None,
                    "avg_output_tokens": None,
                },
            },
        }
        self.logger = logger

    def start_tracking(self, tracker_name: str):
        self.trackers[tracker_name]["timer"]["start_time"] = time.perf_counter()

    def update_lm_costs(self, lm: dspy.LM | None = None, model_type: str = "base_lm"):

        if lm is not None:

            # check how many calls have been made
            prev_calls = self.trackers["models"][model_type]["calls"]
            total_calls = len(lm.history)
            self.trackers["models"][model_type]["calls"] = total_calls
            num_calls = total_calls - prev_calls

            if num_calls == 0:
                return

            history = lm.history[-num_calls:]

            input_tokens = 0
            for h in history:
                if "usage" in h and "prompt_tokens" in h["usage"]:
                    input_tokens += h["usage"]["prompt_tokens"]

            output_tokens = 0
            for h in history:
                if "usage" in h and "completion_tokens" in h["usage"]:
                    output_tokens += h["usage"]["completion_tokens"]

            cost = 0
            for h in history:
                if "cost" in h and h["cost"] is not None:
                    cost += h["cost"]

            if self.trackers["models"][model_type]["input_tokens"] is None:
                self.trackers["models"][model_type]["input_tokens"] = input_tokens
            else:
                self.trackers["models"][model_type]["input_tokens"] += input_tokens

            if self.trackers["models"][model_type]["output_tokens"] is None:
                self.trackers["models"][model_type]["output_tokens"] = output_tokens
            else:
                self.trackers["models"][model_type]["output_tokens"] += output_tokens

            if self.trackers["models"][model_type]["cost"] is None:
                self.trackers["models"][model_type]["cost"] = cost
            else:
                self.trackers["models"][model_type]["cost"] += cost

    def end_tracking(
        self,
        tracker_name: str,
        call_name: str = "",
        base_lm: dspy.LM | None = None,
        complex_lm: dspy.LM | None = None,
    ):
        if self.trackers[tracker_name]["timer"]["start_time"] is None:
            self.logger.warning(f"Tracker {tracker_name} has not been started yet!")
            return

        self.trackers[tracker_name]["timer"]["calls"] += 1

        time_taken = (
            time.perf_counter() - self.trackers[tracker_name]["timer"]["start_time"]
        )
        self.update_avg_time(tracker_name, time_taken)
        self.update_lm_costs(base_lm, "base_lm")
        self.update_lm_costs(complex_lm, "complex_lm")

        if call_name != "":
            self.logger.debug(
                f"Time taken for {call_name} ({tracker_name}): {time_taken: .2f} seconds"
            )
        else:
            self.logger.debug(
                f"Time taken for {tracker_name}: {time_taken: .2f} seconds"
            )

    def update_avg_time(self, tracker_name: str, time_taken: float):
        self.trackers[tracker_name]["timer"]["total_time"] += time_taken
        self.trackers[tracker_name]["timer"]["avg_time"] = (
            self.trackers[tracker_name]["timer"]["total_time"]
            / self.trackers[tracker_name]["timer"]["calls"]
        )

    def remove_tracker(self, tracker_name: str):
        self.trackers.pop(tracker_name)

    def add_tracker(self, tracker_name: str):
        self.trackers[tracker_name] = {
            "timer": {
                "calls": 0,
                "avg_time": 0,
                "total_time": 0,
                "start_time": None,
                "end_time": None,
            }
        }

    def get_num_calls(self, model_type: str):
        return self.trackers["models"][model_type]["calls"]

    def get_average_time(self, tracker_name: str):
        return self.trackers[tracker_name]["timer"]["avg_time"]

    def get_total_input_tokens(self, model_type: str):
        return self.trackers["models"][model_type]["input_tokens"]

    def get_total_output_tokens(self, model_type: str):
        return self.trackers["models"][model_type]["output_tokens"]

    def get_total_cost(self, model_type: str):
        return self.trackers["models"][model_type]["cost"]

    def get_average_input_tokens(self, model_type: str):
        return (
            (
                self.trackers["models"][model_type]["input_tokens"]
                / self.trackers["models"][model_type]["calls"]
            )
            if self.trackers["models"][model_type]["input_tokens"] is not None
            else 0
        )

    def get_average_output_tokens(self, model_type: str):
        return (
            (
                self.trackers["models"][model_type]["output_tokens"]
                / self.trackers["models"][model_type]["calls"]
            )
            if self.trackers["models"][model_type]["output_tokens"] is not None
            else 0
        )

    def get_average_cost(self, model_type: str):
        return (
            (
                self.trackers["models"][model_type]["cost"]
                / self.trackers["models"][model_type]["calls"]
            )
            if self.trackers["models"][model_type]["cost"] is not None
            else 0
        )

    def reset_trackers(self):
        for tracker in self.trackers:
            self.trackers[tracker]["timer"] = {
                "calls": 0,
                "avg_time": 0,
                "total_time": 0,
                "start_time": None,
                "end_time": None,
            }
            self.trackers[tracker]["models"] = {
                "base_lm": {
                    "calls": 0,
                    "input": None,
                    "output": None,
                },
                "complex_lm": {
                    "calls": 0,
                    "input": None,
                    "output": None,
                },
            }


class TreeUpdate:
    """
    Frontend update to represent what nodes have been updated.
    """

    def __init__(
        self,
        from_node: str,
        to_node: str,
        reasoning: str,
        reset_tree: bool = False,
    ):
        """
        Args:
            from_node (str): The node that is being updated from.
            to_node (str): The node that is being updated to.
            reasoning (str): The reasoning for the update.
            last_in_branch (bool): Whether this is the last update in the branch (whether the tree is complete after this - hardcoded)
                e.g. in query tool, sometimes the query is the end of the tree and sometimes summarise objects is
        """
        self.from_node = from_node
        self.to_node = to_node
        self.reasoning = reasoning
        self.reset_tree = reset_tree

    async def to_frontend(
        self,
        user_id: str,
        conversation_id: str,
        query_id: str,
        tree_index: int,
    ):
        return {
            "type": "tree_update",
            "id": str(uuid.uuid4()),
            "user_id": user_id,
            "conversation_id": conversation_id,
            "query_id": query_id,
            "payload": {
                "node": self.from_node,
                "tree_index": tree_index,
                "decision": self.to_node,
                "reasoning": self.reasoning,
                "reset": self.reset_tree,
            },
        }


class TrainingUpdate:
    """
    Record a training example for a module.
    Keep track of the inputs and outputs of the module, and the module name.
    """

    def __init__(
        self,
        module_name: str,
        inputs: dict,
        outputs: dict,
        extra_inputs: dict = {},
    ):
        self.module_name = module_name

        # Format datetime in inputs and outputs
        format_dict_to_serialisable(inputs)
        format_dict_to_serialisable(outputs)

        # Check if any Pydantic base models exist in either inputs or outputs
        # Create copies of inputs and outputs to avoid modifying the original dictionaries
        inputs_copy = deepcopy(inputs)
        outputs_copy = deepcopy(outputs)

        # If so, convert them to a dictionary
        for key, value in inputs_copy.items():
            inputs_copy[key] = self._convert_basemodel(value)
        for key, value in outputs_copy.items():
            outputs_copy[key] = self._convert_basemodel(value)

        self.inputs = {**inputs_copy, **extra_inputs}
        self.outputs = outputs_copy

    def _convert_basemodel(self, value: Any):
        if isinstance(value, BaseModel):
            return value.model_dump()
        elif isinstance(value, dict):
            return {k: self._convert_basemodel(v) for k, v in value.items()}
        elif isinstance(value, list):
            return [self._convert_basemodel(v) for v in value]
        else:
            return value

    def to_json(self):
        return {
            "module_name": self.module_name,
            "inputs": self.inputs,
            "outputs": self.outputs,
        }


class FewShotExamples(Update):
    """
    A set of few shot examples for a module.
    """

    def __init__(self, uuids: list[str]):
        self.uuids = uuids
        Update.__init__(
            self,
            "fewshot_examples",
            {
                "uuids": self.uuids,
            },
        )
