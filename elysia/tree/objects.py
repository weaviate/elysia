import json
from rich import print
from typing import Any

# Util
from elysia.util.parsing import objects_dict_to_str, format_datetime, remove_whitespace

# Objects
from elysia.text.objects import Text
from elysia.api.objects import Update

# -- Retrieval Objects --
class Objects:
    """
    Store of returned objects from a query/aggregate/any displayed results.
    Has associated objects and metadata.
    """

    def __init__(self, objects: list[dict | str], metadata: dict = {}):
        self.objects = self._remove_duplicates(objects)
        self.metadata = metadata

    def _remove_duplicates(self, objects: list[dict | str]):
        unique_objects = []
        seen = set()
        
        for obj in objects:
            if isinstance(obj, dict):
                # Convert dict to a string representation for comparison
                obj_str = str(sorted(obj.items()))
            else:
                obj_str = str(obj)
                
            if obj_str not in seen:
                seen.add(obj_str)
                unique_objects.append(obj)
                
        return unique_objects

    def add(self, objects: list[dict], metadata: dict = {}):
        objects = self._remove_duplicates(objects)
        self.objects.extend(objects)
        for key, value in metadata.items():

            if isinstance(self.metadata[key], list):
                # metadata = list means append new entries
                self.metadata[key].extend(value)
            else:
                # metadata = not list means overwrite
                self.metadata[key] = value

    def to_json(self):
        return {
            "metadata": self.metadata,
            "objects": self.objects
        }
    
    def to_str(self):
        return json.dumps({
            "metadata": self.metadata,
            "objects": objects_dict_to_str(self.objects)
        })
    
    def to_llm_str(self):
        return self.to_str()
    
    def return_value(self, idx: int):
        return self.objects[idx]

    def to_frontend(self, conversation_id: str, query_id: str = None):
        return Update.to_frontend_json(
            "result",
            conversation_id,
            query_id,
            {
                "type": self.type,
                "code": self.metadata.get("last_code", {}),
                **self.to_json()
            }
        )

class SelfInfo(Objects):
    """
    Mimick the returns object with information about the Elysia app itself.
    So when the user asks about Elysia, the assistant can provide information about itself.
    """
    def __init__(self, name: str = "Elysia"):
        objects = [
            {
                "field": "name",
                "value": name,
                "description": "The name of the assistant."
            },
            {
                "field": "description",
                "value": "A helpful and friendly assistant that can answer questions, query from Weaviate collections, and provide summaries and textual responses.",
                "description": "A short description of the assistant."
            },
            {
                "field": "purpose",
                "value": remove_whitespace("""Elysia is an agentic retrieval augmented generation (RAG) service, where users can query from Weaviate collections,
                and the assistant will retrieve the most relevant information and answer the user's question. This includes a variety
                of different ways to query, such as by filtering, sorting, querying multiple collections, and providing summaries
                and textual responses."""),
                "description": "The purpose of the assistant."
            }
        ]
        super().__init__(objects, {})
        self.type = "self_info"

class Returns:
    """
    Store of all objects across different types of queries and responses.
    Essentially the collection of all Objects classes, for retrieval, aggregation, text, and self_info (currently).
    """
    def __init__(self, retrieved: dict, aggregation: dict, text: Text):
        self.self_info = SelfInfo()
        self.retrieved = retrieved
        self.aggregation = aggregation
        self.text = text
    
    def add_retrieval(self, collection_name: str, objects: Objects):
        if collection_name not in self.retrieved:
            self.retrieved[collection_name] = objects
        else:
            self.retrieved[collection_name].add(objects.objects, objects.metadata)

    def add_aggregation(self, collection_name: str, objects: Objects):
        if collection_name not in self.aggregation:
            self.aggregation[collection_name] = objects
        else:
            self.aggregation[collection_name].add(objects.objects, objects.metadata)

    def add_text(self, objects: Objects):
        self.text.add(objects.objects)

    def to_json(self):
        return {
            "retrieval": {collection_name: self.retrieved[collection_name].to_json() for collection_name in self.retrieved},
            "aggregation": {collection_name: self.aggregation[collection_name].to_json() for collection_name in self.aggregation},
            "text": self.text.to_json(),
            "self_info": self.self_info.to_json()
        }

    def to_str(self):
        return json.dumps({
            "retrieval": {collection_name: self.retrieved[collection_name].to_str() for collection_name in self.retrieved},
            "aggregation": {collection_name: self.aggregation[collection_name].to_str() for collection_name in self.aggregation},
            "text": self.text.to_str(),
            "self_info": self.self_info.to_str()
        })
    
    def to_llm_str(self):
        return json.dumps({
            "retrieval": {collection_name: self.retrieved[collection_name].to_llm_str() for collection_name in self.retrieved},
            "aggregation": {collection_name: self.aggregation[collection_name].to_llm_str() for collection_name in self.aggregation},
            "text": self.text.to_llm_str(),
            "self_info": self.self_info.to_llm_str()
        })
    
    @classmethod
    def from_json(cls, json_dict: dict):
        retrieved = {}
        if "retrieval" in json_dict:
            for collection_name in json_dict["retrieval"]:
                retrieved[collection_name] = Objects(json_dict["retrieval"][collection_name]["objects"], json_dict["retrieval"][collection_name]["metadata"])

        aggregation = {}
        if "aggregation" in json_dict:
            for collection_name in json_dict["aggregation"]:
                aggregation[collection_name] = Objects(json_dict["aggregation"][collection_name]["objects"], json_dict["aggregation"][collection_name]["metadata"])

        text = []
        if "text" in json_dict:
            text = json_dict["text"]

        if "self_info" in json_dict:
            self_info = SelfInfo(json_dict["self_info"]["objects"][0]["value"])
        else:
            self_info = SelfInfo()

        return cls(retrieved=retrieved, text=Text(text), self_info=self_info)

    def return_retrieval(self, collection_name: str = "", idx = None):

        if collection_name == "":
            collection_name = list(self.retrieved.keys())[0]
            print(f"[bold yellow]No collection name specified, defaulting to {collection_name}[/bold yellow]")

        if idx is None:
            return self.retrieved[collection_name]
        else:
            return self.retrieved[collection_name].return_value(idx)
        
    def return_aggregation(self, collection_name: str = "", idx = None):
        if collection_name == "":
            collection_name = list(self.aggregation.keys())[0]
            print(f"[bold yellow]No collection name specified, defaulting to {collection_name}[/bold yellow]")

        if idx is None:
            return self.aggregation[collection_name]
        else:
            return self.aggregation[collection_name].return_value(idx)
    
    def return_text(self, idx=None):
        if idx is None:
            return self.text
        else:
            return self.text.return_value(idx)
    
    def __repr__(self):
        if self.retrieved != {}:
            print(f"[bold green]Retrieved from collections[/bold green]")
            for collection_name in self.retrieved:
                print(f"- [italic indigo]{collection_name}[/italic indigo]: {len(self.retrieved[collection_name].objects)} objects")
            print("\n")

        if self.text != []:
            for i, text in enumerate(self.text.objects):
                print(f"[bold green]Text {i+1}[/bold green]")
                print("-"*100)
                print(text)
                print("\n\n")
        
        return ""

    def __str__(self):
        return self.__repr__()

# -----------------------


# -- Prompt Input Data Collection Objects --

class PromptData:
    """
    Store of data used by the prompt executor.
    """
    def __init__(self):
        pass

    def to_json(self):
        return {
            k: v for k, v in self.__dict__.items()
        }

    def set_property(self, property: str, value: Any):
        self.__dict__[property] = value

    def update_string(self, property: str, value: str):
        if property not in self.__dict__:
            self.__dict__[property] = ""
        self.__dict__[property] += value

    def update_list(self, property: str, value: Any):
        if property not in self.__dict__:
            self.__dict__[property] = []
        self.__dict__[property].append(value)

    def update_dict(self, property: str, key: str, value: Any):
        if property not in self.__dict__:
            self.__dict__[property] = {}
        self.__dict__[property][key] = value

    def delete_from_dict(self, property: str, key: str):
        if property in self.__dict__ and key in self.__dict__[property]:
            del self.__dict__[property][key]

class TreeData(PromptData):
    """
    Store of data across the tree.
    This includes things like conversation history, actions, decisions, etc.
    These data are given to ALL agents, so every agent is aware of the stage of the decision processes.
    """
    def __init__(
            self,
            user_prompt: str = "",
            previous_reasoning: dict = {},
            conversation_history: list[dict] = [],
            data_queried: dict = {},
            current_message: str = ""
        ):
        self.user_prompt = user_prompt
        self.previous_reasoning = previous_reasoning
        self.conversation_history = conversation_history
        self.data_queried = data_queried
        self.current_message = current_message

    def soft_reset(self):
        self.previous_reasoning = {}
        self.data_queried = {}
        self.current_message = ""

    def data_queried_string(self):
        out = ""
        for collection_name in self.data_queried:
            out += f" - Queried '{collection_name}' with prompt '{self.data_queried[collection_name]['prompt']}' and retrieved {self.data_queried[collection_name]['count']} objects\n"
        return out

class ActionData(PromptData):
    """
    Store of data used by the action agents.
    """
    def __init__(
            self,
            collection_information: dict = {}
        ):
        self.collection_information = collection_information

class DecisionData(PromptData):
    """
    Store of data used by the decision agents.
    """
    def __init__(
            self,
            instruction: str = "",
            available_information: list[dict] = [],
            available_tasks: list[dict] = [],
            num_trees_completed: int = 0,
            recursion_limit: int = 5,
            future_information: list[dict] = []
        ):
        self.instruction = instruction
        self.available_information = available_information
        self.available_tasks = available_tasks
        self.num_trees_completed = num_trees_completed
        self.recursion_limit = recursion_limit
        self.future_information = future_information

    def tree_count_string(self):
        out = f"{self.num_trees_completed+1}/{self.recursion_limit}"
        if self.num_trees_completed > self.recursion_limit:
            out += " (recursion limit reached, write your full chat response accordingly)"
        return out
