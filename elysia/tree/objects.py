import json
from rich import print
from typing import Any

# Util
from elysia.util.parsing import objects_dict_to_str, format_datetime, remove_whitespace

# Objects
from elysia.text.objects import Text
from elysia.api.objects import Update

# -- Objects --  # Retrieval/Aggregation/Anything we used code to get data # 
class Objects:
    """
    Store of returned objects from a query/aggregate/any displayed results.
    Has associated objects and metadata.
    """

    def __init__(self, objects: list[dict | str], metadata: dict = {}, type: str = "default"):
        self.objects = self._remove_duplicates(objects)
        self.metadata = metadata
        self.type = type

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
    
    def _map_objects(self, objects: list[dict], mapping: dict):
        inverted_mapping = {v: k for k, v in mapping.items()}
        new_objects = []
        for object in objects:
            new_object = {key: "" for key in mapping.keys()}
            for key, value in object.items():
                if key in inverted_mapping.keys():
                    new_object[inverted_mapping[key]] = value
                elif key == "uuid":
                    new_object["uuid"] = value
                elif key == "summary":
                    new_object["summary"] = value

            new_objects.append(new_object)
        return new_objects

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

    def to_json(self, objects: list[dict] = None):
        if objects is None:
            objects = self.objects

        return {
            "metadata": self.metadata,
            "objects": objects
        }
    
    def mapped_to_json(self, mapping: dict):
        return {
            "metadata": self.metadata,
            "objects": self.to_json(objects=self._map_objects(self.objects, mapping))["objects"]
        }
    
    def return_value(self, idx: int):
        return self.objects[idx]

    def to_frontend(self, conversation_id: str, query_id: str = None, mapping: dict = None):
        if mapping is None or self.type == "aggregation":
            items = self.to_json()
        else:
            items = self.mapped_to_json(mapping)
            
        return Update.to_frontend_json(
            "result",
            conversation_id,
            query_id,
            {
                "type": self.type,
                "code": self.metadata.get("last_code", {}),
                **items
            }
        )

class SelfInfo(Objects):
    """
    Mimick the returns object with information about the Elysia app itself.
    So when the user asks about Elysia, the assistant can provide information about itself.
    """
    def __init__(self):
        objects = [
            {
                "name": "Elysia",
                "description": "An agentic RAG service in Weaviate.",
                "purpose": remove_whitespace("""Elysia is an agentic retrieval augmented generation (RAG) service, where users can query from Weaviate collections,
                and the assistant will retrieve the most relevant information and answer the user's question. This includes a variety
                of different ways to query, such as by filtering, sorting, querying multiple collections, and providing summaries
                and textual responses.
                                             
                Elysia will dynamically display retrieved objects from the collections in the frontend.
                Elysia works via a tree-based approach, where the user's question is used to generate a tree of potential
                queries to retrieve the most relevant information.
                Each end of the tree connects to a separate agent that will perform a specific task, such as retrieval, aggregation, or generation.
                                             
                The tree itself has decision nodes that determine the next step in the query.
                The decision nodes are decided via a decision-agent, which decides the task.
                                             
                The agents communicate via a series of different prompts, which are stored in the prompt-library.
                The decision-agent prompts are designed to be as general as possible, so that they can be used for a variety of different tasks.
                Some of these variables include conversation history, retrieved objects, the user's original question, train of thought via model reasoning, and more.        
                """),
            }
        ]
        super().__init__(objects, {})
        self.type = "Elysia_Info"

class Returns:
    """
    Store of all objects across different types of queries and responses.
    Essentially the collection of all Objects classes, for retrieval, aggregation, text, and self_info (currently).
    """
    def __init__(self, retrieved: list[Objects], aggregation: list[Objects], text: Text):
        self.self_info = SelfInfo()
        self.retrieved = retrieved
        self.aggregation = aggregation
        self.text = text
    
    def add_retrieval(self, objects: Objects):

        # If the object is already in the retrieved, add to it
        for already_retrieved in self.retrieved:
            if (
                already_retrieved.type == objects.type and 
                already_retrieved.metadata["collection_name"] == objects.metadata["collection_name"]
            ):
                already_retrieved.add(objects.objects, objects.metadata)
                return    
            
        # Otherwise, add the object to the list at the end
        self.retrieved.append(objects)

    def add_aggregation(self, objects: Objects):
        for already_aggregated in self.aggregation:
            if (
                already_aggregated.type == objects.type and 
                already_aggregated.metadata["collection_name"] == objects.metadata["collection_name"]
            ):
                already_aggregated.add(objects.objects, objects.metadata)
                return
        self.aggregation.append(objects)

    def add_text(self, objects: Objects):
        self.text.add(objects.objects)

    def to_json(self):
        return {
            "retrieval": [objects.to_json() for objects in self.retrieved],
            "aggregation": [objects.to_json() for objects in self.aggregation],
            "text": self.text.to_json(),
            "self_info": self.self_info.to_json()
        }

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

        self.data_queried.update({"Elysia": {"prompt": "What is Elysia?", "count": 1, "type": "self_info"}})

    def soft_reset(self):
        self.previous_reasoning = {}
        self.data_queried = {}
        self.current_message = ""

    def data_queried_string(self):
        out = ""
        for i, collection_name in enumerate(self.data_queried):
            if self.data_queried[collection_name]["type"] == "self_info":
                out += f" - [Search {i+1}] Retrieved all information about Elysia, the agentic RAG agent\n"
            elif self.data_queried[collection_name]["type"] == "retrieval":
                if "impossible_prompt" in self.data_queried[collection_name]:
                    out += f" - [Search {i+1}] Attempted to query '{collection_name}' with prompt '{self.data_queried[collection_name]['prompt']}', but it was judged impossible to complete for this prompt/collection combination\n"
                else:
                    out += f" - [Search {i+1}] Queried '{collection_name}' with prompt '{self.data_queried[collection_name]['prompt']}', retrieved {self.data_queried[collection_name]['count']} objects, returned with type '{self.data_queried[collection_name]['return_type']}' and outputted '{'itemised summaries' if self.data_queried[collection_name]['output_type'] == 'summary' else 'original objects'}'\n"
            elif self.data_queried[collection_name]["type"] == "aggregation":
                if "impossible_prompt" in self.data_queried[collection_name]:
                    out += f" - [Search {i+1}] Attempted to aggregate '{collection_name}' with prompt '{self.data_queried[collection_name]['prompt']}', but it was judged impossible to complete for this prompt/collection combination\n"
                else:
                    out += f" - [Search {i+1}] Aggregated '{collection_name}' with prompt '{self.data_queried[collection_name]['prompt']}'\n"
        return out

class ActionData(PromptData):
    """
    Store of data used by the action agents.
    """
    def __init__(
            self,
            collection_information: dict = {},
            collection_return_types: dict = {}
        ):
        self.collection_information = collection_information
        self.collection_return_types = collection_return_types

    def set_collection_names(self, collection_information: dict, collection_names: list[str]):
        self.collection_information = {collection_name: collection_information[collection_name] for collection_name in collection_names}

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
