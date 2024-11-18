import json
import uuid
import datetime
from rich import print

from elysia.util.parsing import objects_dict_to_str, format_datetime, remove_whitespace
from elysia.text.objects import Text, Response

class Branch:
    def __init__(self, updates: dict):
        self.updates = updates

class Status:
    def __init__(self, status: str):
        self.status = status

    def to_json(self, conversation_id: str):
        return {
            "type": "status",
            "conversation_id": conversation_id,
            "id": "sta-" + str(uuid.uuid4()),
            "payload": {
                "text": self.status
            }
        }

class Objects:
    def __init__(self, objects: list[dict | str], metadata: dict = {}):
        self.objects = objects
        self.metadata = metadata

    def add(self, objects: list[dict], metadata: dict = {}):
        self.objects.extend(objects)
        for key, value in metadata.items():

            if key not in self.metadata:
                self.metadata[key] = value
            
            if isinstance(self.metadata[key], list):
                self.metadata[key].extend(value)
            else:
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

class Warning(Objects):
    def __init__(self, objects: list[str], metadata: dict = {}):
        super().__init__(objects, metadata)
        self.type = "warning"

class Error(Objects):
    def __init__(self, objects: list[str], metadata: dict = {}):
        super().__init__(objects, metadata)
        self.type = "error"

class SelfInfo(Objects):
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
    def __init__(self, retrieved: dict, text: Text):
        self.self_info = SelfInfo()
        self.retrieved = retrieved
        self.text = text
    
    def add_retrieval(self, collection_name: str, objects: Objects):
        if collection_name not in self.retrieved:
            self.retrieved[collection_name] = objects
        else:
            self.retrieved[collection_name].add(objects.objects, objects.metadata)

    def add_text(self, objects: Objects):
        self.text.add(objects.objects)

    def to_json(self):
        return {
            "retrieval": {collection_name: self.retrieved[collection_name].to_json() for collection_name in self.retrieved},
            "text": self.text.to_json(),
            "self_info": self.self_info.to_json()
        }

    def to_str(self):
        return json.dumps({
            "retrieval": {collection_name: self.retrieved[collection_name].to_str() for collection_name in self.retrieved},
            "text": self.text.to_str(),
            "self_info": self.self_info.to_str()
        })
    
    def to_llm_str(self):
        return json.dumps({
            "retrieval": {collection_name: self.retrieved[collection_name].to_llm_str() for collection_name in self.retrieved},
            "text": self.text.to_llm_str(),
            "self_info": self.self_info.to_llm_str()
        })
    
    @classmethod
    def from_json(cls, json_dict: dict):
        retrieved = {}
        if "retrieval" in json_dict:
            for collection_name in json_dict["retrieval"]:
                retrieved[collection_name] = Objects(json_dict["retrieval"][collection_name]["objects"], json_dict["retrieval"][collection_name]["metadata"])

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
