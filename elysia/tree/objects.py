import json
import datetime
from rich import print
from elysia.util.parsing import objects_dict_to_str, format_datetime

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


class GenericRetrieval(Objects):
    def __init__(self, objects: list[dict], metadata: dict = {}):
        super().__init__(objects, metadata)
        self.type = "retrieval"

    def to_json(self):
        for object in self.objects:
            for key, value in object.items():
                if isinstance(value, datetime.datetime):
                    object[key] = format_datetime(value)
        return super().to_json()

class ConversationRetrieval(GenericRetrieval):
    def __init__(self, objects: list[dict], metadata: dict = {}, return_conversation: bool = False):
        super().__init__(objects, metadata)
        self.type = "conversation"
        self.return_conversation = return_conversation   

    def to_json(self):
        if self.return_conversation:
            for conversation in self.objects:
                for message in conversation:
                    for key, value in message.items():
                        if isinstance(value, datetime.datetime):
                            message[key] = format_datetime(value)
            return {
                "metadata": self.metadata,
                "objects": self.objects
            }
        else:
            return super().to_json()

        

    def to_str(self):
        return json.dumps(self.objects)
    
    def to_llm_str(self):
        if self.return_conversation:
            out = "{'objects': \n"
            for i, conversation in enumerate(self.objects):
                out += "{'conversation_" + str(i+1) + "': "
                out += json.dumps(conversation) + "}, \n"
            
            out += "'metadata': " + json.dumps(self.metadata) + "\n"
            out += "}"
        else:
            out = super().to_llm_str()
        
        return out
        
    def __repr__(self):
        if self.return_conversation:
            for i, conversation in enumerate(self.objects):
                print(f"[bold green]Conversation {i+1}[/bold green]")
                for j, message in enumerate(conversation):
                    if message["relevant"]:
                        print(f"[bold green]Message {j+1}[/bold green]:", end="")
                        print(f"[italic green]{message}[/italic green]")
                    else:
                        print(f"[bold]Message {j+1}[/bold]:", end="")
                        print(f"[italic indigo]{message}[/italic indigo]")
                    print("\n")
                print("-"*100)
                print("\n\n")
        else:
            super().__repr__()
        return ""
    
class TicketRetrieval(GenericRetrieval):
    def __init__(self, objects: list[dict], metadata: dict = {}):
        super().__init__(objects, metadata)
        self.type = "ticket"


class Text(Objects):
    def __init__(self, objects: list[str], metadata: dict = {}):
        super().__init__(objects, metadata)
        self.type = "text"

class Returns:
    def __init__(self, retrieved: dict, text: Text):
        self.retrieved = retrieved
        self.text = text
    
    def add_retrieval(self, collection_name: str, objects: Objects):
        if collection_name not in self.retrieved:
            self.retrieved[collection_name] = objects
        else:
            self.retrieved[collection_name].add(objects.objects, objects.metadata)

    def add_text(self, objects: Text):
        self.text.add(objects.objects)

    def to_json(self):
        return {
            "retrieval": {collection_name: self.retrieved[collection_name].to_json() for collection_name in self.retrieved},
            "text": self.text.to_json()
        }

    def to_str(self):
        return json.dumps({
            "retrieval": {collection_name: self.retrieved[collection_name].to_str() for collection_name in self.retrieved},
            "text": self.text.to_str()
        })
    
    def to_llm_str(self):
        return json.dumps({
            "retrieval": {collection_name: self.retrieved[collection_name].to_llm_str() for collection_name in self.retrieved},
            "text": self.text.to_llm_str()
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

        return cls(retrieved=retrieved, text=Text(text))

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
