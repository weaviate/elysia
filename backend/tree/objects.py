import json

from rich import print
from backend.util.parsing import objects_dict_to_str

class Objects:
    def __init__(self, objects: list[dict | str], metadata: dict = {}):
        self.objects = objects
        self.metadata = metadata

    def add(self, objects: list[dict], metadata: dict = {}):
        self.objects.extend(objects)
        for key, value in metadata.items():
            if key not in self.metadata:
                self.metadata[key] = []
            self.metadata[key].extend(value)

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

class ConversationRetrieval(GenericRetrieval):
    def __init__(self, objects: list[dict], metadata: dict = {}):
        super().__init__(objects, metadata)
        self.type = "conversation"

    def return_value(self, idx: int):
        return self.objects[idx][0], self.objects[idx][1]
    

    def to_str(self):
        return json.dumps(self.objects)
    
    def to_llm_str(self):
        out = "{'objects': \n"
        for i, (conversation, retrieved_idx) in enumerate(self.objects):
            out += "{'conversation_" + str(i+1) + "': "
            inner_conversation = conversation.copy()
            inner_conversation[int(retrieved_idx)]["relevant_message"] = True
            out += json.dumps(inner_conversation) + "}, \n"
        
        out += "'metadata': " + json.dumps(self.metadata) + "\n"
        out += "}"
        
        return out
        
    def __repr__(self):
        for i, (conversation, retrieved_idx) in enumerate(self.objects):
            print(f"[bold green]Conversation {i+1}[/bold green]")
            for j, message in enumerate(conversation):
                if j == retrieved_idx:
                    print(f"[bold green]Message {j+1}[/bold green]:", end="")
                    print(f"[italic green]{message}[/italic green]")
                else:
                    print(f"[bold]Message {j+1}[/bold]:", end="")
                    print(f"[italic indigo]{message}[/italic indigo]")
                print("\n")
            print("-"*100)
            print("\n\n")
        return ""
    



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
