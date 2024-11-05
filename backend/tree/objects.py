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
    
    def return_value(self, idx: int):
        return self.objects[idx]

class Conversation(Objects):
    def __init__(self, objects: list[dict], metadata: dict = {}):
        super().__init__(objects, metadata)
        self.type = "conversation"

    def return_value(self, idx: int):
        return self.objects[idx][0], self.objects[idx][1]

class Retrieved(Objects):
    def __init__(self, objects: list[dict], metadata: dict = {}):
        super().__init__(objects, metadata)
        self.type = "retrieval"

class Text(Objects):
    def __init__(self, objects: list[str], metadata: dict = {}):
        super().__init__(objects, metadata)
        self.type = "text"

class Returns:
    def __init__(self, retrieved: dict, text: Text):
        self.retrieved = retrieved
        self.text = text
    
    def add_retrieval(self, collection_name: str, objects: list[dict], metadata: dict = {}):
        if collection_name not in self.retrieved:
            self.retrieved[collection_name] = Retrieved(objects=objects, metadata=metadata)
        else:
            self.retrieved[collection_name].add(objects, metadata)

    def add_text(self, objects: list[str], metadata: dict = {}):
        self.text.add(objects, metadata)

    def to_str(self):
        return json.dumps({
            "retrieval": {collection_name: self.retrieved[collection_name].to_str() for collection_name in self.retrieved},
            "text": self.text.to_str()
        })
    
    def return_retrieval(self, collection_name: str):
        return self.retrieved[collection_name]
    
    def return_text(self):
        return self.text
    
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
