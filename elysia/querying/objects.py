import datetime
import json

from rich import print
from weaviate.classes.query import Filter, Sort

from elysia.util.parsing import format_datetime
from elysia.tree.objects import Objects
from elysia.globals.weaviate_client import client

class Retrieval(Objects):
    def __init__(self, objects: list[dict], metadata: dict):
        
        if objects is None:
            objects = []

        super().__init__(objects, metadata)

    def to_json(self):
        for object in self.objects:
            for key, value in object.items():
                if isinstance(value, datetime.datetime):
                    object[key] = format_datetime(value)
        return super().to_json()
    
class GenericRetrieval(Retrieval):
    def __init__(self, objects: list[dict], metadata: dict):
        super().__init__(objects, metadata)
        self.type = "generic"

class TicketRetrieval(GenericRetrieval):
    def __init__(self, objects: list[dict], metadata: dict):
        super().__init__(objects, metadata)
        self.type = "ticket"

class MessageRetrieval(Retrieval):
    def __init__(self, objects: list[dict], metadata: dict):
        
        if objects is None:
            objects = []
        else:
            for obj in objects:
                obj["relevant"] = False

        super().__init__(objects, metadata)
        self.type = "message"

class ConversationRetrieval(Retrieval):
    def __init__(self, objects: list[dict], metadata: dict):
        if objects is None:
            objects = []
        else:
            objects = self._return_all_messages_in_conversation(objects, metadata)
        super().__init__(objects, metadata)
        self.type = "conversation"

    def _fetch_items_in_conversation(self, conversation_id: str, metadata: dict, message_index: int):
        """
        Use Weaviate to fetch all messages in a conversation based on the conversation ID.
        """
        collection = client.collections.get(metadata["collection_name"])
        items_in_conversation = collection.query.fetch_objects(
            filters=Filter.by_property("conversation_id").equal(conversation_id)
        )

        output = []
        for obj in items_in_conversation.objects:
            output.append({k: v for k, v in obj.properties.items()})
            output[-1]["uuid"] = obj.uuid.hex
            if output[-1]["message_index"] == message_index:
                output[-1]["relevant"] = True
            else:
                output[-1]["relevant"] = False

        return output

    def _return_all_messages_in_conversation(self, objects: list[dict], metadata: dict):
        """
        Return all messages in a conversation based on the response from Weaviate.
        """

        returned_objects = [None] * len(objects)
        for i, o in enumerate(objects):
            items_in_conversation = self._fetch_items_in_conversation(o["conversation_id"], metadata, o["message_index"])
            items_in_conversation.sort(key = lambda x: int(x["message_index"]))
            returned_objects[i] = items_in_conversation

        return returned_objects

    def to_json(self):
        for conversation in self.objects:
            for message in conversation:
                for key, value in message.items():
                    if isinstance(value, datetime.datetime):
                        message[key] = format_datetime(value)
        return {
            "metadata": self.metadata,
            "objects": self.objects
        }

    def to_str(self):
        return json.dumps(self.objects)
    
    def to_llm_str(self):
        out = "{'objects': \n"
        for i, conversation in enumerate(self.objects):
            out += "{'conversation_" + str(i+1) + "': "
            out += json.dumps(conversation) + "}, \n"
        
        out += "'metadata': " + json.dumps(self.metadata) + "\n"
        out += "}"
        
        return out
        
    def __repr__(self):
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
        return ""
    
