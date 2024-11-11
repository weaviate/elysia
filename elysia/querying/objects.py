import datetime
import json

from elysia.util.parsing import format_datetime
from elysia.tree.objects import Objects
from weaviate.classes.query import Filter, Sort

from elysia.globals.weaviate_client import client

class Retrieval(Objects):
    def __init__(self, output, metadata):
        super().__init__(output, metadata)

class GenericRetrieval(Retrieval):
    def __init__(self, response, metadata):
        if response is None:
            output = []
        else:
            output = [{k: v for k, v in obj.properties.items()} for obj in response.objects]
        super().__init__(output, metadata)
        self.type = "generic"

    def to_json(self):
        for object in self.objects:
            for key, value in object.items():
                if isinstance(value, datetime.datetime):
                    object[key] = format_datetime(value)
        return super().to_json()
    
class MessageRetrieval(GenericRetrieval):
    def __init__(self, response, metadata):
        super().__init__(response, metadata)
        self.type = "message"

class TicketRetrieval(GenericRetrieval):
    def __init__(self, response, metadata):
        super().__init__(response, metadata)
        self.type = "ticket"

class ConversationRetrieval(Retrieval):
    def __init__(self, response, metadata):
        if response is None:
            output = []
        else:
            output = self._return_all_messages_in_conversation(response, metadata)
        super().__init__(output, metadata)
        self.type = "conversation"

    def _fetch_items_in_conversation(self, conversation_id: str, metadata: dict):
        """
        Use Weaviate to fetch all messages in a conversation based on the conversation ID.
        """
        collection = client.collections.get(metadata["collection_name"])
        items_in_conversation = collection.query.fetch_objects(
            filters=Filter.by_property("conversation_id").equal(conversation_id)
        )
        items_in_conversation = [obj for obj in items_in_conversation.objects]

        return items_in_conversation

    def _return_all_messages_in_conversation(self, response, metadata: dict):
        """
        Return all messages in a conversation based on the response from Weaviate.
        """

        returned_objects = [None] * len(response.objects)
        for i, o in enumerate(response.objects):
            items_in_conversation = self._fetch_items_in_conversation(o.properties["conversation_id"], metadata)
            to_return = [{
                k: v for k, v in item.properties.items()
            } for item in items_in_conversation]
            to_return.sort(key = lambda x: int(x["message_index"]))
            
            for item in to_return:
                if item["message_index"] == o.properties["message_index"]:
                    item["relevant"] = True
                else:
                    item["relevant"] = False
                
            returned_objects[i] = to_return

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
    
