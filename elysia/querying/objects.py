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

    def to_json(self, objects: list[dict] = None, idx: int = None):

        if objects is None:
            objects = self.objects

        if idx is not None:
            objects = [objects[idx]]

        for object in objects:
            for key, value in object.items():
                
                if isinstance(value, datetime.datetime):
                    object[key] = format_datetime(value)

                elif (
                    not isinstance(value, str) and 
                    not isinstance(value, list) and 
                    not isinstance(value, dict) and 
                    not isinstance(value, float) and 
                    not isinstance(value, int) and 
                    not isinstance(value, bool)
                ):
                    object[key] = str(value)
                
                if isinstance(object[key], str) and object[key].startswith("[") and object[key].endswith("]"):
                    object[key] = eval(object[key], {}, {})

        return super().to_json(objects=objects)
    
class EpicGenericRetrieval(Retrieval):
    def __init__(self, objects: list[dict], metadata: dict):
        super().__init__(objects, metadata)
        self.type = "epic_generic"

class BoringGenericRetrieval(Retrieval):
    def __init__(self, objects: list[dict], metadata: dict):
        super().__init__(objects, metadata)
        self.type = "boring_generic"

class EcommerceRetrieval(Retrieval):
    def __init__(self, objects: list[dict], metadata: dict):
        super().__init__(objects, metadata)
        self.type = "ecommerce"

class TicketRetrieval(Retrieval):
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

    def _map_objects(self, objects: list[dict], mapping: dict):
        new_objects = []
        for object in objects:
            new_object = {key: "" for key in mapping.values()}
            for key, value in object.items():
                if key in mapping.keys():
                    new_object[mapping[key]] = value
                elif key == "uuid":
                    new_object["uuid"] = value
                elif key == "summary":
                    new_object["summary"] = value
            
            new_object["relevant"] = False
            new_objects.append(new_object)
        return new_objects

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

    def to_json(self, objects: list[dict] = None):
        if objects is None:
            objects = self.objects

        for conversation in objects:
            for message in conversation:
                for key, value in message.items():
                    if isinstance(value, datetime.datetime):
                        message[key] = format_datetime(value)
                        
                    elif (
                        not isinstance(value, str) and 
                        not isinstance(value, list) and 
                        not isinstance(value, dict) and 
                        not isinstance(value, float) and 
                        not isinstance(value, int) and 
                        not isinstance(value, bool)
                    ):
                        message[key] = str(value)
                
                    if isinstance(message[key], str) and message[key].startswith("[") and message[key].endswith("]"):
                        message[key] = eval(message[key])
        return {
            "metadata": self.metadata,
            "objects": objects
        }

    def _map_objects(self, objects: list[dict], mapping: dict):
        new_objects = []
        for conversation in objects:
            new_conversation = []
            for message in conversation:
                new_message = {key: "" for key in mapping.values()}
                for key, value in message.items():
                    if key in mapping.keys():
                        new_message[mapping[key]] = value
                    elif key == "uuid":
                        new_message["uuid"] = value
                    elif key == "summary":
                        new_message["summary"] = value
                
                new_message["relevant"] = message["relevant"]
                new_conversation.append(new_message)
            new_objects.append(new_conversation)
        return new_objects
    
# class MappedRetrieval(Retrieval):
#     def __init__(self, objects: list[dict], metadata: dict):
#         super().__init__(objects, metadata)
#         self.type = "mapped"