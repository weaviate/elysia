import datetime

import dspy
from weaviate.classes.query import Filter, Sort

from elysia.globals.weaviate_client import client
from elysia.globals.reference import reference
from elysia.util.logging import backend_print
from elysia.querying.prompt_executors import QueryCreatorExecutor
from elysia.tree.objects import Returns, GenericRetrieval, ConversationRetrieval, TicketRetrieval

def format_datetime(dt: datetime.datetime) -> str:
    dt = dt.isoformat("T")
    return dt[:dt.find("+")] + "Z"

class AgenticQuery:

    def __init__(self, collection_name: str, filepath: str = "elysia/training/dspy_models/agentic_query/fewshot_k12.json"):
        self.collection_name = collection_name
        self.collection = client.collections.get(collection_name)
        self.query_creator = QueryCreatorExecutor().activate_assertions()
        if len(filepath) > 0:
            self.query_creator.load(filepath)

    def _find_previous_queries(self, available_information: Returns):

        previous_queries = []
        if self.collection_name in available_information.retrieved:
            metadata = available_information.retrieved[self.collection_name].metadata
            if "previous_queries" in metadata:
                previous_queries.extend(metadata["previous_queries"])

        return previous_queries

    def _obtain_code(
        self, 
        user_prompt: str, 
        previous_queries: list
    ) -> str:

        example_field = self.collection.query.fetch_objects(limit=1).objects[0].properties
        for key in example_field:
            if isinstance(example_field[key], datetime.datetime):
                example_field[key] = example_field[key].isoformat()

        data_fields = list(example_field.keys())

        query_code = self.query_creator(
            user_prompt=user_prompt,
            reference=reference,
            data_fields=data_fields,
            example_field=example_field,
            previous_queries=previous_queries
        )


        return query_code

    def _execute_code(self, query_code: str) -> dict:
        if query_code.startswith("```python") and query_code.endswith("```"):
            query_code = query_code[8:-3]
        elif query_code.startswith("```") and query_code.endswith("```"):
            query_code = query_code[3:-3]
        try:
            return eval("self." + query_code)
        except Exception as e:
            backend_print(f"Error executing query code: {e}, returning 0 objects")
            return None

    def query(self, user_prompt: str, available_information: Returns):

        previous_queries = self._find_previous_queries(available_information)

        query_code = self._obtain_code(user_prompt, previous_queries)
        response = self._execute_code(query_code)

        previous_queries.append(query_code)

        metadata = {"previous_queries": previous_queries, "collection_name": self.collection_name}

        return response, metadata


class MessageQuery(AgenticQuery):
    
    """
    Applicable to slack conversations and emails.
    """

    def __init__(self, return_conversation: bool = False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.return_conversation = return_conversation
        
    def fetch_items_in_conversation(self, conversation_id: str):
        """
        Use Weaviate to fetch all messages in a conversation based on the conversation ID.
        """

        items_in_conversation = self.collection.query.fetch_objects(
            filters=Filter.by_property("conversation_id").equal(conversation_id)
        )
        items_in_conversation = [obj for obj in items_in_conversation.objects]

        return items_in_conversation

    def return_all_messages_in_conversation(self, response):
        """
        Return all messages in a conversation based on the response from Weaviate.
        """

        returned_objects = [None] * len(response.objects)
        for i, o in enumerate(response.objects):
            items_in_conversation = self.fetch_items_in_conversation(o.properties["conversation_id"])
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

    def __call__(self, user_prompt: str, available_information: Returns, limit: int = 10, type: str = "hybrid", rewrite_query: bool = True, **kwargs):
        response, metadata = self.query(user_prompt, available_information)

        if self.return_conversation:
            output = self.return_all_messages_in_conversation(response)
        else:
            if response is None:
                output = []
            else:
                output = [{k: v for k, v in obj.properties.items()} for obj in response.objects]

        return ConversationRetrieval(output, metadata, self.return_conversation)

class GenericQuery(AgenticQuery):
    def __call__(self, user_prompt: str, available_information: Returns, limit: int = 10, type: str = "hybrid", rewrite_query: bool = True, **kwargs):
        response, metadata = self.query(user_prompt, available_information)

        if response is None:
            output = []
        else:
            output = [{k: v for k, v in obj.properties.items()} for obj in response.objects]
        return GenericRetrieval(output, metadata)

class TicketQuery(GenericQuery):
    """
    Applicable to github issues.
    """
    def __call__(self, user_prompt: str, available_information: Returns, limit: int = 10, type: str = "hybrid", rewrite_query: bool = True, **kwargs):
        response, metadata = self.query(user_prompt, available_information)

        if response is None:
            output = []
        else:
            output = [{k: v for k, v in obj.properties.items()} for obj in response.objects]
        return TicketRetrieval(output, metadata)



QueryOptions = {
    "message": MessageQuery,
    "ticket": TicketQuery,
    "generic": GenericQuery
}