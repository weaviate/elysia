from weaviate.classes.query import Filter
from backend.globals.weaviate_client import client

from backend.globals.reference import reference

from backend.util.logging import backend_print
from backend.querying.prompt_executors import QueryRewriterExecutor
from backend.tree.objects import Returns, GenericRetrieval, ConversationRetrieval

class Query:

    def __init__(self, collection_name: str):
        self.collection_name = collection_name
        self.collection = client.collections.get(collection_name)

    def _find_previous_queries(self, available_information: Returns):

        previous_queries = []
        if self.collection_name in available_information.retrieved:
            metadata = available_information.retrieved[self.collection_name].metadata
            if "previous_queries" in metadata:
                previous_queries.extend(metadata["previous_queries"])

        return previous_queries
    
    def query(self, user_prompt: str, available_information: Returns, limit: int = 10, type: str = "hybrid", rewrite_query: bool = True, **kwargs):

        previous_queries = self._find_previous_queries(available_information)

        if rewrite_query:
            query_rewriter = QueryRewriterExecutor()
            query = query_rewriter(user_prompt=user_prompt, previous_queries=previous_queries)
            previous_queries.append(query)
        else:
            query = user_prompt

        metadata = {"previous_queries": previous_queries, "collection_name": self.collection_name}

        if type == "hybrid":
            output = self.hybrid(query, limit)
        elif type == "semantic":
            output = self.near_text(query, limit)
        else:
            raise ValueError(f"Invalid query type: {type}")
        
        return output, metadata

    def __call__(self, user_prompt: str, available_information: Returns, limit: int = 10, type: str = "hybrid", rewrite_query: bool = True, **kwargs):
        return self.query(user_prompt, available_information, limit, type, rewrite_query, **kwargs)
    
class MessageQuery(Query):
    """
    Applicable to slack conversations and emails.
    """

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
            returned_objects[i] = (to_return, o.properties["message_index"])

        return returned_objects

    def near_text(self, query: str, limit: int = 10):

        # backend_print(f"Querying collection ([italic magenta]semantic[/italic magenta]): {self.collection.name} with query: {query}")
        response = self.collection.query.near_text(
            query = query,
            limit = limit
        )

        return self.return_all_messages_in_conversation(response)

    def hybrid(self, query: str, limit: int = 10):

        # backend_print(f"Querying collection ([italic magenta]hybrid[/italic magenta]): {self.collection.name} with query: {query}")
        response = self.collection.query.hybrid(
            query = query,
            limit = limit
        )

        return self.return_all_messages_in_conversation(response)

    def __call__(self, user_prompt: str, available_information: Returns, limit: int = 10, type: str = "hybrid", rewrite_query: bool = True, **kwargs):
        output, metadata = self.query(user_prompt, available_information, limit, type, rewrite_query, **kwargs)
        return ConversationRetrieval(output, metadata)

class IssueQuery(Query):
    """
    Applicable to github issues.
    """
    def near_text(self, query: str, limit: int = 10):

        # backend_print(f"Querying collection ([italic magenta]semantic[/italic magenta]): {self.collection.name} with query: {query}")
        response = self.collection.query.near_text(
            query = query,
            limit = limit
        )

        return [{k: v for k, v in obj.properties.items()} for obj in response.objects]
    
    def hybrid(self, query: str, limit: int = 10):

        # backend_print(f"Querying collection ([italic magenta]hybrid[/italic magenta]): {self.collection.name} with query: {query}")
        response = self.collection.query.hybrid(
            query = query,
            limit = limit
        )

        return [{k: v for k, v in obj.properties.items()} for obj in response.objects]
    
    def __call__(self, user_prompt: str, available_information: Returns, limit: int = 10, type: str = "hybrid", rewrite_query: bool = True, **kwargs):
        output, metadata = self.query(user_prompt, available_information, limit, type, rewrite_query, **kwargs)
        return GenericRetrieval(output, metadata)

QueryOptions = {
    "message": MessageQuery,
    "issue": IssueQuery
}