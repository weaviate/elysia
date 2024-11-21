import datetime
import dspy

from weaviate.classes.aggregate import GroupByAggregate

from elysia.globals.weaviate_client import client
from elysia.globals.reference import create_reference
from elysia.util.logging import backend_print
from elysia.util.parsing import update_current_message
from elysia.querying.prompt_executors import (
    QueryExecutor, 
    QueryInitialiserExecutor, 
    PropertyGroupingExecutor, 
    ObjectSummaryExecutor
)
from elysia.tree.objects import Returns, Objects, Status, Warning, Error, Branch, TreeUpdate
from elysia.text.objects import Response, Code
from elysia.querying.objects import GenericRetrieval, MessageRetrieval, ConversationRetrieval, TicketRetrieval

class AgenticAggregate:

    def __init__(self, 
                 aggregate_initialiser_filepath: str = "elysia/training/dspy_models/aggregate_initialiser/fewshot_k12.json", 
                 aggregate_executor_filepath: str = "elysia/training/dspy_models/aggregate_executor/fewshot_k12.json",
                 collection_names: list[str] = None):
        
        self.collection_names = collection_names
        