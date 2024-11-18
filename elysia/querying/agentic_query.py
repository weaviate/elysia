import datetime
import dspy

from weaviate.classes.aggregate import GroupByAggregate

from elysia.globals.weaviate_client import client
from elysia.globals.reference import reference
from elysia.util.logging import backend_print
from elysia.querying.prompt_executors import (
    QueryExecutor, 
    QueryInitialiserExecutor, 
    PropertyGroupingExecutor, 
    ObjectSummaryExecutor
)
from elysia.tree.objects import Returns, Objects, Status, Warning, Error, Branch
from elysia.text.objects import Response, Code
from elysia.querying.objects import GenericRetrieval, MessageRetrieval, ConversationRetrieval, TicketRetrieval

class AgenticQuery:

    def __init__(self, 
                 query_initialiser_filepath: str = "elysia/training/dspy_models/query_initialiser/fewshot_k12.json", 
                 query_creator_filepath: str = "elysia/training/dspy_models/agentic_query/fewshot_k12.json", 
                 collection_names: list[str] = None, 
                 return_types: dict[str, str] = None): 
        
        self.collection_names = collection_names

        self.query_initialiser = QueryInitialiserExecutor(self.collection_names, return_types).activate_assertions()        
        self.property_grouper = PropertyGroupingExecutor()
        self.querier = QueryExecutor().activate_assertions(max_backtracks=3)
        # self.aggregator = AggregateCollectionExecutor().activate_assertions(max_backtracks=3)
        self.object_summariser = ObjectSummaryExecutor().activate_assertions()
        

        if len(query_creator_filepath) > 0:
            self.querier.load(query_creator_filepath)
            backend_print(f"[green]Loaded querier[/green] model at [italic magenta]{query_creator_filepath}[/italic magenta]")

        if len(query_initialiser_filepath) > 0:
            self.query_initialiser.load(query_initialiser_filepath)
            backend_print(f"[green]Loaded query initialiser[/green] model at [italic magenta]{query_initialiser_filepath}[/italic magenta]")

    def set_collection_names(self, collection_names: list[str]):
        self.collection_names = collection_names
        self.query_initialiser.available_collections = collection_names

    def _find_previous_queries(self, collection_name: str, available_information: Returns):

        self.previous_queries = []
        if collection_name in available_information.retrieved:
            metadata = available_information.retrieved[collection_name].metadata
            if "previous_queries" in metadata:
                self.previous_queries.extend(metadata["previous_queries"])

    def _initialise_query(self, user_prompt: str, previous_reasoning: dict, data_queried: list[str], current_message: str):

        # run initialiser to get collection name and return type
        initialiser = self.query_initialiser(
            user_prompt=user_prompt,
            reference=reference,
            previous_reasoning=previous_reasoning,
            data_queried=data_queried,
            current_message=current_message
        )

        return initialiser

    def _aggregate(self, collection_name: str, property_name: str):
        collection = client.collections.get(collection_name)
        aggregation = collection.aggregate.over_all(
            total_count=True,
            group_by=GroupByAggregate(prop=property_name)
        )

        out = {}
        for result in aggregation.groups:
            if result.grouped_by.prop in out:
                out[result.grouped_by.prop][result.grouped_by.value] = result.total_count
            else:
                out[result.grouped_by.prop] = {result.grouped_by.value: result.total_count}

        return out
        
    async def __call__(self, user_prompt: str, available_information: Returns, previous_reasoning: dict, **kwargs):
        
        data_queried = kwargs.get("data_queried", [])
        current_message = kwargs.get("current_message", "")

        # -- Step 1: Determine collection and other fields
        Branch({
            "name": "Query Initialiser",
            "description": "Determine the collection and return type to query.",
            "returns": "collection_name, return_type, output_type"
        })
        initialiser = self._initialise_query(user_prompt, previous_reasoning, data_queried, current_message)

        reasoning = initialiser.reasoning
        collection_name = initialiser.collection_name
        return_type = initialiser.return_type
        output_type = initialiser.output_type

        if initialiser.text_return is not None:
            yield Response([{"text": initialiser.text_return}], {})

        self._find_previous_queries(collection_name, available_information)

        # add reasoning from initialiser to previous reasoning just for query step
        previous_reasoning["query_initialiser"] = reasoning

        # get example fields from collection
        example_field = client.collections.get(collection_name).query.fetch_objects(limit=1).objects[0].properties
        for key in example_field:
            if isinstance(example_field[key], datetime.datetime):
                example_field[key] = example_field[key].isoformat()

        data_fields = list(example_field.keys())

        # -- Step 2: Determine property to group by and aggregate to get information (TODO: somehow cache this)
        Branch({
            "name": "Property Grouper",
            "description": "Determine the property to group by to get information about the collection.",
            "returns": "property_name"
        })
        property_grouper = self.property_grouper(user_prompt, reference, previous_reasoning, data_fields, example_field, current_message)
        property_name = property_grouper.property_name

        try:
            aggregation = self._aggregate(collection_name, property_name)
            yield Response([{"text": property_grouper.text_return}], {})
            current_message += " " + property_grouper.text_return
        except Exception as e:
            aggregation = {}
            yield Warning([f"Aggregation error: {e}"], {})
            
        # -- Step 3: Query the collection
        Branch({
            "name": "Query Executor",
            "description": "Write code and query the collection to retrieve objects.",
            "returns": "objects, code, text_return, is_query_possible"
        })
        yield Status(f"Querying {collection_name}")
        response, code, text_return, is_query_possible = self.querier(
            user_prompt = user_prompt, 
            reference = reference, 
            previous_queries = self.previous_queries, 
            data_fields = data_fields, 
            example_field = example_field, 
            collection_metadata = aggregation,
            previous_reasoning = previous_reasoning,
            collection_name = collection_name,
            current_message = current_message
        )
        yield Response([{"text": text_return}], {})
        current_message += " " + text_return
        yield Code([{"text": code, "language": "python", "title": "Query"}], {})

        if is_query_possible:

            yield Status(f"Retrieved {len(response.objects)} objects from {collection_name}")

            if code is not None:
                self.previous_queries.append(code)

            objects = []
            for obj in response.objects:
                objects.append({k: v for k, v in obj.properties.items()})
                objects[-1]["uuid"] = obj.uuid.hex

            if output_type == "summary":
                yield Status(f"Generating summaries of the retrieved objects")
                object_summaries = self.object_summariser(objects)
                yield Status(f"Summarised {len(object_summaries)} objects")

                # attach summaries to objects
                for i, obj in enumerate(objects):
                    if i < len(object_summaries):
                        obj["summary"] = object_summaries[i]
                    else:
                        obj["summary"] = ""

            else:
                for obj in objects:
                    obj["summary"] = ""

            metadata = {
                "previous_queries": self.previous_queries, 
                "collection_name": collection_name,
                "collection_metadata": aggregation
            }

            if return_type == "conversation":
                yield ConversationRetrieval(objects, metadata)
            elif return_type == "message":
                yield MessageRetrieval(objects, metadata)
            elif return_type == "ticket":
                yield TicketRetrieval(objects, metadata)
            else:
                yield GenericRetrieval(objects, metadata)
        
        else:
            yield GenericRetrieval([], {"collection_name": collection_name, "impossible_prompts": [user_prompt]})
