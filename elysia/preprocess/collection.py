# preprocess a collection
# 1. evaluate all the data fields and groups/statistics of the data fields as a whole
# 2. write a summary of the collection via an LLM
# 3. evaluate what return types are available for this collection
# 4. for each data field in the collection, evaluate what corresponding entry goes to what field in the return type
# 5. save as a ELYSIA_METADATA_{collection_name}__ collection

import random
import spacy # for tokenisation

# Weaviate
import weaviate
import weaviate.classes as wvc
from weaviate.classes.init import Auth
from weaviate.classes.aggregate import GroupByAggregate
from weaviate.classes.query import Filter, Metrics, Sort

# Util
from elysia.util.collection_metadata import get_collection_data_types

# Globals
from elysia.globals import client
from elysia.globals import return_types as rt


# Prompt Executors
from elysia.preprocess.prompt_executors import (
    CollectionSummariserExecutor, DataMappingExecutor, ReturnTypeExecutor
)

class ProcessUpdate:

    def __init__(self, collection_name: str):
        self.collection_name = collection_name

    def to_frontend(
        self,
        progress: int,
        total: int,
        error: str = ""
    ):
        return {
            "type": "update" if progress != total else "completed",
            "collection_name": self.collection_name,
            "progress": float(progress) / float(total),
            "error": error
        }
    
    def __call__(self, *args, **kwargs):
        return self.to_frontend(*args, **kwargs)

class CollectionPreprocessor:

    def __init__(self):
        self.collection_summariser_executor = CollectionSummariserExecutor()
        self.data_mapping_executor = DataMappingExecutor()
        self.return_type_executor = ReturnTypeExecutor().activate_assertions()
        self.nlp = spacy.load("en_core_web_sm")
        self.process_update = ProcessUpdate()

    def _summarise_collection(self, collection, properties: dict, summary_sample_size: int = 200):

        # Randomly sample sample_size objects for the summary
        indices = random.sample(range(len(collection)), summary_sample_size)
        subset_objects = []
        for i, item in enumerate(collection.iterator()):
            if i in indices:
                subset_objects.append(item.properties)

        # Summarise the collection
        summary = self.collection_summariser_executor.forward(data=subset_objects, data_fields=list(properties.keys()))
        return summary
    
    def _evaluate_field_statistics(
        self, 
        collection, 
        properties: dict, 
        property: str, 
        full_response = None
    ):

        out = {}
        out["type"] = properties[property]

        # Number (summary statistics)
        if properties[property] == "number":
            response = collection.aggregate.over_all(
                return_metrics=[
                    Metrics(property).number(
                        mean=True,
                        maximum=True,
                        minimum=True
                    )
                ]
            )
            out["range"] = [response.properties[property].minimum, response.properties[property].maximum]
            out["mean"] = response.properties[property].mean
            out["groups"] = []

        # Text (grouping + lengths)
        elif properties[property] == "text":
            response = collection.aggregate.over_all(
                total_count=True,
                group_by=GroupByAggregate(prop=property)
            )
            groups = [
                group.grouped_by.value for group in response.groups
            ]

            if len(groups) < 30:
                out["groups"] = groups
            else:
                out["groups"] = []

            if full_response is not None:

                # For text, we want to evaluate the length of the text in tokens (use spacy)
                lengths = [
                    len(self.nlp(obj.properties[property])) for obj in full_response.objects
                ]

                out["range"] = [
                    min(lengths),
                    max(lengths)
                ]
                out["mean"] = sum(lengths) / len(lengths)

            else:
                out["range"] = []
                out["mean"] = -9999999

        # Boolean (grouping + mean)
        elif properties[property] == "boolean":
            response = collection.aggregate.over_all(
                return_metrics=[
                    Metrics(property).boolean(
                        percentage_true=True
                    )
                ]
            )
            out["groups"] = [True, False]
            out["mean"] = response.properties[property].percentage_true
            out["range"] = [0, 1]

        # Date (summary statistics)
        elif properties[property] == "date":
            response = collection.aggregate.over_all(
                return_metrics=[Metrics(property).date_(
                    median=True,
                    minimum=True,
                    maximum=True
                )]
            )
            out["range"] = [response.properties[property].minimum, response.properties[property].maximum]
            out["mean"] = response.properties[property].median

        # List (lengths)
        elif properties[property].endswith("[]"):
            
            if full_response is not None:
                lengths = [
                    len(obj.properties[property]) for obj in full_response.objects
                ]

                out["range"] = [min(lengths), max(lengths)]
                out["mean"] = sum(lengths) / len(lengths)
                out["groups"] = []

        else:
            out["range"] = []
            out["mean"] = -9999999
            out["groups"] = []

        return out


    def _evaluate_return_types(self, collection_summary: str, data_fields: dict, example_objects: list[dict]):

        return_types = self.return_type_executor(
            collection_summary = collection_summary,
            data_fields = data_fields,
            example_objects = example_objects,
            possible_return_types = rt.specific_return_types
        )

        return return_types
    
    def _define_mappings(self, input_fields: list, output_fields: list, properties: dict, collection_information: dict, example_objects: list[dict]):

        mapping, mapper, error_message = self.data_mapping_executor(
            input_data_fields = input_fields, 
            output_data_fields = output_fields,
            input_data_types = properties,
            collection_information = collection_information,
            example_objects = example_objects
        )

        return mapping, error_message
    
    def __call__(
        self, 
        collection_name: str, 
        manageable_sample_size: int = 1000, 
        summary_sample_size: int = 200,
        force: bool = False
    ):

        if not client.collections.exists(f"ELYSIA_METADATA_{collection_name}__") or force:

            # Start saving the updates
            total = len(rt.specific_return_types) + 1 + 1
            progress = 0
            error = ""

            # Get the collection and its properties
            try:
                collection = client.collections.get(collection_name)
                properties = get_collection_data_types(collection_name)
            except Exception as e:
                yield self.process_update(progress=0, total=total, error=str(e))
                return

            # Summarise the collection using LLM
            try:
                summary = self._summarise_collection(collection, properties, summary_sample_size)
            except Exception as e:
                error = str(e)
                yield self.process_update(progress=1, total=total, error=error)
                return
            
            yield self.process_update(progress=1, total=total)

            # Evaluate if the collection is manageable, if not, we will not fetch all objects
            collection_is_manageable = len(collection) < manageable_sample_size

            # If the collection is manageable, fetch all objects
            if collection_is_manageable:
                full_response = collection.query.fetch_objects(
                    limit=len(collection)
                )

            # Get some example objects
            example_objects = collection.query.fetch_objects(
                limit=3
            )
            example_objects = [obj.properties for obj in example_objects.objects]

            # Initialise the output
            out = {
                "name": collection_name,
                "length": len(collection),
                "summary": summary,
                "fields": {},
                "mappings": {}
            }

            try:
                # Evaluate the summary statistics of each field
                for property in properties:
                    out["fields"][property] = self._evaluate_field_statistics(collection, properties, property, full_response)
            except Exception as e:
                yield self.process_update(progress=1, total=total, error=str(e))
                return

            # Evaluate the return types
            return_types = self._evaluate_return_types(summary, properties, example_objects)

            total = len(return_types) + 1 + 1
            yield self.process_update(progress=2, total=total)

            # Define the mappings
            for return_type in return_types:
                fields = rt.types_dict[return_type]

                mapping, error_message = self._define_mappings(
                    input_fields=list(fields.keys()), 
                    output_fields=list(properties.keys()), 
                    properties=properties, 
                    collection_information=out, 
                    example_objects=example_objects
                )

                if error_message != "":
                    yield self.process_update(progress=progress, total=total, error=error_message)
                    return
                
                progress += 1
                yield self.process_update(progress=progress, total=total)

                out["mappings"][return_type] = mapping

            # Save to a collection
            if client.collections.exists(f"ELYSIA_METADATA_{collection_name}__"):
                client.collections.delete(f"ELYSIA_METADATA_{collection_name}__")

            metadata_collection = client.collections.create(f"ELYSIA_METADATA_{collection_name}__")
            metadata_collection.data.insert(out)

        yield self.process_update(progress=total, total=total)


# if __name__ == "__main__":
#     import dspy
#     lm = dspy.LM(model="claude-3-5-haiku-20241022")
#     dspy.settings.configure(lm=lm)

#     preprocessor = CollectionPreprocessor()
#     processed = preprocessor("ecommerce", force=True)

#     from rich import print
#     print(processed)