import spacy
import dspy
import os
import random
import datetime
import weaviate
import weaviate.classes as wvc
from weaviate.classes.init import Auth
from weaviate.classes.aggregate import GroupByAggregate
from weaviate.classes.query import Filter, Metrics, Sort
from weaviate.collections.classes.aggregate import (
    GroupByAggregate,
    AggregateGroup,
    AggregateReturn,
    AggregateGroupByReturn,
    AggregateText,
    AggregateNumber,
    AggregateBoolean,
    AggregateDate,
    AggregateInteger
)

from elysia.util.parsing import format_datetime
from elysia.globals.weaviate_client import client
from elysia.util.collection_metadata import get_collection_data_types
from rich import print

lm = dspy.LM(model="claude-3-5-haiku-20241022")
dspy.settings.configure(lm=lm)

nlp = spacy.load("en_core_web_sm")

class DataSummariser(dspy.Signature):
    """
    You are an expert data analyst who provides summaries of datasets.
    Your task is to provide a summary of the data. This should be concise, one paragraph maximum and no more than 5 sentences. 
    Do not calculate any statistics such as length, just describe the data.
    This is to inform the user about the data in the collection.
    """

    data = dspy.InputField(description="""
    A subset of the data to summarise. This will be a list of JSON objects. Each item is an individual item in the dataset and has the same fields.
    Because this is a subset, do not make definitive statements about _all_ of what the data contains.
    Instead, you can make statements like "this data includes..."
    """)
    data_fields = dspy.InputField(description="""
    The fields that exist in the data. This will be a list of field names.
    """)
    sentence1 = dspy.OutputField(description="""
    A single sentence that summarises the type of data in the collection, what the content is. 
    Remember that this is a subset of the data, so do not make definitive statements about _all_ of what the data contains.
    """)
    sentence2 = dspy.OutputField(description="""
    A breakdown of EACH OF THE DIFFERENT FIELDS, and what sort of data they contain, this is to give the user a good indication of what different fields within the data exist.
    """)
    sentence3 = dspy.OutputField(description="""
    A summary of the different categories that exist in the data.
    """)
    sentence4 = dspy.OutputField(description="""
    A summary of the different data types that exist in the data.
    """)
    sentence5 = dspy.OutputField(description="""
    Any additional information that you think will be useful to the user to inform them about the data.
    """)

summariser = dspy.ChainOfThought(DataSummariser)

def aggregate_by_data_type(collection, data_type, property, full_response = None):

    out = {"type": data_type}
    if data_type == "number":
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
        out["average"] = response.properties[property].mean
        out["groups"] = []

    elif data_type == "text":
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
            lengths = [
                len(nlp(obj.properties[property])) for obj in full_response.objects
            ]

            out["range"] = [
                min(lengths),
                max(lengths)
            ]
            out["average"] = sum(lengths) / len(lengths)

        else:
            out["range"] = []
            out["mean"] = None


    elif data_type == "boolean":
        response = collection.aggregate.over_all(
            return_metrics=[
                Metrics(property).boolean(
                    percentage_true=True
                )
            ]
        )
        out["groups"] = [True, False]
        out["average"] = response.properties[property].percentage_true
        out["range"] = [0, 1]

    elif data_type == "date":
        response = collection.aggregate.over_all(
            return_metrics=[Metrics(property).date_(
                median=True,
                minimum=True,
                maximum=True
            )]
        )
        minimum = response.properties[property].minimum
        if isinstance(minimum, datetime.datetime):
            minimum = format_datetime(minimum)

        maximum = response.properties[property].maximum
        if isinstance(maximum, datetime.datetime):
            maximum = format_datetime(maximum)

        average = response.properties[property].median
        if isinstance(average, datetime.datetime):
            average = format_datetime(average)

        out["range"] = [minimum, maximum]
        out["average"] = average

    elif data_type.endswith("[]"):

        if full_response is not None:
            lengths = [
                len(obj.properties[property]) for obj in full_response.objects
            ]

            out["range"] = [min(lengths), max(lengths)]
            out["average"] = sum(lengths) / len(lengths)
            out["groups"] = []

    else:
        out["range"] = []
        out["average"] = 0
        out["groups"] = []

    return out

def get_collection_summary(data_types, collection):

    indices = random.sample(range(len(collection)), 100)

    subset_objects = []
    for i, item in enumerate(collection.iterator()):
        if i in indices:
            subset_objects.append(item)

    summary = summariser(data=[obj.properties for obj in subset_objects], data_fields=list(data_types.keys()))
    summary_concat = f"{summary.sentence1}\n{summary.sentence2}\n{summary.sentence3}\n{summary.sentence4}\n{summary.sentence5}"

    return summary_concat

def get_collection_metadata(collection_name):
    collection = client.collections.get(collection_name)
    data_types = get_collection_data_types(collection_name)

    collection_is_manageable = len(collection) < 10000

    # if the collection is manageable, fetch all objects
    if collection_is_manageable:
        full_response = collection.query.fetch_objects(
            limit=len(collection)
        )
    else:
        full_response = None

    # initialise collection metadata
    collection_metadata = {
        "name": collection_name,
        "length": len(collection),
        "summary": get_collection_summary(data_types, collection),
        "fields": {}
    }

    # aggregate data by data type
    for property in data_types:
        collection_metadata["fields"][property] = aggregate_by_data_type(collection, data_types[property], property, full_response)

    return collection_metadata


def create_elysia_collection_metadata(collection_name):
    collection_metadata = get_collection_metadata(collection_name)
    
    name = f"ELYSIA_METADATA_{collection_name}__"
    
    if client.collections.exists(name):
        client.collections.delete(name)

    client.collections.create(name)

    client.collections.get(name).data.insert(collection_metadata)


if __name__ == "__main__":
    create_elysia_collection_metadata("ecommerce")
    create_elysia_collection_metadata("example_verba_github_issues")
    create_elysia_collection_metadata("example_verba_slack_conversations")
    create_elysia_collection_metadata("example_verba_email_chains")