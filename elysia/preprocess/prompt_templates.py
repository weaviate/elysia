import dspy
from typing import Literal

class CollectionSummariserPrompt(dspy.Signature):
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

class ReturnTypePrompt(dspy.Signature):
    """
    You are an expert at determining the type of data in a collection.
    Given some possible return types, you should choose the most appropriate ones for this data.
    Base this decision on the data fields and the collection summary.
    """

    collection_summary = dspy.InputField(desc="A description of the collection.", format = str)
    data_fields = dspy.InputField(desc="The fields that exist in the data and their data types.", format = dict)
    example_objects = dspy.InputField(desc="Example objects to help you understand the data.", format = list[dict])
    possible_return_types = dspy.InputField(desc="The possible return types for the collection to choose from, a dictionary with key value pairs corresponding to the return type and its description.", format = dict)
    return_types: list[str] = dspy.OutputField(
        desc="""
        All different types of return types that would suite this data.
        This should be a list of strings.
        You can choose as many as possible from the keys of possible_return_types.
        These should be ONLY the return types that are in possible_return_types, and nothing else.
        Your output should be formatted as a Python list and nothing else.
        If no return types are appropriate, you should return an empty list i.e. []. This possibility should be considered.
        """.strip(), 
        format = str
    )


class DataMappingPrompt(dspy.Signature):
    """
    You are an expert at mapping data fields to existing field names.
    """

    input_data_fields = dspy.InputField(desc="The input fields to map.", format = str)
    output_data_fields = dspy.InputField(desc="The output fields to map to.", format = str)
    input_data_types = dspy.InputField(desc="The data types of the input fields, a dictionary with key value pairs corresponding to the field and its data type.", format = str)
    collection_information = dspy.InputField(desc="""
        Information about the collection.
        This is of the form:
        {
            "name": collection name,
            "length": number of objects in the collection,
            "summary": summary of the collection,
            "fields": {
                "field_name": {
                    "groups": a comprehensive list of all unique text values that exist in the field. if the field is not text, this should be an empty list,
                    "mean": mean of the field. if the field is text, this refers to the means length (in tokens) of the texts in this field. if the type is a list, this refers to the mean length of the lists,
                    "range": minimum and maximum values of the length.
                    "type": the data type of the field.
                },
                ...
            }
        }
        """.strip(), 
        format = str
    )
    example_objects = dspy.InputField(desc="Example objects to help you understand the data.", format = list[dict])

    field_mapping = dspy.OutputField(
        desc="""
        Your output should be a dictionary of the form:
        {
            "input_field_name": "output_field_name",
            ...
        }
        E.g.
        {
            "title": "name",
            ...
        }
        would map the `title` field to the `name` field.
        If there is no mapping, you should include an empty string, e.g.
        {
            "title": "",
            ...
        }
        Complete all the fields, even if some of them are empty.
        But you can leave out certain output_field's if they are not needed.
        In summary, you should ALWAYS include all input fields as keys of the dictionary, but you can leave out certain output_field's if they are not needed.
        This should be formatted as a Python dictionary and nothing else.
        """.strip(), 
        format = str
    )