import dspy
from pydantic import BaseModel, Field


class CollectionSummariserPrompt(dspy.Signature):
    """
    You are an expert data analyst who provides summaries of datasets.
    Your task is to provide a summary of the data. This should be concise, one paragraph maximum and no more than 5 sentences.
    Do not calculate any statistics such as length, just describe the data.
    This is to inform the user about the data in the collection.
    Use markdown formatting (such as headers, bold, italics, lists, etc.) to make the summary more readable.
    """

    sample_size = dspy.InputField(
        description="""
    The number of objects in the data_sample, out of a total number of objects in the collection.
    """
    )
    data_sample = dspy.InputField(
        description="""
    A subset of the data to summarise. This will be a list of JSON objects. Each item is an individual item in the dataset and has the same fields.
    Because this is a subset, do not make definitive statements about _all_ of what the data contains.
    Instead, you can make statements like "this data includes..."
    """
    )
    data_fields = dspy.InputField(
        description="""
    The fields that exist in the data. This will be a list of field names.
    """
    )
    overall_summary: str = dspy.OutputField(
        description="""
        An overall summary of the dataset. Use markdown formatting (such as headers, bold, italics, lists, etc.) to make the summary more readable.
        A brief paragraph.
        """
    )
    relationships: str = dspy.OutputField(
        description="""
    A brief overview of the relationships between the fields. Mention any overlap or relevant points about how the fields are related. Around 2-4 sentences.
    """
    )
    structure: str = dspy.OutputField(
        description="""
    An overview of the structure of the dataset. How is it formatted. Around 1-3 sentences.
    """
    )
    irregularities: str = dspy.OutputField(
        description="""
    Any irregularities in the dataset that you think are important to know. Anything that could cause problems for the user when querying the data. Around 1-2 sentences.
    If there are no irregularities, you don't need to return anything. Say 'No irregularities found.' or something similar.
    """
    )
    field_descriptions: dict[str, str] = dspy.OutputField(
        description="""
    A dictionary of the field names and a description of the type of data in the field, be descriptive. Around 1-2 sentences per field.
    The keys of this dictionary MUST be the field names in the data_fields input.
    """
    )


class ReturnTypePrompt(dspy.Signature):
    """
    You are an expert at determining the type of data in a collection.
    Given some possible return types, you should choose the most appropriate ones for this data.
    Base this decision on the data fields and the collection summary.
    """

    collection_summary = dspy.InputField(
        desc="A description of the collection.", format=str
    )
    data_fields = dspy.InputField(
        desc="The fields that exist in the data and their data types.", format=dict
    )
    example_objects = dspy.InputField(
        desc="Example objects to help you understand the data.", format=list[dict]
    )
    possible_return_types = dspy.InputField(
        desc="The possible return types for the collection to choose from, a dictionary with key value pairs corresponding to the return type and its description.",
        format=dict,
    )
    return_types: list[str] = dspy.OutputField(
        desc="""
        All different types of return types that would suite this data.
        This should be a list of strings.
        You can choose as many as possible from the keys of possible_return_types.
        These should be ONLY the return types that are in possible_return_types, and nothing else.
        Your output should be formatted as a Python list and nothing else.
        If no return types are appropriate, you should return an empty list i.e. []. This possibility should be considered.
        """.strip(),
        format=str,
    )


class DataMappingPrompt(dspy.Signature):
    """
    You are an expert at mapping data fields to existing field names.
    """

    mapping_type: str = dspy.InputField(
        desc=(
            "The type of mapping to perform. "
            "Some examples are 'conversation', 'message', 'ticket', 'product', 'document', 'table', 'generic'. "
            "If the type is 'conversation', then the field mapping should include a 'conversation_id' field and a 'message_id' field. "
            "If there is no suitable mapping, return these fields as empty strings."
        ),
        format=str,
    )
    input_data_fields: list[str] = dspy.InputField(
        desc="The input fields to map.", format=list[str]
    )
    output_data_fields: list[str] = dspy.InputField(
        desc="The output fields to map to.", format=list[str]
    )
    input_data_types: dict[str, str] = dspy.InputField(
        desc="The data types of the input fields, a dictionary with key value pairs corresponding to the field and its data type.",
        format=dict[str, str],
    )
    collection_information: dict[str, str | dict] = dspy.InputField(
        desc="""
        Information about the collection.
        This is of the form:
        {
            "name": collection name,
            "length": number of objects in the collection,
            "summary": summary of the collection,
            "fields": [
                {
                    "name": field_name,
                    "type": the data type of the field.
                    "groups": a dict with the value and count of each group.
                        a comprehensive list of all unique values that exist in the field.
                        if this is None, then no relevant groups were found.
                        these values are string, but the actual values in the collection are the 'type' of the field.
                    "mean": mean of the field. if the field is text, this refers to the means length (in tokens) of the texts in this field. if the type is a list, this refers to the mean length of the lists,
                    "range": minimum and maximum values of the length.
                },
                ...
            ]
        }
        """.strip(),
        format=str,
    )
    example_objects: list[dict] = dspy.InputField(
        desc="Example objects to help you understand the data.", format=list[dict]
    )

    field_mapping: dict[str, str] = dspy.OutputField(
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
        Do not create any new fields, only map existing fields.
        The keys in your output dictionary should be the input_field_names and no others.
        """.strip(),
        format=dict[str, str],
    )


class PromptSuggestorPrompt(dspy.Signature):
    """
    You are an expert at suggesting prompts for a given data collection.
    """

    collection_information: dict[str, str | dict] = dspy.InputField(
        desc="""
        Information about the collection.
        This is of the form:
        {
            "name": collection name,
            "length": number of objects in the collection,
            "summary": summary of the collection,
            "fields": [
                {
                    "name": field_name,
                    "groups": a dict with the value and count of each group.
                        a comprehensive list of all unique values that exist in the field.
                        if this is None, then no relevant groups were found.
                        these values are string, but the actual values in the collection are the 'type' of the field.
                    "mean": mean of the field. if the field is text, this refers to the means length (in tokens) of the texts in this field. if the type is a list, this refers to the mean length of the lists,
                    "range": minimum and maximum values of the length.
                    "type": the data type of the field.
                },
                ...
            ]
        }
        """.strip(),
        format=str,
    )
    example_objects: list[dict] = dspy.InputField(
        desc="Example objects to help you understand the data.", format=list[dict]
    )

    prompt_suggestions: list[str] = dspy.OutputField(
        desc="""
        A list of prompts that would be useful for the user to use to query the data.
        These should be in the form of a question that the user could ask to get information about the data.
        These prompts should be around 5-10 words long.
        They should show actual understanding of the data, and not just be a generic question.
        Use no formatting, just the plain text of the prompt.
        Aim to impress the user with your understanding of the data, and generate interesting and meaningful questions.
        Also aim to create questions that will help the user dive deeper into their data.
        Look for interactions between the fields, and connections that the user may not see themselves.
        Produce 10 questions.
        The prompts should either: specifically reference the collection, or be specific enough to the data that at a glance someone would recognise the data.
        """
    )
