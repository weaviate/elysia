import dspy

class QueryCreatorPrompt(dspy.Signature):
    """
    Given a user prompt, create a weaviate function query to retrieve relevant documents.
    You can use one of the following functions:

    - `collection.query.near_text(query, limit)`: this is a semantic search on the text of the documents.
    - `collection.query.hybrid(query, limit)`: this is a hybrid search on the text of the documents.
    - `collection.query.fetch_objects(filters, limit)`: this is a filter search on the documents.

    Here are some examples of how this code should be written:

    # Basic query
    ```
    collection.query.near_text(
        query="fashion icons",
        limit=3
    )
    ```
    The `limit` parameter controls the number of results returned.

    # Basic hybrid search
    ```
    collection.query.hybrid(
        query="fashion icons",
        limit=3
    )
    ```

    # Basic filter with one condition
    ```
    collection.query.fetch_objects(
        filters=Filter.by_property("round").equal("Double Jeopardy!"),
        limit=3
    )
    ```
    The above is used to retrieve objects from the collection _only_ using filters, no searching.

    # Filter with multiple conditions
    ```
    collection.query.fetch_objects(
        filters=(
            Filter.by_property("round").equal("Double Jeopardy!") &
            Filter.by_property("points").less_than(600)
        ),
        limit=3
    )
    ```
    The above is also used to retrieve objects from the collection _only_ using filters, no searching. 
    You can also use `|` for OR.

    # Nested filters
    ```
    collection.query.fetch_objects(
        filters=Filter.by_property("answer").like("*bird*") &
                (Filter.by_property("points").greater_than(700) | Filter.by_property("points").less_than(300)),
        limit=3
    )
    ```
    To create a nested filter, follow these steps.
    - Set the outer operator equal to And or Or.
    - Add operands.
    - Inside an operand expression, set operator equal to And or Or to add the nested group.
    - Add operands to the nested group as needed.

    # Combining filters and search
    ```
    collection.query.near_text(
        query="fashion icons",
        filters=Filter.by_property("points").greater_than(200),
        limit=3
    )
    ```
    This performs vector search and also filters the results.

    ```
    collection.query.fetch_objects(
        filters=Filter.by_property("answer").contains_any(["australia", "india"]),
        limit=3
    )
    ```
    This is used to retrieve objects where the `answer` property in the data contains any of the strings in `["australia", "india"]`.

    ```
    collection.query.hybrid(
        query="shoes",
        filters=Filter.by_property("answer").like("*inter*"),
        limit=3
    )
    ```
    If the object property is a text, or text-like data type such as object ID, use Like to filter on partial text matches.

    ____
    Now that you have learned how the query function works, your job is to create a query based on the user prompt.
    Use the above examples to guide you, but create your own query that is specific to the user prompt.
    You should not use one of the above examples directly, but rather use them as a guide to create your own query.
    Filters are optional, and if not specified in the user prompt, you should not use them.

    Assume you have access to the object `collection` which is a Weaviate collection.
    """
    user_prompt = dspy.InputField(desc="The user's original query")
    data_fields = dspy.InputField(desc="""
        A list of fields that are available to search over.
        ["field_name", ...]
        """.strip(), 
        format = str
    )
    example_field = dspy.InputField(desc="""
        An example from the collection of what the fields look like, in the following format:
        {
            "field_name": "field_value",
            ...
        }
        You should use these to understand the format of the data, and to create your query.
        """.strip(), 
        format = str
    )
    previous_queries = dspy.InputField(
        desc="""
        A comma separated list of existing code that has been used to query the collection. 
        This can be used to avoid generating duplicate queries. 
        If this field is an empty list, you are generating the first query.
        """.strip()
    )
    code = dspy.OutputField(
        desc="The generated query code only. Do not enclose it in quotes or in ```. Just the code only.",
        format = str
    )