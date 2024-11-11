import dspy

from typing import Literal, get_args, Union

def construct_query_initialiser_prompt(collection_names: list[str] = None, return_types: list[str] = None) -> dspy.Signature:

    # Create dynamic Literal type from the list, or use str if None
    CollectionLiteral = (Literal[tuple(collection_names)] if collection_names is not None  # type: ignore
                  else str)
    
    ReturnTypeLiteral = (Literal[tuple(return_types)] if return_types is not None  # type: ignore
                  else str)
    
    class QueryInitialiserPrompt(dspy.Signature):
        f"""
        Given a user prompt, create a query to retrieve relevant documents.

        # Collection instructions
        You have access to the following collections:
        {collection_names}

        Pick one that best represents the user prompt. If multiple collections are relevant, pick one.

        # Return type instructions
        You have access to the following return types:
        {return_types}

        Pick one that best represents the user prompt. You may only choose one, so pick the best one, most relevant to the user prompt.
        This information will be displayed to the user in a dynamic way, so pick the one that will be most useful.
        """

        user_prompt = dspy.InputField(desc="The user's original query")
        reference = dspy.InputField(desc="Information about the state of the world NOW such as the date and time, used to frame the query.")
        previous_reasoning = dspy.InputField(
            desc="""
            Your reasoning that you have output from previous decisions.
            This is so you can use the information from previous decisions to help you decide which collection and return type to choose.

            This is a dictionary of the form:
            {
                "tree_1": 
                {
                    "decision_1": "Your reasoning for the decision 1",
                    "decision_2": "Your reasoning for the decision 2",
                    ...
                },
                "tree_2": {
                    "decision_1": "Your reasoning for the decision 1",
                    "decision_2": "Your reasoning for the decision 2",
                    ...
                }
            }
            where `tree_1`, `tree_2`, etc. are the ids of the trees in the tree, and `decision_1`, `decision_2`, etc. are the ids of the decisions in the tree.
            
            Use this to base your current action from previous reasoning.
            """.strip()
        )
        
        collection_name: CollectionLiteral = dspy.OutputField(
            desc="The name of the collection to query. Only provide the name exactly as it appears.",
            format = str
        )
        return_type: ReturnTypeLiteral = dspy.OutputField(
            desc="The type of objects to return. Only provide the type name exactly as it appears.",
            format = str
        )

    return QueryInitialiserPrompt

class QueryCreatorPrompt(dspy.Signature):
    """
    You must write code to retrieve the objects from the collection.

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

    You can also sort the results using the `sort` parameter, but only when using `fetch_objects`.
    So you CANNOT use it with `near_text` or `hybrid`.

    For example:
    ```
    collection.query.fetch_objects(
        filters=Filter.by_property("answer").like("*inter*"),
        sort = Sort.by_property("answer", ascending=True),
        limit=3
    )
    ```

    Use the above examples to guide you, but create your own query that is specific to the user prompt.
    You should not use one of the above examples directly, but rather use them as a guide to create your own query.
    Filters are optional, and if not specified in the user prompt, you should not use them.

    You have access to a function called `format_datetime(dt)` which formats a datetime object to the ISO format without the timezone offset. 
    Use this function to format the datetime objects in the filters.

    Assume you have access to the object `collection` which is a Weaviate collection.  
    """
    user_prompt = dspy.InputField(desc="The user's original query")
    reference = dspy.InputField(desc="""
        Information about the state of the world NOW such as the date and time, used to frame the query.
        """.strip(), 
        format = str
    )
    previous_reasoning = dspy.InputField(
        desc="""
        Your reasoning that you have output from previous decisions.
        This is so you can use the information from previous decisions to help you decide what type of query to create.

        This is a dictionary of the form:
        {
            "tree_1": 
            {
                "decision_1": "Your reasoning for the decision 1",
                "decision_2": "Your reasoning for the decision 2",
                ...
            },
            "tree_2": {
                "decision_1": "Your reasoning for the decision 1",
                "decision_2": "Your reasoning for the decision 2",
                ...
            }
        }
        where `tree_1`, `tree_2`, etc. are the ids of the trees in the tree, and `decision_1`, `decision_2`, etc. are the ids of the decisions in the tree.
        
        Use this to base your current action from previous reasoning.
        """.strip(), 
        format = str
    )
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
