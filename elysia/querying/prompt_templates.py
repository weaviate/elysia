import dspy

from typing import Literal, get_args, Union

def construct_query_initialiser_prompt(collection_names: list[str] = None, return_types: dict[str, str] = None) -> dspy.Signature:

    # Create dynamic Literal type from the list, or use str if None
    CollectionLiteral = (Literal[tuple(collection_names)] if collection_names is not None  # type: ignore
                  else str)
    
    ReturnTypeLiteral = (Literal[tuple(return_types.keys())] if return_types is not None  # type: ignore
                  else str)
    
    class QueryInitialiserPrompt(dspy.Signature):
        """
        Given a user prompt, choose the most appropriate collection, return type and output type for a later query.

        Pick ones that best represents the user prompt. You may only choose one of each, so pick the best one, most relevant to the user prompt.
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
        
        data_queried = dspy.InputField(
            description="""
            A list of items, showing whether a query has been completed or not.
            This is an itemised list, showing which collections have been queried, and how many items have been retrieved from each.
            If there are 0 items retrieved, then the collection _has_ been queried, but no items were found. Use this in your later judgement.
            """.strip(),
            format = str
        )

        available_collections = dspy.InputField(
            description="A list of the collections that you have access to.",
            format = str
        )

        available_return_types = dspy.InputField(
            description="A list of the return types that you have access to, with corresponding descriptions of each one.",
            format = dict
        )
        
        collection_name: CollectionLiteral = dspy.OutputField(
            desc="The name of the collection to query. Only provide the name exactly as it appears.",
            format = str
        )

        return_type: ReturnTypeLiteral = dspy.OutputField(
            desc="The type of objects to return. Only provide the type name exactly as it appears.",
            format = str
        )

        output_type = dspy.OutputField(
            desc="""
            One of: 'original' or 'summary'. Output the name exactly as it appears.
            'original' means you will return the original objects returned. Use this most of the time, unless the user has specifically asked for summaries.
            'summary' means you will return an itemised summary of each of the objects returned.
            You should only choose 'summary' if the user has specifically asked for individual summaries of the objects in the user_prompt.
            I.e., in the user prompt, they have directly asked for summaries of the objects.
            Otherwise, you should choose 'original'.
            Be _very_ sparing with 'summary', as it is more expensive to compute.
            """.strip(),
            format = str
        )

        current_message = dspy.InputField(
            description="""
            The current message you, the assistant, have written to send to the user. 
            This message has not been sent yet, you will add text to it, to be sent to the user later.
            In essence, the concatenation of this field, current_message, and the response field, will be sent to the user.
            """.strip(),
            format = str
        )
        text_return = dspy.OutputField(
            desc="""
            Begin this field with the text in current_message field, which is your message _so far_ to the user. Avoid repeating yourself (from the current_message field). If this field is empty, this is a new message you are starting.
            You should write out exactly what it says in current_message, and then afterwards, continue with your new reasoning to communicate anything else to the user.
            Your additions should be a brief succint version of the reasoning field, that will be communicated to the user. Do not complete the task within this field, this is just a summary of the reasoning for the decision.
            Communicate this in a friendly and engaging way, as if you are explaining your reasoning to the user in a chat message.
            Do not ask any questions, and do not ask the user to confirm or approve of your actions.
            You should only add one extra sentence to the current_message field, and that is it. Do not add any more.
            If current_message is empty, then this is a new message you are starting, so you should write out only a new message.
            Use gender neutral language.
            """.strip(),
            format = str
        )

    return QueryInitialiserPrompt

def construct_property_grouping_prompt(property_names: list[str]) -> dspy.Signature:

    PropertyLiteral = (Literal[tuple(property_names)] if property_names is not None  # type: ignore
                  else str)

    class PropertyGroupingPrompt(dspy.Signature):
        """
        Determine which property to group the results by.
        The goal is to provide a list of properties that the user is likely to filter on in a later query.
        You are essentially tasked with collecting metadata about the collection for this specific query/user prompt,
        by using a group by command. But you only need to provide the property name, not the entire command.
        For example, if the user prompt asks for articles by a specific author, you should group by "author".
        If the user prompt asks for articles about a specific topic, you should group by "topic".
        """
        user_prompt = dspy.InputField(desc="The user's original query")

        reference = dspy.InputField(desc="""
            Information about the state of the world NOW such as the date and time.
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

        data_types = dspy.InputField(desc="""
            A dictionary of the data types of the fields in the collection.
            {
                "field_name": "field_type",
                ...
            }
            Use this to understand the format of the data, and to create your query.
            Also, use this as a list of fields which you can use in the query, the keys of this dictionary are the names of the fields exactly as they appear.
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

        current_message = dspy.InputField(
            description="""
            The current message you, the assistant, have written to send to the user. 
            This message has not been sent yet, you will add text to it, to be sent to the user later.
            In essence, the concatenation of this field, current_message, and the response field, will be sent to the user.
            """.strip(),
            format = str
        )
        
        property_name: PropertyLiteral = dspy.OutputField(desc="""
            A single property name to group the results by.
            Only provide the property name exactly as it appears.
            """.strip(), 
            format = str
        )
        
        text_return = dspy.OutputField(
            desc="""
            Begin this field with the text in current_message field, which is your message _so far_ to the user. Avoid repeating yourself (from the current_message field). If this field is empty, this is a new message you are starting.
            You should write out exactly what it says in current_message, and then afterwards, continue with your new reasoning to communicate anything else to the user.
            Your additions should be a brief succint version of the reasoning field, that will be communicated to the user. Do not complete the task within this field, this is just a summary of the reasoning for the decision.
            Communicate this in a friendly and engaging way, as if you are explaining your reasoning to the user in a chat message.
            Do not ask any questions, and do not ask the user to confirm or approve of your actions.
            You should only add one extra sentence to the current_message field, and that is it. Do not add any more.
            If current_message is empty, then this is a new message you are starting, so you should write out only a new message.
            Use gender neutral language.
            """.strip(),
            format = str
        )

    return PropertyGroupingPrompt

class QueryCreatorPrompt(dspy.Signature):
    """
    You must write code to retrieve the objects from the collection.

    You can use one of the following functions:

    - `collection.query.near_text(query, limit)`: this is a semantic search on the text of the documents.
    - `collection.query.hybrid(query, limit)`: this is a hybrid search on the text of the documents.
    - `collection.query.fetch_objects(filters, limit)`: this is a filter search on the documents.

    Here are some examples of how this code should be written:

    # Semantic search query
    This query uses semantic search _only_.
    ```
    collection.query.near_text(
        query="fashion icons",
        limit=3
    )
    ```
    Use semantic search if the user prompt requests a search that needs to be based on a specific meaning, and not just a keyword.
    The `query` argument is the search term, and the `limit` argument is the number of results to return.
    The search term searches the _content_ of the documents, and should only contain words or meanings of words that will be in the text of the searched database, NOT the categories or properties.

    # Keyword search query
    This query uses keyword search _only_.
    ```
    collection.query.bm25(
        query="food",
        limit=3
    )
    ```
    Use keyword search if the user prompt requests a search that needs a specific keyword.
    The search term in `query` argument searches the _content_ of the documents for keywords ONLY, not the categories or properties of the data.

    # Hybrid search query
    This query uses hybrid search, a combination of semantic search and keyword search.
    ```
    collection.query.hybrid(
        query="fashion icons",
        limit=3
    )
    ```
    Use hybrid search if the user prompt requests a search that needs a combination of semantic search and keyword search.
    The `query` argument searches the _content_ of the documents for keywords and meaning ONLY, not the categories or properties of the data.

    # Fetch objects query
    This query uses filters to retrieve objects from the collection.
    Use fetch objects if the user prompt requests a search that needs to be based on certain properties of the data, and not just a keyword.
    ```
    collection.query.fetch_objects(
        filters=Filter.by_property("round").equal("Double Jeopardy!"),
        limit=3
    )
    ```
    This does not use any searching, it only uses filters, similar to an SQL based query.

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

    # Filter with contains any
    ```
    collection.query.fetch_objects(
        filters=Filter.by_property("answer").contains_any(["australia", "india"]),
        limit=3
    )
    ```
    This is used to retrieve objects where the `answer` property in the data contains any of the strings in `["australia", "india"]`.

    # Combining filters and search
    This query performs vector search and also filters the results.
    ```
    collection.query.near_text(
        query="fashion icons",
        filters=Filter.by_property("points").greater_than(200),
        limit=3
    )
    ```
    # Combining filter with hybrid search
    ```
    collection.query.hybrid(
        query="shoes",
        filters=Filter.by_property("answer").like("*inter*"),
        limit=3
    )
    ```
    If the object property is a text, or text-like data type such as object ID, use Like to filter on partial text matches.
    This can be on `bm25` also.

    # Sorting results
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

    # Next level queries
    If the user has already queried and received results, and is either asking a follow up question or you are performing a follow up query,
    you can use the results from the previous query to help you create the next query.
    For example, look for specific IDs or categories in the collection which you can use as filters, such as using .equal() or .contains_any() to find
    other results with the same ID or category, but with different properties and modifying the query to the new request.
    E.g.
    ```
    collection.query.fetch_objects(
        filters=Filter.by_property("id").equal("123"),
        limit=3
    )
    ```
    
    ___ 

    Remember the most important distinction between the three types of queries:
    - `near_text` and `hybrid` have the `query` argument, which you use for _searching_ the database. These _can_ use `filters` but they CANNOT use `sort`.
    - `fetch_objects` is for retrieving objects that does not need and sort of search (and only using filters/sorting). This has the `filters` argument, and `sort` argument.

    So, if the user prompt requires a search, you should use `near_text`, `hybrid` or `bm25`. 
    But if it is only asking for objects based on certain properties, something you can achieve by ONLY filtering, you should use `fetch_objects`.

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
    data_types = dspy.InputField(desc="""
        A dictionary of the data types of the fields in the collection.
        {
            "field_name": "field_type",
            ...
        }
        Use this to understand the format of the data, and to create your query.
        Also, use this as a list of fields which you can use in the query, the keys of this dictionary are the names of the fields exactly as they appear.
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
    collection_metadata = dspy.InputField(desc="""
        Metadata about the collection, in the following format:
        {
            "field_name": {
                "item_1": quantity,
                "item_2": quantity,
                ...
            },
            ...
        }
        For each field, the quantity (for each item) is the number of unique instances that item appears in the collection, for that field.
        So you can think of this as a frequency distribution of the field.
        And you can use this to understand the what unique values exist in the collection for that field, to determine what filters you can use.
        As well as whether the query is possible, by checking if the required filter values exist in the collection metadata.
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

    is_query_possible = dspy.OutputField(
        desc="""
        A boolean value indicating whether the query is able to return any information. (True/False). Return True if the query is able to return information, and False otherwise.
        Base this decision on the collection metadata, and the user prompt.
        If, for example, the filter/sort values do not exist in the collection metadata, you should return False, as the query is not possible.
        However, if the query is extremely generic (just a regular search with no filters or search), and the user prompt does not specify what to filter on, you should return True, 
        as the query will just be a regular search.
        Return True or False only, nothing else.
        """.strip(),
        format = bool
    )

    current_message = dspy.InputField(
        description="""
        The current message you, the assistant, have written to send to the user. 
        This message has not been sent yet, you will add text to it, to be sent to the user later.
        In essence, the concatenation of this field, current_message, and the response field, will be sent to the user.
        """.strip(),
        format = str
        )
    
    text_return = dspy.OutputField(
        desc="""
        Begin this field with the text in current_message field, which is your message _so far_ to the user. Avoid repeating yourself (from the current_message field). If this field is empty, this is a new message you are starting.
        You should write out exactly what it says in current_message, and then afterwards, continue with your new reasoning to communicate anything else to the user.
        Your additions should be a brief succint version of the reasoning field, that will be communicated to the user. Do not complete the task within this field, this is just a summary of the reasoning for the decision.
        Communicate this in a friendly and engaging way, as if you are explaining your reasoning to the user in a chat message.
        Do not ask any questions, and do not ask the user to confirm or approve of your actions.
        You should only add one extra sentence to the current_message field, and that is it. Do not add any more.
        If current_message is empty, then this is a new message you are starting, so you should write out only a new message.
        Use gender neutral language.
        """.strip(),
        format = str
    )


class ObjectSummaryPrompt(dspy.Signature):
    """
    You must write code to summarise the objects.

    Given a list of objects (a list of dictionaries, where each item in the dictionary is a field from the object), 
    you must provide a list of strings, where each string is a summary of the object.

    These objects can be of any type, and you should summarise them in a way that is useful to the user.
    """
    objects = dspy.InputField(desc="The objects to summarise.", format = list[dict])
    summaries = dspy.OutputField(desc="""
        The summaries of each individaual object, in a list of strings.
        Your output should be a list of strings in Python format, e.g. `["summary_1", "summary_2", ...]`.
        """.strip(), 
        format = list[str]
    )
