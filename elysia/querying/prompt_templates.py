import dspy
from typing import Literal, get_args, Union


def construct_query_prompt(collection_names: list[str] = None) -> dspy.Signature:

    # Create dynamic Literal type from the list, or use str if None
    CollectionLiteral = (Literal[tuple(collection_names)] if collection_names is not None  # type: ignore
                  else str)
        
    class QueryCreatorPrompt(dspy.Signature):
        """
        You are an expert Query agent. You decide what collection to query, how the queried objects should be returned, and most importantly, how to query the collection via writing python code in Weaviate's Python client.
        
        Instructions for writing the query:
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

        If using `hybrid`, `bm25` or `near_text` (i.e. any of the _searches_), you MUST include text as part of the `query=` argument, it cannot be an empty string.

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

        You can only use sorting once. Do not combine multiple sorts.

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

        The `Filter` class has the following methods:
        |  Static methods defined here:
        |
        |  .all_of(filters: List[weaviate.collections.classes.filters._Filters]) -> weaviate.collections.classes.filters._Filters
        |      Combine all filters in the input list with an AND operator.
        |
        |  .any_of(filters: List[weaviate.collections.classes.filters._Filters]) -> weaviate.collections.classes.filters._Filters
        |      Combine all filters in the input list with an OR operator.
        |
        |  .by_property(name: str, length: bool = False) -> weaviate.collections.classes.filters._FilterByProperty
        |      Define a filter based on a property to be used when querying and deleting from a collection.

        Most of the time when using filters, you will be using the `.by_property` method.
        The `.by_property` method has the following methods:
        |  .contains_all(self, val: Union[Sequence[str], Sequence[bool], Sequence[int], Sequence[float], Sequence[datetime.datetime], Sequence[Union[str, uuid.UUID]]]) -> weaviate.collections.classes.filters._Filters
        |      Filter on whether the property contains all of the given values.
        |
        |  .contains_any(self, val: Union[Sequence[str], Sequence[bool], Sequence[int], Sequence[float], Sequence[datetime.datetime], Sequence[Union[str, uuid.UUID]]]) -> weaviate.collections.classes.filters._Filters
        |      Filter on whether the property contains any of the given values.
        |
        |  .equal(self, val: Union[int, float, str, bool, datetime.datetime, uuid.UUID, weaviate.collections.classes.filters._GeoCoordinateFilter, NoneType, Sequence[str], Sequence[bool], Sequence[int], Sequence[float], Sequence[datetime.datetime], Sequence[Union[str, uuid.UUID]]]) -> weaviate.collections.classes.filters._Filters
        |      Filter on whether the property is equal to the given value.
        |
        |  .greater_or_equal(self, val: Union[int, float, str, bool, datetime.datetime, uuid.UUID, weaviate.collections.classes.filters._GeoCoordinateFilter, NoneType, Sequence[str], Sequence[bool], Sequence[int], Sequence[float], Sequence[datetime.datetime], Sequence[Union[str, uuid.UUID]]]) -> weaviate.collections.classes.filters._Filters
        |      Filter on whether the property is greater than or equal to the given value.
        |
        |  .greater_than(self, val: Union[int, float, str, bool, datetime.datetime, uuid.UUID, weaviate.collections.classes.filters._GeoCoordinateFilter, NoneType, Sequence[str], Sequence[bool], Sequence[int], Sequence[float], Sequence[datetime.datetime], Sequence[Union[str, uuid.UUID]]]) -> weaviate.collections.classes.filters._Filters
        |      Filter on whether the property is greater than the given value.
        |
        |  .is_none(self, val: bool) -> weaviate.collections.classes.filters._Filters
        |      Filter on whether the property is `None`.
        |
        |  .less_or_equal(self, val: Union[int, float, str, bool, datetime.datetime, uuid.UUID, weaviate.collections.classes.filters._GeoCoordinateFilter, NoneType, Sequence[str], Sequence[bool], Sequence[int], Sequence[float], Sequence[datetime.datetime], Sequence[Union[str, uuid.UUID]]]) -> weaviate.collections.classes.filters._Filters
        |      Filter on whether the property is less than or equal to the given value.
        |
        |  .less_than(self, val: Union[int, float, str, bool, datetime.datetime, uuid.UUID, weaviate.collections.classes.filters._GeoCoordinateFilter, NoneType, Sequence[str], Sequence[bool], Sequence[int], Sequence[float], Sequence[datetime.datetime], Sequence[Union[str, uuid.UUID]]]) -> weaviate.collections.classes.filters._Filters
        |      Filter on whether the property is less than the given value.
        |
        |  .like(self, val: str) -> weaviate.collections.classes.filters._Filters
        |      Filter on whether the property is like the given value.
        |
        |      This filter can make use of `*` and `?` as wildcards. 
        |
        |  .not_equal(self, val: Union[int, float, str, bool, datetime.datetime, uuid.UUID, weaviate.collections.classes.filters._GeoCoordinateFilter, NoneType, Sequence[str], Sequence[bool], Sequence[int], Sequence[float], Sequence[datetime.datetime], Sequence[Union[str, uuid.UUID]]]) -> weaviate.collections.classes.filters._Filters
        |      Filter on whether the property is not equal to the given value.
        
        ___ 

        Remember the most important distinction between the three types of queries:
        - `near_text` and `hybrid` have the `query` argument, which you use for _searching_ the database. These _can_ use `filters` but they CANNOT use `sort`.
        - `fetch_objects` is for retrieving objects that does not need and sort of search (and only using filters/sorting). This has the `filters` argument, and `sort` argument.

        So, if the user prompt requires a search, you should use `near_text`, `hybrid` or `bm25`. 
        But if it is only asking for objects based on certain properties, something you can achieve by ONLY filtering, you should use `fetch_objects`.

        Use the above examples to guide you, but create your own query that is specific to the user prompt.
        You should not use one of the above examples directly, but rather use them as a guide to create your own query.
        Filters are optional, and if not specified in the user prompt, you should not use them.

        Use python's `datetime.datetime` function, not `datetime` on its own, to process date or time based filters/sorting/etc.

        Assume you have access to the object `collection` which is a Weaviate collection.  
        """
        user_prompt: str = dspy.InputField(desc="The user's original query")
        reference: dict = dspy.InputField(desc="""
            Information about the state of the world NOW such as the date and time, used to frame the query.
            """.strip(), 
            format = str
        )
        conversation_history: list[dict] = dspy.InputField(
            description="""
            The conversation history between the user and the assistant (you), including all previous messages.
            During this conversation, the assistant has also generated some information, which is also relevant to the decision.
            This information is stored in `available_information` field.
            If this is non-empty, then you have already been speaking to the user, and these were your responses, so future responses should use these as context.
            The history is a list of dictionaries of the format:
            [
                {
                    "role": "user" or "assistant",
                    "content": The message
                }
            ]
            In the order which the messages were sent.
            """.strip(),
            format = str
        )
        previous_reasoning: dict = dspy.InputField(
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


        collection_information: dict = dspy.InputField(desc="""
            Information about each of the collections, so that you can choose which collection to query, as well as understand the format of the collection you will eventually query.
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
            You will be given one of these for each collection that you can choose from.
            Use this to determine which collection to query, based on the user prompt.
            """.strip(), 
            format = str
        )

        data_queried: str = dspy.InputField(
            description="""
            An itemised list of items, showing whether a query has been completed or not.
            This is an itemised list, showing which collections have been queried, and how many items have been retrieved from each.
            If there are 0 items retrieved, then the collection _has_ been queried, but no items were found. Use this in your later judgement.
            Use this to determine which collection to query next, based on what has been queried already (and the user prompt).
            """.strip(),
            format = str
        )

        previous_queries: list[str] = dspy.InputField(
            desc="""
            For each collection, a comma separated list of existing code that has been used to query the collection. 
            This can be used to avoid generating duplicate queries. 
            If this field is an empty list, you are generating the first query.
            """.strip()
        )

        current_message: str = dspy.InputField(
            description="""
            The current message you, the assistant, have written to send to the user. 
            This message has not been sent yet, you will add text to it, to be sent to the user later.
            In essence, the concatenation of this field, current_message, and the response field, will be sent to the user.
            """.strip(),
            format = str
        )

        collection_return_types: dict[str, list[str]] = dspy.InputField(
            desc="""
            A dictionary of the return types that you can choose from for each collection.
            This is of the form:
            {
                "collection_name": ["return_type_1", "return_type_2", ...],
                ...
            }
            where `collection_name` is the name of the collection, that _you will decide_, and `return_type_1`, `return_type_2`, etc. are the return types that you can choose from for that collection.
            The output of `return_type` must be one of the values in the list for the collection. Make sure you pick from the correct collection and do not pick a return type that does not exist for the collection.
            """.strip(),
            format = dict[str, list[str]]
        )
                
        collection_name: CollectionLiteral = dspy.OutputField(
            desc="The name of the collection to query. Only provide the name exactly as it appears.",
            format = str
        )

        return_type: str = dspy.OutputField(
            desc="""
            The type of objects to return. Only provide the type name exactly as it appears.
            This must be one of the values in the list for that particular collection.
            """.strip(),
            format = str
        )

        output_type: Literal["original", "summary"] = dspy.OutputField(
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

        code: str = dspy.OutputField(
            desc="The generated query code only. Do not enclose it in quotes or in ```. Just the code only. Do not add any comments.",
            format = str
        )

        is_query_possible: bool = dspy.OutputField(
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
        
        reasoning_update_message: str = dspy.OutputField(
            description="Write out current_message in full, then add one sentence to the paragraph which explains your task selection logic. Mark your new sentence with <NEW></NEW>. If current_message is empty, your whole message should be enclosed in <NEW></NEW>. Use gender-neutral language and communicate to the user in a friendly way."
        )

    return QueryCreatorPrompt

class ObjectSummaryPrompt(dspy.Signature):
    """
    You must write code to summarise the objects.

    Given a list of objects (a list of dictionaries, where each item in the dictionary is a field from the object), 
    you must provide a list of strings, where each string is a summary of the object.

    These objects can be of any type, and you should summarise them in a way that is useful to the user.
    """
    objects = dspy.InputField(desc="The objects to summarise.", format = list[dict])
    current_message = dspy.InputField(
        description="""
        The current message you, the assistant, have written to send to the user. 
        This message has not been sent yet, you will add text to it, to be sent to the user later.
        In essence, the concatenation of this field, current_message, and the response field, will be sent to the user.
        """.strip(),
        format = str
    )

    reasoning_update_message: str = dspy.OutputField(
        description="Write out current_message in full, then add one sentence to the paragraph which explains your task selection logic. Mark your new sentence with <NEW></NEW>. If current_message is empty, your whole message should be enclosed in <NEW></NEW>. Use gender-neutral language and communicate to the user in a friendly way."
    )
    summaries = dspy.OutputField(desc="""
        The summaries of each individaual object, in a list of strings.
        Your output should be a list of strings in Python format, e.g. `["summary_1", "summary_2", ...]`.
        """.strip(), 
        format = list[str]
    )
