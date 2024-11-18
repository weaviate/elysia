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
            A brief, punctual explanation of what actions you have carried out during this task, to display to the user. 
            Do not include many technical details e.g. variable names, 
            just a brief explanation in plain English, in a chat message format, 
            so you should use markdown and respond to the user in a friendly way.
            Do not use emojis, and do not ask the user to confirm or approve of your actions.
            Do not ask the user any questions.
            This is a continuation of the current_message field. 
            This response should be a natural continuation of the current_message field, as if you are continuing the paragraph.
            Use present tense in your text, as if you are currently completing the action.
            If the current_message field is empty, then this response is the beginning of a new message.
            """.strip(),
            format = str
        )

    return QueryInitialiserPrompt

class PropertyGroupingPrompt(dspy.Signature):
    """
    Determine which property(ies) to group the results by.
    The goal is to provide a list of properties that the user is likely to filter on in a later query.
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
    
    property_name = dspy.OutputField(desc="""
        A single property name to group the results by.
        Only provide the property name exactly as it appears.
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
        A brief, punctual explanation of what actions you have carried out during this task, to display to the user. 
        Do not include many technical details e.g. variable names, 
        just a brief explanation in plain English, in a chat message format, 
        so you should use markdown and respond to the user in a friendly way.
        Do not use emojis, and do not ask the user to confirm or approve of your actions.
        Do not ask the user any questions.
        This is a continuation of the current_message field. 
        This response should be a natural continuation of the current_message field, as if you are continuing the paragraph.
        Use present tense in your text, as if you are currently completing the action.
        If the current_message field is empty, then this response is the beginning of a new message.
        """.strip(),
        format = str
    )

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
    
    Remember the most important distinction between the three types of queries:
    - `near_text` and `hybrid` have the `query` argument, which you use for _searching_ the database. These _can_ use `filters` but they CANNOT use `sort`.
    - `fetch_objects` is for retrieving objects that does not need and sort of search (and only using filters/sorting). This has the `filters` argument, and `sort` argument.

    So, if the user prompt requires a search, you should use `near_text` or `hybrid`. But if it is only asking for objects based on certain properties, you should use `fetch_objects`.

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
        A brief, punctual explanation of what actions you have carried out during this task, to display to the user. 
        Do not include many technical details e.g. variable names, 
        just a brief explanation in plain English, in a chat message format, 
        so you should use markdown and respond to the user in a friendly way.
        Do not use emojis, and do not ask the user to confirm or approve of your actions.
        Do not ask the user any questions.
        This is a continuation of the current_message field. 
        This response should be a natural continuation of the current_message field, as if you are continuing the paragraph.
        Use present tense in your text, as if you are currently completing the action.
        If the current_message field is empty, then this response is the beginning of a new message.
        """.strip(),
        format = str
    )

class AggregateCollectionPrompt(dspy.Signature):
    """
    You are an expert at retrieving metadata about data collections.
    You must write code to aggregate the objects from the collection.
    Your goal is retrieving metadata about the data collection, such as the number of objects, the fields, and the types of the fields.

    You should base your aggregation on the previous reasoning, and the user prompt mostly. 
    The information you return from this will be used to query the collection later. 
    For example, if the user prompt asks for something related to an author, you should be attempting to find the number of objects per author.
    This will also determine what authors are available to filter on in a later query.
    Aim to retrieve as much information as possible.

    To do so, you should use the `collection.aggregate` function, assume you have access to the object `collection` which is a Weaviate collection, `GroupByAggregate`, and `Metrics`.

    The `collection.aggregate.over_all` function has the following signature:
    
    ___

    collection.aggregate.over_all(*, 
            filters: Optional[weaviate.collections.classes.filters._Filters] = None, 
            group_by: Union[str, weaviate.collections.classes.aggregate.GroupByAggregate, NoneType] = None, 
            total_count: bool = True, 
            return_metrics: ... = None
        ) 
        -> Union[weaviate.collections.classes.aggregate.AggregateReturn, weaviate.collections.classes.aggregate.AggregateGroupByReturn] method of weaviate.collections.aggregate._AggregateCollection instance
    Aggregate metrics over all the objects in this collection without any vector search.

    Arguments:
        `filters`
            The filters to apply to the search.
        `group_by`
            The property name to group the aggregation by.
        `total_count`
            Whether to include the total number of objects that match the query in the response.
        `return_metrics`
            A list of property metrics to aggregate together after the text search.

    Returns:
        Depending on the presence of the `group_by` argument, either a `AggregateReturn` object or a `AggregateGroupByReturn that includes the aggregation objects.

    ___
    
    Each data type has its own set of available aggregated properties. The following table shows the available properties for each data type.

    Data type	Available properties
    Text	    (count, topOccurrences (value, occurs))
    Number	    (count, minimum, maximum, mean, median, mode, sum)
    Integer	    (count, minimum, maximum, mean, median, mode, sum)
    Boolean	    (count, totalTrue, totalFalse, percentageTrue, percentageFalse)
    Date	    (count, minimum, maximum, mean, median, mode)
    ___

    Here are some examples of how this code should be written:

    Example [1]:

    In the following example, the articles are grouped by the property "inPublication", referring to the article's publisher.
    The "wordCount" is a property of the dataset, and is aggregated by the all possible values (count, maximum, mean, median, minimum, mode, sum).

    ```
    collection.aggregate.over_all(
        group_by=GroupByAggregate(prop="inPublication"),
        total_count=True,
        return_metrics=Metrics("wordCount").integer(
            count=True,
            maximum=True,
            mean=True,
            median=True,
            minimum=True,
            mode=True,
            sum_=True,
        )
    )
    ```
    This returns the total number of articles, and statistics about the word count of the articles, grouped by the publication.

    
    Example [2]:

    In the following example, github issues are grouped by the property "issue_author", referring to the author of the issue.
    ```
    collection.aggregate.over_all(
        total_count=True,
        group_by=GroupByAggregate(prop="issue_author")
    )
    ```
    This returns the total number of issues, each unique issue author, and the number of issues per issue author.
    

    Example [3]:

    In this example, some generic conversations are grouped by the property "conversation_id", referring to the id of the conversation (integer).

    ```
    collection.aggregate.over_all(
        total_count=True,
        group_by=GroupByAggregate(prop="conversation_id")
    )
    ```
    This returns the total number of conversations, each unique conversation id, and the number of conversations per conversation id.
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
        A list of properties within the collection that are available to aggregate over.
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
        You should use these to understand the format of the data, and to create your aggregation code.
        """.strip(), 
        format = str
    )
    code = dspy.OutputField(
        desc="The generated code only. Do not enclose it in quotes or in ```. Just the code only.",
        format = str
    )
    text_return = dspy.OutputField(
        desc="""
        A brief, punctual explanation of what actions you have carried out during this task, to display to the user. 
        Do not include many technical details e.g. variable names, 
        just a brief explanation in plain English, in a chat message format, 
        so you should use markdown and respond to the user in a friendly way.
        Do not use emojis, and do not ask the user to confirm or approve of your actions.
        Do not ask the user any questions.
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
