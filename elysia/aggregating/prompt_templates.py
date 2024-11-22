import dspy

from typing import Literal, get_args, Union

___filtering = """
    # Example [6]:

    In this example, an ecommerce collection is grouped by the property name "collection", referring to the collection of the product.
    Only collections with a rating greater than 4 are returned.

    ```
    response = collection.aggregate.over_all(
        group_by=GroupByAggregate(prop="collection"),
        filters=Filter.by_property("rating").greater_than(4.),
    )
    ```

    Note that the `filters` argument is used to filter the results after the aggregation has been performed.
    
    Also note that the `4.` is used with the full stop `.` to convert the integer 4 to a float. This is based on the data type of the property, which is a number.
    If it was an integer, then you would use `4` without the full stop.

    The `Filter` class has the following methods:
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

    Do NOT assume any information about the contents of the collection when filtering, only use what information you have.
    For example, if the user prompt asks for the number of articles by a particular author, you do not know how the authors name is spelled, so you should not use the `.equal` method.
    Instead, you should first aggregate by the data field that likely contains the author field, and then the information will be grouped by the author field, and can be parsed later.
    
    ONLY use filter if the data type is numerical, in which case the user prompt should specify exactly the numerical value to filter on.
"""

def construct_aggregate_initialiser_prompt(collection_names: list[str] = None) -> dspy.Signature:

    # Create dynamic Literal type from the list, or use str if None
    CollectionLiteral = (Literal[tuple(collection_names)] if collection_names is not None  # type: ignore
                  else str)
    
    class AggregateInitialiserPrompt(dspy.Signature):
        """
        Given a user prompt, choose the most appropriate collection for a later aggregation.

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

        current_message = dspy.InputField(
            description="""
            The current message you, the assistant, have written to send to the user. 
            This message has not been sent yet, you will add text to it, to be sent to the user later.
            In essence, the concatenation of this field, current_message, and the response field, will be sent to the user.
            """.strip(),
            format = str
        )
        
        collection_name: CollectionLiteral = dspy.OutputField(
            desc="The name of the collection to aggregate. Only provide the name exactly as it appears.",
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

    return AggregateInitialiserPrompt

class AggregatePrompt(dspy.Signature):
    """
    You are an expert at retrieving metadata about data collections.
    You must write code to aggregate the objects from the collection.
    Your goal is retrieving metadata about the data collection, such as the number of objects, the fields, and the types of the fields.

    You should base your aggregation on the previous reasoning, and the user prompt mostly. 
    The information you return from this will be used to query the collection later. 
    For example, if the user prompt asks for something related to an author, you should be attempting to find the number of objects per author.
    This will also determine what authors are available to filter on in a later query.
    Aim to retrieve as much information as possible.

    To do so, you should use the `collection.aggregate.over_all` function, assume you have access to the object `collection` which is a Weaviate collection, `GroupByAggregate`, and `Metrics`.

    The `collection.aggregate.over_all` function has the following signature:
    
    ___

    collection.aggregate.over_all(*, 
            group_by: Union[str, weaviate.collections.classes.aggregate.GroupByAggregate, NoneType] = None, 
            total_count: bool = True, 
            return_metrics: ... = None
        ) 
        -> Union[weaviate.collections.classes.aggregate.AggregateReturn, weaviate.collections.classes.aggregate.AggregateGroupByReturn] method of weaviate.collections.aggregate._AggregateCollection instance
    Aggregate metrics over all the objects in this collection without any vector search.

    Arguments:
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

    You need to match the data type of the property you are aggregating over.

    You also have access to `Metrics` which is a class with the following methods:
    
    `Metrics(property_name).text()`
    `Metrics(property_name).number()`
    `Metrics(property_name).integer()`
    `Metrics(property_name).boolean()`
    `Metrics(property_name).date_()`
    
    Where the property_name is the name of the property you are aggregating over, as a string.

    Metrics refers to the outputs of the aggregation, and can be combined with the `group_by` argument to return the metrics you require grouped by a particular property that you will choose.

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

    # Example [4]:


    In this example, an ecommerce collection is grouped by the property "category", referring to the category of the product.
    The most common categories are returned, and the number of products in each category are also returned.
    Common categories is returned by setting `top_occurrences_count=True`
    The value of the most common category is returned by setting `top_occurrences_value=True`
    Only categories with at least 5 products are returned by setting `min_occurrences=5`

    ```
    response = collection.aggregate.over_all(
        return_metrics=Metrics("message_content").text(
            top_occurrences_count=True,
            top_occurrences_value=True,
            min_occurrences=5  # Threshold minimum count
        )
    )
    ```

    # Example [5]:

    In this example, an ecommerce collection is grouped by the property "category", referring to the category of the product.
    The sum, maximum, and minimum of the price of the products are returned, grouped by the category.

    ```
    response = collection.aggregate.over_all(
        group_by=GroupByAggregate(prop="category"),
        return_metrics=Metrics("price").number(sum_=True, maximum=True, minimum=True),
    )
    ```
    Note the underscore `_` after the `sum` argument. This is intended and required.

    Use the above examples to guide you, but create your own aggregation function that is specific to the user prompt.
    You should not use one of the above examples directly, but rather use them as a guide to create your own aggregation function.
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
        You should use these to understand the format of the data, and to create your aggregation code.
        """.strip(), 
        format = str
    )
    # property_name = dspy.OutputField(
    #     desc="""
    #     The name of the property in the collection (one of the data fields) you are aggregating over.
    #     Return the name exactly as it appears in the data fields.
    #     """.strip(),
    #     format = str
    # )
    # aggregate_type = dspy.OutputField(
    #     desc="""
    #     The type of aggregation you are performing on the property. 
    #     This should be as specific as possible, but can be any text that is an extremely brief (1-2 words) description of the aggregation you are performing.
    #     This is to be displayed to the user, so it should be friendly and informative.
    #     E.g. "Count", "Total", "Top Occurrences", "Mean", "Median", "Minimum", "Maximum", "Sum"
    #     """.strip(),
    #     format = str
    # )
    description = dspy.OutputField(
        desc="A description of the aggregation you are performing, concise and informative.",
        format = str
    )
    is_aggregation_possible = dspy.OutputField(
        desc="""
        A boolean value indicating whether the aggregation is able to return any information. (True/False). Return True if the aggregation is able to return information, and False otherwise.
        Base this decision on the collection metadata, and the user prompt.
        If, for example, the data fields do not likely contain the information the user is asking for, you should return False, as the aggregation is not possible.
        However, if the aggregation is extremely generic, and the user prompt does not specify what to filter on, you should return True.
        Return True or False only, nothing else.
        """.strip(),
        format = bool
    )
    code = dspy.OutputField(
        desc="The generated code only. Do not enclose it in quotes or in ```. Just the code only.",
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