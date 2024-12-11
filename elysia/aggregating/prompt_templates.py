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

def construct_aggregate_prompt(collection_names: list[str] = None) -> dspy.Signature:

    # Create dynamic Literal type from the list, or use str if None
    CollectionLiteral = (Literal[tuple(collection_names)] if collection_names is not None  # type: ignore
                  else str)
    

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
        Another example of using the filters argument is something like
        ```
        collection.aggregate.over_all(
            ...
            filters=Filter.by_property("id").equal("123")
        )
        ```
        which is ensuring the "id" property is equal to "123" across all results.

        Use the collection information to determine the property to filter on, as this is likely requiring exact string matching between what is in the data.
        
        ONLY use filter if the data type is numerical, in which case the user prompt should specify exactly the numerical value to filter on.

        Use the above examples to guide you, but create your own aggregation function that is specific to the user prompt.
        You should not use one of the above examples directly, but rather use them as a guide to create your own aggregation function.
        """

        user_prompt: str = dspy.InputField(desc="The user's original query")
        reference: dict = dspy.InputField(desc="""
            Information about the state of the world NOW such as the date and time, used to frame the aggregation.
            """.strip(), 
            format = str
        )
        previous_reasoning: dict = dspy.InputField(
            desc="""
            Your reasoning that you have output from previous decisions.
            This is so you can use the information from previous decisions to help you decide what type of aggregation to create.

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
        conversation_history: list[dict] = dspy.InputField(
            description="""
            The conversation history between the user and the assistant (you), including all previous messages.
            During this conversation, the assistant has also generated some information, which is also relevant to the decision.
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
        data_queried: str = dspy.InputField(
            description="""
            A list of items, showing whether a query has been completed or not.
            This is an itemised list, showing which collections have been queried and aggregated over already, and how many items have been retrieved from each (for the query), and a description of the aggregation if it exists.
            Use this to determine which collection to aggregate over next, based on what has been aggregated over already (and the user prompt).
            """.strip(),
            format = str
        )

        collection_information: dict = dspy.InputField(desc="""
            Information about each of the collections, so that you can choose which collection to aggregate over, as well as understand the format of the collection you will eventually aggregate over.
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
            Use this to determine which collection to aggregate over, based on the user prompt.
            """.strip(), 
            format = str
        )

        previous_aggregations: list[str] = dspy.InputField(
            description="""
            A list of previous aggregations that have been performed.
            Use this so that you can avoid performing the same aggregation twice.
            Do not use any code that exists within this list, only use the information to avoid performing the same aggregation twice.
            """.strip(),
            format = str
        )
        
        collection_name: CollectionLiteral = dspy.OutputField(
            desc="The name of the collection to aggregate over.",
            format = str
        )
        description: str = dspy.OutputField(
            desc="A description of the aggregation you are performing, concise and informative.",
            format = str
        )
        is_aggregation_possible: bool = dspy.OutputField(
            desc="""
            A boolean value indicating whether the aggregation is able to return any information. (True/False). Return True if the aggregation is able to return information, and False otherwise.
            Base this decision on the collection metadata, and the user prompt.
            If, for example, the data fields do not likely contain the information the user is asking for, you should return False, as the aggregation is not possible.
            However, if the aggregation is extremely generic, and the user prompt does not specify what to filter on, you should return True.
            """.strip(),
            format = bool
        )
        code: str = dspy.OutputField(
            desc="The generated code only. Do not enclose it in quotes or in ```. Just the code only. Do not include any comments.",
            format = str
        )
        reasoning_update_message: str = dspy.OutputField(
            description="Write out current_message in full, then add one sentence to the paragraph which explains your task selection logic. Mark your new sentence with <NEW></NEW>. If current_message is empty, your whole message should be enclosed in <NEW></NEW>. You are communicating directly to the user, so use gender-neutral language, be friendly, and do not say 'the user', as you're speaking to them directly."
        )

    return AggregatePrompt