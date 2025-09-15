from typing import List

import dspy

from elysia.tree.objects import Atlas
from elysia.tools.retrieval.util import (
    DataDisplay,
    QueryOutput,
)
from typing import Union


def construct_aggregation_output_prompt(vectorised: bool = False) -> str:
    output_prompt = """
    The AggregationOutput format contains these key components:

    1. target_collections: The collections you want to query (can be multiple, all will share the same aggregation query, but different queries can be chosen by providing multiple AggregationOutput objects)

    2. Filtering mechanism:
    - filter_buckets: Groups of filters connected with AND/OR logic
    - Each filter_bucket contains:
        - filters: List of property-specific filters or nested filter buckets
        - operator: "AND" or "OR" to connect filters within the bucket

    The FilterBucket is recursive - it can contain other FilterBuckets. This allows you to build complex logical expressions like (A AND B) OR (C AND D) by nesting buckets with different operators. For example:
    - A top-level bucket with OR operator containing:
        - A nested bucket with AND operator (for A AND B)
        - Another nested bucket with AND operator (for C AND D)
    - Example: For "January OR March", create a structure like: OR(AND(Jan-start, Jan-end), AND(Mar-start, Mar-end))

    3. Property filter types:
        - IntegerPropertyFilter: For integer comparisons (=, !=, <, >, <=, >=)
        - FloatPropertyFilter: For float comparisons (=, !=, <, >, <=, >=)
            Note that this is for float comparisons, not integer comparisons.
            If the property is an "int", you should use the IntegerPropertyFilter instead.
        - TextPropertyFilter: For text equality (=, !=) or pattern matching (LIKE). 
            When using LIKE, you can use the ? wildcard to match one unknown single character. e.g. "car?" matches "cart", "care", but not "car"
            You can also use the * wildcard to match any number of unknown characters. e.g. "car*" matches "car", "care", "carpet", etc
            "*car*" matches "car", "healthcare", etc
        - BooleanPropertyFilter: For boolean comparisons (=, !=)
        - DatePropertyFilter: For date comparisons (=, !=, <, >, <=, >=)
       Make sure to use the correct filter type for the property type. The property type is given in the schema (field.[field_name].type).
       If you use the incorrect filter type, the query will error. It is extremely important that you use the correct filter type!
       You CANNOT use any type of filter on an 'object' or 'object[]' property. It will raise an error!
       Note most filters are INCLUSIVE. You can use <=, <, >, >=, !=, to filter out results. But e.g. LIKE and = are INCLUSIVE.


    4. Aggregation operations (can choose many per AggregationOutput object, match the property type based on the field in the schema):
        - integer_property_aggregation: MIN, MAX, MEAN, MEDIAN, MODE, SUM on numeric fields
        - text_property_aggregation: TOP_OCCURRENCES for most frequent values
        - boolean_property_aggregation: TOTAL_TRUE, TOTAL_FALSE, PERCENTAGE_TRUE, PERCENTAGE_FALSE
        - date_property_aggregation: MIN, MAX, MEAN, MEDIAN, MODE on date fields
        Make sure to use the correct aggregation operation for the property type. The property type is given in the schema (field.[field_name].type).
        If you use the incorrect aggregation operation, the query will error. It is extremely important that you use the correct aggregation operation!

    5. Grouping:
        - groupby_property: Group aggregation results by a specific property value
        Grouping is optional but recommended for complex questions or those that require segmentation of results
    """

    if vectorised:
        output_prompt += """
    6. Optional Search:
        You can also use search to narrow the results for the aggregation.
        Use this sparingly. Only use if you judge it necessary based on the user's prompt. 
        - search_type: The type of search to use for the aggregation (hybrid, vector). Vector is semantic search, hybrid is a combination of semantic and keyword search. If None, no search is used.
        - search_query: The text to search for (using vector search). Only set this if search_type is specified.
        - limit: The number of results to narrow down to with the search. Only set this if search_type is specified.
            The limit should be large - aggregations are looking for statistics across the dataset.
            Judge this on the size of the dataset and the potential number of results from the search.
            Limit should be greater than 100 unless you judge the task requires a smaller limit.
    """

    return output_prompt


class AggregationPrompt(dspy.Signature):
    """
    You are a database query builder specializing in converting natural language questions into structured aggregation queries for Weaviate databases.
    You are part of an ensemble, a larger system of an agentic framework the user is interacting with, so you are not the only LLM they interact with.
    Given a user question, your job is to formulate precise aggregation queries based on the provided schema information.

    When constructing aggregation queries:
    1. Identify the relevant collection(s) from the schemas provided
    2. Choose the appropriate aggregation operation based on the user's intent (MIN, MAX, TOP_OCCURRENCES, etc.)
    3. Build filter_buckets when specific conditions are needed, using nested buckets for complex logic
    4. Apply grouping when needed to segment the aggregation results

    Provide detailed reasoning about your approach, explaining why you chose specific collections, properties, operations, and filters.
    In your reasoning also be detailed about the nested conditions of filter buckets.
    Include all of your thoughts in any reasoning fields, as this will be used to communicate to later models about your decision process here.

    The schema is provided in the `collection_schemas` field. This is important to use to understand the data types and properties of the collection.
    Ensure that you choose the correct filter types and aggregation operations for the properties in the schema, including values you filter/aggregate on, they must match the data type of the property.
        E.g., if the property is an integer, do not specify a float such as 1.0, it must be 1.
    Use the groups in the schema to understand the data for filters and aggregation operations.

    Return a list of AggregationOutput objects that satisfy the user's request. If multiple approaches are possible, return multiple queries.
    If the aggregation is impossible to execute, set impossible=True and return None.

    Remember that this is NOT the only query being performed in the larger program! So if you need to aggregate as a first or second step towards achieving a goal,
    you should do so, and explain in your reasoning what your future steps are towards this goal.
    You can only aggregate, which is creating summary statistics of data.
    If you think the goal is better suited to another task such as searching, you should still create an aggregation query to answer the question as best as you can.

    IMPORTANT NOTE: any query you perform will be on the FULL collection, not a filtered subset from any previous steps.
    Do not enclose any answers in ```python or ```.
    """

    available_collections: list[str] = dspy.InputField(
        description="A list of collections that are available to query. You can ONLY choose from these collections."
    )
    previous_aggregation_queries: list[str] = dspy.InputField(
        desc="""
        For each collection, a list of existing code that has been used to query the collection.
        This can be used to avoid generating duplicate queries.
        If any of the collections have an empty list, you are generating the first query for that collection.
        """.strip(),
        format=list[dict],
    )


def construct_query_output_prompt(vectorised: bool = False) -> str:
    output_prompt = f"""
    The QueryOutput format contains these key components:

    1. target_collection: The collection you want to query
    2. search_type: Choose from:
        - "filter_only": Standard SQL style search with filters, sorting, limits, fetching etc.
        - "keyword": Pure text-matching search
    """
    if vectorised:
        output_prompt += """
        - "vector": Semantic search using embeddings
        - "hybrid": Combined keyword and vector search
    """

    output_prompt += """
    3. search_query: The text to search for (optional, not used with filter_only)

    4. Filtering mechanism:
        - filter_buckets: Groups of filters connected with AND/OR logic
        - Each filter_bucket contains:
            - filter: List of property-specific filters or nested filter buckets
            - operator: "AND" or "OR" to connect filters within the bucket

        The FilterBucket is recursive - it can contain other FilterBuckets. This allows you to build complex logical expressions like (A AND B) OR (C AND D) by nesting buckets with different operators. For example:
        - A top-level bucket with OR operator containing:
            - A nested bucket with AND operator (for A AND B)
            - Another nested bucket with AND operator (for C AND D)
        - Example: For "January OR March", create a structure like: OR(AND(Jan-start, Jan-end), AND(Mar-start, Mar-end))

    5. Property filter types:
        - IntegerPropertyFilter: For integer comparisons (=, !=, <, >, <=, >=)
            Note that you cannot filter on integers with float values. 
            If you want to make a comparison between an integer property with a float value,
            you should convert the float value to an integer manually.
        - FloatPropertyFilter: For float comparisons (=, !=, <, >, <=, >=)
            Note that this is for float comparisons, not integer comparisons.
            If the property is an "int", you should use the IntegerPropertyFilter instead.
        - TextPropertyFilter: For text equality (=, !=) or pattern matching (LIKE), or IS_NULL
            When using LIKE, you can use the ? wildcard to match one unknown single character. e.g. "car?" matches "cart", "care", but not "car"
            You can also use the * wildcard to match any number of unknown characters. e.g. "car*" matches "car", "care", "carpet", etc
            "*car*" matches "car", "healthcare", etc
        - BooleanPropertyFilter: For boolean comparisons (=, !=, IS_NULL)
        - DatePropertyFilter: For date comparisons (=, !=, <, >, <=, >=, IS_NULL)
        - ListPropertyFilter: For list comparisons (=, !=, CONTAINS_ANY, CONTAINS_ALL, IS_NULL) (only if the property type is a list, e.g. "text[]", or "number[]")
        Make sure to use the correct filter type for the property type. The property type is given in the schema (field.[field_name].type).
        If you use the incorrect filter type, the query will error. It is extremely important that you use the correct filter type!
        You can also use the `length` parameter on NumberPropertyFilter/ListPropertyFilter ONLY (even if the property is not a number) to filter by the length of a list,
            ONLY if `isLengthIndexed` is set to True in the schema.
            (found in `index_properties.isLengthIndexed` in the schema)
        When `length` is set to True, you are not searching the properties, you are filtering based on the _length_ of the properties
            (if the property is a list, then this is the list length, if the property is a string, then this is the character length)
        Do NOT use the length filter if the property is not indexed for length. It will raise an error!
        You can also use the IS_NULL filter to filter for properties that are None or null (not an empty list or string etc.), ONLY if `isNullIndexed` is set to True in the schema.
            (found in `index_properties.isNullIndexed` in the schema)
        Do NOT use the IS_NULL filter if the property is not indexed for null. It will raise an error!
        You CANNOT use any type of filter on an 'object' or 'object[]' property. It will raise an error!
        Note most filters are INCLUSIVE. You can use <=, <, >, >=, !=, to filter out results. But e.g. LIKE and = are INCLUSIVE.


    6. CreationTimeFilter: For date comparisons for object creation time metadata (=, <, >, <=, >=)
        - This is a special filter that is used to filter by the creation time of an object.
        - You cannot filter individual properties, only the creation time of objects.
        - The value to filter on should be datetime formatted as a string, e.g. "2021-01-01T00:00:00Z"
        - IMPORTANT: This is only achievable if the collection schema has property `isTimestampIndexed` set to True.
            (found in `index_properties.isTimestampIndexed` in the schema)
            Do NOT use this filter if the property is not set to True. It will raise an error!

    7. Additional query parameters:
        - sort_by: Property and direction for sorting results
        - limit: Maximum number of results to return (default: 5)
        - groupby_property: Group results by a specific property value

    When constructing queries:
    1. Identify the relevant collection from the schemas provided
    2. Choose the appropriate search_type based on the user's intent
    3. Build filter_buckets when specific conditions are needed, using nested buckets for logic
    4. Apply sorting and limits as needed
    """

    if vectorised:
        output_prompt += """
    Tips for query_output:
    1.  a. Use 'hybrid' if you are unsure of the search type (but still need a search term), as this combines semantic and keyword search.
        b. Use 'filter_only' if there are no search terms needed.
        c. 'vector' and 'keyword' are good but should be reserved for specific cases where they are needed.
            e.g. keyword is good for looking for when the search terms WILL appear in the text, but do NOT use this as a filter if you can use a field instead
                    vector is good for matching meaning only, and the words themselves are not that important
    """
    else:
        output_prompt += """
    Tips for query_output:
    1.  a. Use 'filter_only' if there are no search terms needed.
        b. 'keyword' should be used as a more fine-grained way of reducing the result set down - it performs BM25 search on the text.
            e.g. keyword is good for looking for when the search terms WILL appear in the text, but do NOT use this as a filter if you can use fields for more direct filtering instead
    """

    output_prompt += """
    2. Err on the side of using larger limits, because you can parse the results later, and it's better to have more than too few.
        Additionally, if this query is part of a larger task, you may need to retrieve a lot of objects to infer correlations or trends.
        If you are looking for overall patterns or trends, use a larger limit (greater than 10).
    3. Do not use a search term to look for specific categories of items if there is a corresponding field in the schema you can use instead as a filter.
    4. You cannot combine sorting with a search term (mutually exclusive), so if sorting is important, do not specify a search term.
    5. Nesting filter buckets is really powerful, and if the queries are particularly complex, definitely use this feature.
    6. Do not be afraid of using lots of features (filters, limits, search terms) at once, as this may be required for bigger queries.
    7. Do not feel the need to answer everything all at once. You are just one step in a larger program. This step can be repeated at a later date.
    8. If you are querying multiple collections for the same thing, do not repeat the same query for each collection, just provide multiple collections a single query_output.
    9. Always make sure your filter matches the property type in the dataset.

    Return None if query is impossible to execute.
    """
    return output_prompt


class QueryCreatorPrompt(dspy.Signature):
    """
    You are a database query builder. Your task is to convert natural language questions into structured queries using the provided schema information.

    Instructions:
    1. Carefully read the user's question and the schema details.
    2. Formulate clear, precise queries that best address the user's request.
    3. If the user's question is missing important search terms, use your knowledge to identify and include what should be searched, even if not explicitly mentioned.
    4. You may create multiple queries for the same collection if needed (e.g., with different search terms or filters). Clearly explain your reasoning for each.
    5. Write all your reasoning in the provided reasoning fields. This will help later models understand your decision process.
    6. Return a list of QueryOutput objects that fully satisfy the user's request. If there are multiple valid approaches, include multiple queries.
    7. Remember: Each query operates on the entire collection, not on a subset filtered by previous steps. Ensure filters and arguments are consistent across all queries.
    8. Do not write any code or define new classes—just provide the correct arguments as specified by the schema.

    The schema is provided in the `collection_schemas` field. This is important to use to understand the data types and properties of the collection.
    Ensure that you choose the correct filter types for the properties in the schema, including values you filter on, they must match the data type of the property.
        E.g., if the property is an integer, do not specify a float such as 1.0, it must be 1.
    Use the groups in the schema to understand the data for filters.

    Be concise, clear, and thorough in your reasoning and query construction.
    """

    advanced_reasoning: str = dspy.OutputField(
        desc="""
        In draft form (no more than 5 words per sentence), provide some extra reasoning and thinking.
        Provide one sentence for each of the following:
            - Whether the collection isNullIndexed, isLengthIndexed, or isTimestampIndexed
                If so, do you need to use `length`, `IS_NULL`, or number_property filters?
            - How the filter buckets are nested, and why you chose this structure.
                This logic is important, what should be an OR and what should be an AND?
            - What type of search to perform, and why you chose this search type.
            - If there are any fields to search on, why you chose these fields.
            - How you are defining the query_output, and that you are providing only the schema, no extra code.
              This includes defining any variables, you cannot do this. EVERYTHING MUST BE IN THE QUERY_OUTPUT (WITHOUT EXTRA DEFINITIONS).
              NO CODE!
        """.strip()
    )
    available_collections: list[str] = dspy.InputField(
        description="A list of collections that are available to query. You can ONLY choose from these collections."
    )
    previous_queries: list[str] = dspy.InputField(
        desc="""
        For each collection, a list of existing code that has been used to query the collection.
        This can be used to avoid generating duplicate queries.
        If any of the collections have an empty list, you are generating the first query for that collection.
        """.strip(),
        format=list[dict],
    )
    collection_display_types: dict[str, list[str]] = dspy.InputField(
        desc="""
        A dictionary of the display types for each collection. Use this to determine the display type in `display_type`, for each collection.
        The output of `display_type` must be one of the values in the list for the dictionary entry for that collection.
        """.strip(),
    )
    display_type_descriptions: dict[str, str] = dspy.InputField(
        desc="""
        Descriptions of each of the available display types.
        Use this to determine your output display type(s).
        The keys are the display types, and the values are the descriptions.
        """.strip(),
        format=dict[str, str],
    )
    searchable_fields: dict[str, list[str]] = dspy.InputField(
        desc="""
        A dictionary of the named vector fields to search for each collection. Use this to determine the field in `vector_to_search`, for each collection.
        IMPORTANT: these are not property names, they are the names of vectors to search on (if using hybrid or vector search).
        The keys are the collection names, and the values are the fields that are possible to perform the search on.
        """.strip(),
        format=dict[str, list[str]],
    )

    fields_to_search: dict[str, list[str] | None] = dspy.OutputField(
        desc="""
        A dictionary of the named vector fields to search for each collection.
        IMPORTANT: these are not property names, they are the names of vectors to search on (if using hybrid or vector search).
        Always only choose a field from the `searchable_fields` list.
        The keys are the collection names, and the values are the fields to search for.
        These come from the `searchable_fields` list. If this list is empty, the value should be None.
        Otherwise, you should choose the field(s) that would be most relevant to the user's query.
        If in doubt, set the value inside the dictionary to None.
        """.strip()
    )
    data_display: dict[str, DataDisplay] = dspy.OutputField(
        desc="""
        Some metadata about the data you will be returning.
        The keys are the collection names, and the values are the DataDisplay objects.
        The `display_type` should match ONE of the display types in the `collection_display_types` for the corresponding collection.
        The `summarise_items` should be True if the user has specifically asked for summaries of the objects (in an itemised fashion, not an overall summary).
        Otherwise, it should be False.
        """.strip()
    )
