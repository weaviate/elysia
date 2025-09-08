from datetime import timezone
from typing import Any, List, Literal, Optional, Union
from typing_extensions import TypeAlias
from typing import get_args, get_origin

import weaviate
from dateutil import parser
from pydantic import BaseModel, Field
from weaviate.collections import CollectionAsync
from weaviate.classes.query import Filter, Metrics, QueryReference, Sort
from weaviate.classes.aggregate import GroupByAggregate
from weaviate.collections.classes.filters import _Filters
from weaviate.collections.classes.grpc import Sorting
from weaviate.outputs.aggregate import AggregateGroupByReturn, AggregateReturn
from weaviate.outputs.query import QueryReturn
from weaviate.exceptions import (
    WeaviateBaseError,
    WeaviateQueryError,
    AuthenticationFailedError,
)

# == Define Pydantic models for structured outputs of LLMs


# -- Filters
class IntegerPropertyFilter(BaseModel):
    property_name: str
    operator: Literal["=", "!=", "<", ">", "<=", ">=", "IS_NULL"]
    value: int | bool
    length: Optional[bool] = False


class FloatPropertyFilter(BaseModel):
    property_name: str
    operator: Literal["=", "!=", "<", ">", "<=", ">=", "IS_NULL"]
    value: float | bool
    length: Optional[bool] = False


class TextPropertyFilter(BaseModel):
    property_name: str
    operator: Literal["=", "!=", "LIKE", "IS_NULL"]
    value: str | bool


class BooleanPropertyFilter(BaseModel):
    property_name: str
    operator: Literal["=", "!=", "IS_NULL"]
    value: bool


class DatePropertyFilter(BaseModel):
    property_name: str
    operator: Literal["=", "!=", "<", ">", "<=", ">=", "IS_NULL"]
    value: str | bool


class ListPropertyFilter(BaseModel):
    property_name: str
    operator: Literal["=", "!=", "CONTAINS_ANY", "CONTAINS_ALL", "IS_NULL"]
    value: list[str | int | float | bool] | bool
    length: Optional[bool] = False


class CreationTimeFilter(BaseModel):
    operator: Literal["=", "<", ">", "<=", ">="]
    value: str


# -- Aggregation Fields
class IntegerAggregation(BaseModel):
    property_name: str
    metrics: List[Literal["MIN", "MAX", "MEAN", "MEDIAN", "MODE", "SUM", "COUNT"]]


class FloatAggregation(BaseModel):
    property_name: str
    metrics: List[Literal["MIN", "MAX", "MEAN", "MEDIAN", "MODE", "SUM", "COUNT"]]


class TextAggregation(BaseModel):
    property_name: str
    metrics: List[Literal["TOP_OCCURRENCES", "COUNT"]]
    min_occurrences: Optional[int] = None


class BooleanAggregation(BaseModel):
    property_name: str
    metrics: List[
        Literal[
            "TOTAL_TRUE", "TOTAL_FALSE", "PERCENTAGE_TRUE", "PERCENTAGE_FALSE", "COUNT"
        ]
    ]


class DateAggregation(BaseModel):
    property_name: str
    metrics: List[Literal["MIN", "MAX", "MEAN", "MEDIAN", "MODE", "COUNT"]]


# -- Define FilterBucket type recursively
class FilterBucket(BaseModel):
    filters: List[
        Union[
            "FilterBucket",
            IntegerPropertyFilter,
            FloatPropertyFilter,
            TextPropertyFilter,
            BooleanPropertyFilter,
            DatePropertyFilter,
            ListPropertyFilter,
            CreationTimeFilter,
        ]
    ]
    operator: Literal["AND", "OR"]


FilterBucketType: TypeAlias = FilterBucket


# == Sorting for fetch objects
class SortBy(BaseModel):
    property_name: str
    direction: Literal["ascending", "descending"]


# == Full Query Definition
class QueryOutput(BaseModel):
    target_collections: List[str]
    search_type: Literal["hybrid", "keyword", "vector", "filter_only"]
    search_query: Optional[str] = None
    sort_by: Optional[SortBy] = None
    filter_buckets: Optional[List[FilterBucket] | FilterBucket] = None
    limit: Optional[int] = 5


class NonVectorisedQueryOutput(QueryOutput):
    search_type: Literal["keyword", "filter_only"]


# -- Full Aggregation Definition
class AggregationOutput(BaseModel):
    target_collections: List[str]
    filter_buckets: Optional[List[FilterBucket] | FilterBucket] = None
    groupby_property: Optional[str] = None
    integer_property_aggregations: Optional[List[IntegerAggregation]] = None
    float_property_aggregations: Optional[List[FloatAggregation]] = None
    text_property_aggregations: Optional[List[TextAggregation]] = None
    boolean_property_aggregations: Optional[List[BooleanAggregation]] = None
    date_property_aggregations: Optional[List[DateAggregation]] = None


class VectorisedAggregationOutput(AggregationOutput):
    search_type: Optional[Literal["vector", "hybrid"]] = Field(
        default=None,
        description=(
            "The type of search to use for the aggregation. If None, no search is used. "
        ),
    )
    search_query: Optional[str] = Field(
        default=None,
        description="The text to search for (using search). Only set this if search_type is specified.",
    )
    limit: Optional[int] = Field(
        default=100,
        description="The number of results to return. Only set this if search_type is specified.",
    )


# -- Schema for collections/data
class FieldSchema(BaseModel):
    type: Literal[
        "text", "float", "boolean", "date", "text[]", "float[]", "boolean[]", "date[]"
    ]
    groups: List[str]
    mean: float


class CollectionSchema(BaseModel):
    name: str
    length: int
    summary: str
    properties: dict[str, FieldSchema]


# -- Additional output fields
class DataDisplay(BaseModel):
    display_type: str
    summarise_items: bool


class QueryError(Exception):
    pass


def _catch_filter_errors(
    filters: list,
    collection_property_types: dict[str, str],
    collection_name: str,
    schema: dict | None = None,
):
    for filter in filters:
        if isinstance(filter, FilterBucket):
            _catch_filter_errors(
                filter.filters, collection_property_types, collection_name, schema
            )
        else:
            # check inverted index
            if schema:
                if (
                    filter.operator == "IS_NULL"
                    and not schema[collection_name]["index_properties"]["isNullIndexed"]
                ):
                    raise QueryError(
                        f"For collection '{collection_name}', attempted to filter on property "
                        f"'{filter.property_name}' using a IS_NULL filter, "
                        "but the property is not indexed for null values, so this filter is unavailable. "
                    )

            # check existence
            if filter.property_name not in collection_property_types:
                if (
                    "." in filter.property_name
                    and filter.property_name.split(".")[0] in collection_property_types
                ):
                    raise QueryError(
                        f"During the filter: {filter}"
                        f"Property '{filter.property_name}' not found in the collection '{collection_name}'. "
                        f"If you are trying to filter on a nested property of '{filter.property_name.split('.')[0]}', "
                        "you cannot do this. Object properties cannot be filtered on. "
                        "This is a Weaviate limitation."
                    )
                raise QueryError(
                    f"During the filter: {filter}"
                    f"Property '{filter.property_name}' not found in the collection '{collection_name}'. "
                    f"Available properties: {list(collection_property_types.keys())}"
                )

            # check if object
            if collection_property_types[filter.property_name].startswith("object"):
                raise QueryError(
                    f"Attempted to filter on property '{filter.property_name}', "
                    "but the property type is an object. "
                    "Object types cannot be filtered on."
                )

            if "length" in filter.model_dump() and filter.length:
                if collection_property_types[filter.property_name].endswith("[]"):
                    pass

                elif (
                    schema
                    and not schema[collection_name]["index_properties"][
                        "isLengthIndexed"
                    ]
                ):
                    raise QueryError(
                        f"For collection '{collection_name}', attempted to filter on property "
                        f"'{filter.property_name}' using a length filter, "
                        "but the property is not indexed for length values, so this filter is unavailable. "
                    )
                continue

            # get the types from the union and extract their origins for isinstance check
            union_list_types = get_args(ListPropertyFilter.__annotations__["value"])
            valid_list_types = tuple(
                get_origin(type_) or type_ for type_ in union_list_types
            )

            if not isinstance(
                filter.value, valid_list_types
            ) and collection_property_types[filter.property_name].endswith("[]"):
                if collection_property_types[filter.property_name] == "text":
                    raise QueryError(
                        f"Attempted to filter on property '{filter.property_name}' using a list filter, "
                        "but the property type is text. Text properties cannot be filtered on. "
                        "Hint: use a text filter with the 'LIKE' operator instead."
                    )
                else:
                    raise QueryError(
                        f"Attempted to filter on property '{filter.property_name}' using a list filter, "
                        "but the property type is not a list. It is a "
                        f"{collection_property_types[filter.property_name]}."
                    )

            # check number
            if (
                not isinstance(
                    filter.value, IntegerPropertyFilter.__annotations__["value"]
                )
                and collection_property_types[filter.property_name] == "int"
            ):
                if collection_property_types[filter.property_name] == "float":
                    raise QueryError(
                        f"Attempted to filter on property '{filter.property_name}' using a integer filter, "
                        f"but the property type is a {collection_property_types[filter.property_name]}. "
                        f"Filter value: {filter.value} is a {type(filter.value)}, not an integer. "
                        f"Hint: you could use {float(filter.value)} instead (may require adjusting the filter operator)."
                    )
                else:
                    raise QueryError(
                        f"Attempted to filter on property '{filter.property_name}' using a integer filter, "
                        f"but the property type is a {collection_property_types[filter.property_name]}. "
                        f"Filter value: {filter.value} is a {type(filter.value)}, not an integer. "
                    )

            # check number
            if (
                not isinstance(
                    filter.value, FloatPropertyFilter.__annotations__["value"]
                )
                and collection_property_types[filter.property_name] == "float"
            ):
                if collection_property_types[filter.property_name] == "int":
                    raise QueryError(
                        f"Attempted to filter on property '{filter.property_name}' using a float filter, "
                        f"but the property type is a {collection_property_types[filter.property_name]}. "
                        f"Filter value: {filter.value} is a {type(filter.value)}, not a float. "
                        f"Hint: you could use {int(filter.value)} instead (may require adjusting the filter operator)."
                    )
                else:
                    raise QueryError(
                        f"Attempted to filter on property '{filter.property_name}' using a float filter, "
                        f"but the property type is a {collection_property_types[filter.property_name]}. "
                        f"Filter value: {filter.value} is a {type(filter.value)}, not a float. "
                    )

            # check date
            if (
                not isinstance(
                    filter.value, DatePropertyFilter.__annotations__["value"]
                )
                and collection_property_types[filter.property_name] == "date"
            ):
                raise QueryError(
                    f"Attempted to filter on property '{filter.property_name}' using a date filter, "
                    f"but the property type is a {collection_property_types[filter.property_name]}. "
                    f"Filter value: {filter.value} is a {type(filter.value)}, not a date."
                )

            # check boolean
            if (
                not isinstance(
                    filter.value, BooleanPropertyFilter.__annotations__["value"]
                )
                and collection_property_types[filter.property_name] == "boolean"
            ):
                raise QueryError(
                    f"Attempted to filter on property '{filter.property_name}' using a boolean filter, "
                    f"but the property type is a {collection_property_types[filter.property_name]}. "
                    f"Filter value: {filter.value} is a {type(filter.value)}, not a boolean."
                )

            # check text
            if (
                not isinstance(
                    filter.value, TextPropertyFilter.__annotations__["value"]
                )
                and collection_property_types[filter.property_name] == "text"
            ):
                raise QueryError(
                    f"Attempted to filter on property '{filter.property_name}' using a text filter, "
                    f"but the property type is a {collection_property_types[filter.property_name]}. "
                    f"Filter value: {filter.value} is a {type(filter.value)}, not a text."
                )


# == Define functions for executing queries and aggregations
def _catch_typing_errors(
    tool_args: dict[str, Any],
    property_types: dict[str, dict[str, str]],
    schema: dict | None = None,
) -> None:
    """Catch typing errors and raise a QueryError."""

    # check search query
    if (
        "search_type" in tool_args
        and tool_args["search_type"] in ["hybrid", "keyword", "vector"]
        and "search_query" not in tool_args
    ):
        raise QueryError("Search query cannot be None for non-filter-only search.")

    if (
        "search_query" in tool_args
        and tool_args["search_query"] is not None
        and ("search_type" not in tool_args or tool_args["search_type"] is None)
    ):
        raise QueryError(
            "A search query was provided but no search type was provided. "
            "You can retry with a search type included. "
        )

    for collection_name, collection_property_types in property_types.items():

        aggregations = [
            arg for arg in list(tool_args.keys()) if arg.endswith("aggregations")
        ]

        if len(aggregations) > 0:
            for aggregation in aggregations:
                for agg_operator in tool_args[aggregation]:

                    if agg_operator.property_name not in collection_property_types:
                        if (
                            "." in agg_operator.property_name
                            and agg_operator.property_name.split(".")[0]
                            in collection_property_types
                        ):
                            raise QueryError(
                                f"During the aggregation '{aggregation}': "
                                f"Property '{agg_operator.property_name}' not found in the collection '{collection_name}'. "
                                f"If you are trying to aggregate on a nested property of '{agg_operator.property_name.split('.')[0]}', "
                                "you cannot do this. Object properties cannot be aggregated on. "
                                "This is a Weaviate limitation."
                            )
                        raise QueryError(
                            f"During the aggregation: {aggregation}, "
                            f"Property '{agg_operator.property_name}' not found in the collection '{collection_name}'. "
                            f"Available properties: {list(collection_property_types.keys())}"
                        )

                    # check if object
                    if collection_property_types[agg_operator.property_name].startswith(
                        "object"
                    ):
                        raise QueryError(
                            f"Attempted to filter on property '{filter.property_name}', "
                            "but the property type is an object. "
                            "Object types cannot be filtered on."
                        )

        if "filter_buckets" in tool_args:
            _catch_filter_errors(
                tool_args["filter_buckets"],
                collection_property_types,
                collection_name,
                schema,
            )


def _catch_weaviate_errors(e: WeaviateBaseError):
    if isinstance(e, WeaviateQueryError):
        if "VectorFromInput was called without vectorizer" in e.message:
            raise QueryError(
                "You are trying to do hybrid or vector search on a collection that has no vectorizer. "
                "You can only perform filter-only or keyword search on this collection. "
            )
        else:
            raise e
    elif isinstance(e, AuthenticationFailedError):
        raise QueryError(
            "Weaviate authentication failed. The user should check their API key and cluster URL or Weaviate connection details."
        )
    else:
        raise e


async def execute_weaviate_query(
    weaviate_client: weaviate.WeaviateAsyncClient,
    predicted_query: QueryOutput,
    property_types: dict[str, dict[str, str]],
    reference_property: str | None = None,
    named_vector_fields: dict[str, list[str]] | None = None,
    schema: dict | None = None,
) -> tuple[list[QueryReturn], list[str]]:
    """Execute from a QueryOutput and return response."""

    # Convert WeaviateQuery to tool args format
    tool_args: dict[str, Any] = {
        "collection_names": predicted_query.target_collections,
    }

    if predicted_query.search_query:
        tool_args["search_query"] = predicted_query.search_query

    if predicted_query.search_type:
        tool_args["search_type"] = predicted_query.search_type

    if predicted_query.filter_buckets:
        if isinstance(predicted_query.filter_buckets, list):
            tool_args["filter_buckets"] = predicted_query.filter_buckets
        elif isinstance(predicted_query.filter_buckets, FilterBucket):
            tool_args["filter_buckets"] = [predicted_query.filter_buckets]

    if predicted_query.sort_by:
        tool_args["sort_by"] = predicted_query.sort_by.model_dump()

    if predicted_query.limit:
        tool_args["limit"] = min(predicted_query.limit, 100)
    else:
        tool_args["limit"] = min(QueryOutput.model_fields["limit"].default, 100)

    if reference_property:
        tool_args["reference_property"] = reference_property

    if named_vector_fields:
        tool_args["named_vector_fields"] = {
            name: fields if fields != [] else None
            for name, fields in named_vector_fields.items()
        }

    _catch_typing_errors(tool_args, property_types, schema)

    try:
        final_responses, str_responses = await _handle_search(
            weaviate_client, tool_args
        )
    except WeaviateBaseError as e:
        _catch_weaviate_errors(e)

    return final_responses, str_responses


async def _handle_search(
    weaviate_client: weaviate.WeaviateAsyncClient, tool_args: dict
) -> tuple[list[QueryReturn], list[str]]:
    """Do vector/keyword/hybrid search from a QueryOutput."""

    collection_names = tool_args["collection_names"]
    collections = [weaviate_client.collections.get(name) for name in collection_names]

    # Build reference property
    if "reference_property" in tool_args:
        reference = QueryReference(
            link_on=tool_args["reference_property"],
            return_properties=[],
        )
    else:
        reference = None

    if "named_vector_fields" in tool_args:
        named_vector_fields = tool_args["named_vector_fields"]
    else:
        named_vector_fields = None

    # Filter and sort
    combined_filter = _build_filters(tool_args)
    sort = _build_sort(tool_args)

    # Loop over collections and execute search
    responses = []
    str_responses = []
    for i, collection in enumerate(collections):
        if tool_args["search_type"] == "filter_only":
            response = await collection.query.fetch_objects(
                limit=tool_args["limit"],
                filters=combined_filter,
                return_references=reference,
                sort=sort,
            )
        else:
            if tool_args["search_type"] == "hybrid":
                response = await collection.query.hybrid(
                    query=tool_args["search_query"],
                    limit=tool_args["limit"],
                    filters=combined_filter,
                    return_references=reference,
                    target_vector=(
                        named_vector_fields[collection.name]
                        if named_vector_fields
                        else None
                    ),
                )

            elif tool_args["search_type"] == "vector":
                response = await collection.query.near_text(
                    query=tool_args["search_query"],
                    limit=tool_args["limit"],
                    filters=combined_filter,
                    return_references=reference,
                    target_vector=(
                        named_vector_fields[collection.name]
                        if named_vector_fields
                        else None
                    ),
                )

            elif tool_args["search_type"] == "keyword":
                response = await collection.query.bm25(
                    query=tool_args["search_query"],
                    limit=tool_args["limit"],
                    filters=combined_filter,
                    return_references=reference,
                )

        str_response = _construct_string_search_query(tool_args, combined_filter)
        responses.append(response)
        str_responses.append(str_response)

    return responses, str_responses


def _build_sort(tool_args: dict) -> Sorting | None:
    """Build a sort object from a QueryOutput."""

    if "sort_by" in tool_args and tool_args["search_type"] == "filter_only":
        return Sort.by_property(
            tool_args["sort_by"]["property_name"],
            ascending=tool_args["sort_by"]["direction"] == "ascending",
        )
    return None


def _build_single_filter(
    filter: (
        IntegerPropertyFilter
        | FloatPropertyFilter
        | TextPropertyFilter
        | BooleanPropertyFilter
        | DatePropertyFilter
        | ListPropertyFilter
        | CreationTimeFilter
    ),
    reference_property: str | None = None,
) -> Filter:
    """Build a Weaviate Filter from a typed Filter object."""

    if isinstance(filter, CreationTimeFilter):
        prop_name = ""
        filter_operator = filter.operator
        filter_value = filter.value
        filter_length = False
    else:
        prop_name = filter.property_name
        filter_operator = filter.operator
        filter_value = filter.value
        if not (
            isinstance(filter, TextPropertyFilter)
            or isinstance(filter, DatePropertyFilter)
            or isinstance(filter, BooleanPropertyFilter)
        ):
            filter_length = filter.length
        else:
            filter_length = False

    if (
        isinstance(filter, FloatPropertyFilter)
        and filter_operator != "IS_NULL"
        and isinstance(filter_value, (int, float, bool))
    ):
        filter_value = float(filter_value)
    elif isinstance(filter, IntegerPropertyFilter) and isinstance(filter_value, float):
        filter_value = int(filter_value)
    elif isinstance(filter, DatePropertyFilter) and isinstance(filter_value, str):
        filter_value = parser.parse(filter_value).replace(tzinfo=timezone.utc)
    elif isinstance(filter, CreationTimeFilter) and isinstance(filter_value, str):
        filter_value = parser.parse(filter_value).replace(tzinfo=timezone.utc)

    if reference_property:
        filter_obj = Filter.by_ref(link_on=reference_property)
    else:
        filter_obj = Filter

    if isinstance(filter, CreationTimeFilter):
        filter_obj = filter_obj.by_creation_time()
    else:
        filter_obj = filter_obj.by_property(
            prop_name, length=filter_length if filter_length else False
        )

    if filter_operator == "=":
        return filter_obj.equal(filter_value)  # type: ignore
    elif filter_operator == "!=":
        return filter_obj.not_equal(filter_value)  # type: ignore
    elif filter_operator == ">":
        return filter_obj.greater_than(filter_value)  # type: ignore
    elif filter_operator == "<":
        return filter_obj.less_than(filter_value)  # type: ignore
    elif filter_operator == ">=":
        return filter_obj.greater_or_equal(filter_value)  # type: ignore
    elif filter_operator == "<=":
        return filter_obj.less_or_equal(filter_value)  # type: ignore
    elif filter_operator == "LIKE":
        return filter_obj.like(filter_value)  # type: ignore
    elif filter_operator == "CONTAINS_ANY":
        return filter_obj.contains_any(filter_value)  # type: ignore
    elif filter_operator == "CONTAINS_ALL":
        return filter_obj.contains_all(filter_value)  # type: ignore
    elif filter_operator == "IS_NULL":
        return filter_obj.is_none(filter_value)  # type: ignore
    else:
        raise ValueError(f"Invalid filter operator: {filter_operator}")


def _build_filter_bucket(
    filter_bucket: FilterBucket, reference_collection: str | None = None
) -> _Filters:
    """
    Build a collection of individual filters from a FilterBucket.
    Recursively build the filters if it finds more FilterBuckets.
    """
    bucket_operator = filter_bucket.operator
    filters = []
    for filter in filter_bucket.filters:
        if isinstance(filter, FilterBucket):
            filters.append(_build_filter_bucket(filter))
        elif (
            isinstance(filter, IntegerPropertyFilter)
            or isinstance(filter, FloatPropertyFilter)
            or isinstance(filter, TextPropertyFilter)
            or isinstance(filter, BooleanPropertyFilter)
            or isinstance(filter, DatePropertyFilter)
            or isinstance(filter, ListPropertyFilter)
            or isinstance(filter, CreationTimeFilter)
        ):
            filters.append(_build_single_filter(filter, reference_collection))

    if bucket_operator == "AND":
        return Filter.all_of(filters)
    elif bucket_operator == "OR":
        return Filter.any_of(filters)
    else:
        raise ValueError(f"Invalid bucket operator: {bucket_operator}")


def _build_filters(tool_args: dict) -> _Filters | None:
    """
    Build all filters from a QueryOutput.
    """

    if "filter_buckets" not in tool_args:
        return None

    filters = []

    for bucket in tool_args["filter_buckets"]:
        bucket_filters = _build_filter_bucket(bucket)
        filters.append(bucket_filters)

    return Filter.all_of(filters)


# -- Aggregation
async def execute_weaviate_aggregation(
    weaviate_client: weaviate.WeaviateAsyncClient,
    predicted_query: AggregationOutput,
    property_types: dict[str, dict[str, str]],
    vectorised_by_collection: dict[str, bool],
    schema: dict | None = None,
) -> tuple[list[AggregateReturn], list[str]]:
    """Execute a predicted WeaviateAggregation and return formatted results."""

    # Convert WeaviateAggregation to tool args format
    tool_args: dict[str, Any] = {
        "collection_names": predicted_query.target_collections,
    }

    if isinstance(predicted_query, VectorisedAggregationOutput):
        if predicted_query.search_query:
            tool_args["search_query"] = predicted_query.search_query

        if predicted_query.search_type:
            tool_args["search_type"] = predicted_query.search_type

        if predicted_query.limit:
            tool_args["limit"] = predicted_query.limit
        else:
            tool_args["limit"] = VectorisedAggregationOutput.model_fields[
                "limit"
            ].default

    if predicted_query.filter_buckets:
        if isinstance(predicted_query.filter_buckets, list):
            tool_args["filter_buckets"] = predicted_query.filter_buckets
        elif isinstance(predicted_query.filter_buckets, FilterBucket):
            tool_args["filter_buckets"] = [predicted_query.filter_buckets]

    if predicted_query.integer_property_aggregations:
        tool_args["integer_property_aggregations"] = (
            predicted_query.integer_property_aggregations
        )

    if predicted_query.float_property_aggregations:
        tool_args["float_property_aggregations"] = (
            predicted_query.float_property_aggregations
        )

    if predicted_query.text_property_aggregations:
        tool_args["text_property_aggregations"] = (
            predicted_query.text_property_aggregations
        )

    if predicted_query.boolean_property_aggregations:
        tool_args["boolean_property_aggregations"] = (
            predicted_query.boolean_property_aggregations
        )

    if predicted_query.date_property_aggregations:
        tool_args["date_property_aggregations"] = (
            predicted_query.date_property_aggregations
        )

    if predicted_query.groupby_property:
        tool_args["groupby_property"] = predicted_query.groupby_property

    _catch_typing_errors(tool_args, property_types, schema)

    # Execute query based on type
    try:
        responses, str_responses = await _handle_aggregation_query(
            weaviate_client, tool_args, vectorised_by_collection
        )
    except WeaviateBaseError as e:
        _catch_weaviate_errors(e)

    return responses, str_responses


async def _handle_aggregation_query(
    weaviate_client: weaviate.WeaviateAsyncClient,
    tool_args: dict,
    vectorised_by_collection: dict[str, bool],
) -> tuple[list[AggregateReturn], list[str]]:
    collection_names = tool_args["collection_names"]
    collections = [weaviate_client.collections.get(name) for name in collection_names]

    agg_args = _build_aggregation_args(tool_args)
    combined_filter = _build_filters(tool_args)
    responses = []
    str_responses = []

    for collection in collections:
        if (
            "search_query" in tool_args
            and tool_args["search_query"]
            and "search_type" in tool_args
            and tool_args["search_type"]
            and vectorised_by_collection[collection.name]
        ):
            responses.append(
                await _execute_aggregation_with_search(
                    collection, tool_args, agg_args, combined_filter
                )
            )
            str_responses.append(
                _get_string_aggregation_with_search(tool_args, combined_filter)
            )
        else:
            responses.append(
                await _execute_aggregation_over_all(
                    collection, agg_args, combined_filter
                )
            )
            str_responses.append(
                _get_string_aggregation_over_all(tool_args, combined_filter)
            )

    return responses, str_responses


def _build_return_metrics(tool_args: dict) -> list[Metrics] | None:
    metrics_types = [
        "integer_property_aggregations",
        "float_property_aggregations",
        "text_property_aggregations",
        "boolean_property_aggregations",
        "date_property_aggregations",
    ]

    full_metrics = []
    for agg_type in metrics_types:
        if agg_type in tool_args:
            for agg in tool_args[agg_type]:
                prop_name = agg.property_name
                metrics = [m.upper() for m in agg.metrics]
                if prop_name:
                    if agg_type.startswith("integer"):
                        # Map to correct integer metric names
                        metric_mapping = {
                            "MEAN": "mean",
                            "SUM": "sum_",
                            "MAX": "maximum",
                            "MIN": "minimum",
                            "COUNT": "count",
                        }
                        metric_names = [
                            metric_mapping.get(m, m.lower()) for m in metrics  # type: ignore
                        ]
                        full_metrics.append(
                            Metrics(prop_name).integer(
                                **{metric_name: True for metric_name in metric_names}
                            )
                        )

                    elif agg_type.startswith("float"):
                        # Map to correct integer metric names
                        metric_mapping = {
                            "MEAN": "mean",
                            "SUM": "sum_",
                            "MAX": "maximum",
                            "MIN": "minimum",
                            "COUNT": "count",
                        }
                        metric_names = [
                            metric_mapping.get(m, m.lower()) for m in metrics  # type: ignore
                        ]
                        full_metrics.append(
                            Metrics(prop_name).number(
                                **{metric_name: True for metric_name in metric_names}
                            )
                        )

                    elif agg_type.startswith("text"):
                        # Map to correct text metric names
                        metric_mapping = {
                            "COUNT": ["count"],
                            "TOP_OCCURRENCES": [
                                "top_occurrences_count",
                                "top_occurrences_value",
                            ],
                        }
                        metric_names = [
                            metric_name
                            for m in metrics
                            for metric_name in metric_mapping.get(m, [m.lower()])
                        ]

                        if (
                            any(m.startswith("top_occurrences") for m in metric_names)
                            and agg.min_occurrences
                        ):
                            full_metrics.append(
                                Metrics(prop_name).text(
                                    limit=agg.min_occurrences,
                                    **{
                                        metric_name: True
                                        for metric_name in metric_names
                                    },
                                )
                            )
                        else:
                            full_metrics.append(
                                Metrics(prop_name).text(
                                    **{
                                        metric_name: True
                                        for metric_name in metric_names
                                    }
                                )
                            )

                    elif agg_type.startswith("boolean"):
                        metric_mapping = {
                            "PERCENTAGE_TRUE": "percentage_true",
                            "PERCENTAGE_FALSE": "percentage_false",
                            "TOTAL_TRUE": "total_true",
                            "TOTAL_FALSE": "total_false",
                            "COUNT": "count",
                        }
                        metric_names: list[str] = [
                            metric_mapping.get(m, m.lower()) for m in metrics.keys()  # type: ignore
                        ]
                        return Metrics(prop_name).boolean(
                            **{metric_name: True for metric_name in metric_names}  # type: ignore
                        )

                    elif agg_type.startswith("date"):
                        # Map to correct date metric names
                        metric_mapping = {
                            "MEAN": "mean",
                            "MAX": "maximum",
                            "MIN": "minimum",
                            "MEDIAN": "median",
                            "MODE": "mode",
                            "COUNT": "count",
                        }
                        metric_names = [
                            metric_mapping.get(m, m.lower()) for m in metrics  # type: ignore
                        ]
                        full_metrics.append(
                            Metrics(prop_name).date_(
                                **{metric_name: True for metric_name in metric_names}
                            )
                        )

    return None if full_metrics == [] else full_metrics


def _build_aggregation_args(tool_args: dict) -> dict:
    agg_args: dict = {"total_count": True}

    if "groupby_property" in tool_args:
        agg_args["group_by"] = GroupByAggregate(
            prop=tool_args["groupby_property"],
            limit=200,
        )

    agg_args["return_metrics"] = _build_return_metrics(tool_args)

    return agg_args


async def _execute_aggregation_with_search(
    collection: CollectionAsync,
    tool_args: dict,
    agg_args: dict,
    combined_filter: _Filters | None,
) -> AggregateReturn | AggregateGroupByReturn:
    if tool_args["search_type"] == "hybrid" or tool_args["search_type"] is None:
        search = collection.aggregate.hybrid
    elif tool_args["search_type"] == "vector":
        search = collection.aggregate.near_text
    else:
        raise ValueError(f"Invalid search type: {tool_args['search_type']}")

    return await search(
        query=tool_args["search_query"],
        object_limit=tool_args["limit"],
        total_count=agg_args["total_count"],
        group_by=agg_args.get("group_by"),
        return_metrics=agg_args["return_metrics"],
        filters=combined_filter,
    )


async def _execute_aggregation_over_all(
    collection: CollectionAsync,
    agg_args: dict,
    combined_filter: _Filters | None,
) -> AggregateReturn | AggregateGroupByReturn:
    return await collection.aggregate.over_all(
        total_count=agg_args["total_count"],
        group_by=agg_args.get("group_by"),
        return_metrics=agg_args.get("return_metrics"),
        filters=combined_filter,
    )


# == Strings for code display
# Repeats of the same functions above but string output
def _build_single_filter_string(
    filter: (
        IntegerPropertyFilter
        | FloatPropertyFilter
        | TextPropertyFilter
        | BooleanPropertyFilter
        | DatePropertyFilter
        | ListPropertyFilter
        | CreationTimeFilter
    ),
) -> str:

    if isinstance(filter, CreationTimeFilter):
        filter_operator = filter.operator
        filter_value = filter.value
    else:
        prop_name = filter.property_name
        filter_operator = filter.operator
        filter_value = filter.value
        if not (
            isinstance(filter, TextPropertyFilter)
            or isinstance(filter, DatePropertyFilter)
            or isinstance(filter, BooleanPropertyFilter)
        ):
            filter_length = filter.length
        else:
            filter_length = False

    # Format value based on type
    if isinstance(filter_value, str):
        formatted_value = f"'{filter_value}'"
    else:
        formatted_value = str(filter_value)

    # Map operators to their method names
    operator_map = {
        "=": "equal",
        "!=": "not_equal",
        ">": "greater_than",
        "<": "less_than",
        ">=": "greater_or_equal",
        "<=": "less_or_equal",
        "LIKE": "like",
        "CONTAINS_ANY": "contains_any",
        "CONTAINS_ALL": "contains_all",
        "IS_NULL": "is_none",
    }

    method_name = operator_map.get(filter_operator)
    if method_name is None:
        raise ValueError(f"Invalid filter operator: {filter_operator}")

    if isinstance(filter, CreationTimeFilter):
        start_string = "Filter.by_creation_time()"
    else:
        start_string = f"Filter.by_property('{prop_name}'"
        if filter_length:
            start_string += ", length=True"
        start_string += ")"

    return f"{start_string}.{method_name}({formatted_value})"


def _build_filter_bucket_string(filter_bucket: FilterBucket) -> str:
    bucket_operator = filter_bucket.operator
    filter_strings = []

    for filter in filter_bucket.filters:
        if isinstance(filter, FilterBucket):
            filter_strings.append(_build_filter_bucket_string(filter))
        elif (
            isinstance(filter, IntegerPropertyFilter)
            or isinstance(filter, FloatPropertyFilter)
            or isinstance(filter, TextPropertyFilter)
            or isinstance(filter, BooleanPropertyFilter)
            or isinstance(filter, DatePropertyFilter)
            or isinstance(filter, ListPropertyFilter)
            or isinstance(filter, CreationTimeFilter)
        ):
            filter_strings.append(_build_single_filter_string(filter))

    indented_filters = ",\n    ".join(filter_strings)

    if bucket_operator == "AND":
        return f"Filter.all_of([\n    {indented_filters}\n])"
    elif bucket_operator == "OR":
        return f"Filter.any_of([\n    {indented_filters}\n])"
    else:
        raise ValueError(f"Invalid bucket operator: {bucket_operator}")


def _build_filter_string(tool_args: dict) -> str:
    if "filter_buckets" not in tool_args:
        return "None"

    filter_strings = []

    for bucket in tool_args["filter_buckets"]:
        bucket_filter_string = _build_filter_bucket_string(bucket)
        filter_strings.append(bucket_filter_string)

    if len(filter_strings) > 1:
        indented_filters = ",\n    ".join(filter_strings)
        return f"Filter.all_of([\n    {indented_filters}\n])"
    elif len(filter_strings) == 1:
        return filter_strings[0]
    else:
        return "None"


def _construct_string_search_query(
    tool_args: dict, combined_filter: _Filters | None
) -> str:
    if tool_args["search_type"] == "hybrid":
        search_type = "hybrid"
    elif tool_args["search_type"] == "keyword":
        search_type = "bm25"
    elif tool_args["search_type"] == "vector":
        search_type = "near_text"
    elif tool_args["search_type"] == "filter_only":
        search_type = "fetch_objects"

    query_str = (
        f"query='{tool_args['search_query']}',\n    "
        if "search_query" in tool_args and tool_args["search_query"] is not None
        else ""
    )

    # Create filter string with proper indentation
    filter_string = (
        _build_filter_string(tool_args) if combined_filter is not None else ""
    )
    if filter_string and filter_string != "None":
        # If filter string contains newlines, add additional indentation
        if "\n" in filter_string:
            # Add 4 spaces to each line for proper indentation
            indented_filter = filter_string.replace("\n", "\n    ")
            filter_str = f"filters={indented_filter},\n    "
        else:
            filter_str = f"filters={filter_string},\n    "
    else:
        filter_str = ""

    limit_str = (
        f"limit={tool_args['limit']},\n    "
        if "limit" in tool_args and tool_args["limit"] is not None
        else ""
    )

    # Handle sort with proper indentation
    if "sort_by" in tool_args and tool_args["sort_by"] is not None:
        sort_str = (
            f"sort=Sort.by_property(\n"
            f"        '{tool_args['sort_by']['property_name']}',\n"
            f"        ascending={tool_args['sort_by']['direction'] == 'ascending'}\n"
            f"    )"
        )
    else:
        sort_str = ""

    # Remove trailing comma and whitespace from the last parameter
    params = f"{query_str}{filter_str}{limit_str}{sort_str}"
    if params.endswith(",\n    "):
        params = params[:-6]

    str_response = f"""collection.query.{search_type}(
    {params}
)"""
    return str_response


# -- Aggregation Strings
def _get_string_aggregation_with_search(
    tool_args: dict, combined_filter: _Filters | None
) -> str:
    metrics_str = _build_return_metrics_string(tool_args)

    if tool_args["search_type"] == "hybrid":
        search = "collection.aggregate.hybrid"
    elif tool_args["search_type"] == "vector":
        search = "collection.aggregate.near_text"

    params = []
    if "search_query" in tool_args and tool_args["search_query"]:
        params.append(f"query='{tool_args['search_query']}'")

    if "limit" in tool_args and tool_args["limit"]:
        params.append(f"object_limit={tool_args['limit']}")

    params.append("total_count=True")

    if "groupby_property" in tool_args and tool_args["groupby_property"]:
        params.append(
            f"group_by=GroupByAggregate(prop='{tool_args['groupby_property']}')"
        )

    # Create filter string with proper indentation
    filter_string = (
        _build_filter_string(tool_args) if combined_filter is not None else ""
    )
    if filter_string and filter_string != "None":
        if "\n" in filter_string:
            indented_filter = filter_string.replace("\n", "\n    ")
            params.append(f"filters={indented_filter}")
        else:
            params.append(f"filters={filter_string}")

    if metrics_str != "None":
        params.append(f"return_metrics={metrics_str}\n    ")

    # Format with proper indentation
    formatted_params = ",\n    ".join(params)

    return f"""{search}(
    {formatted_params}
)"""


def _get_string_aggregation_over_all(
    tool_args: dict, combined_filter: _Filters | None
) -> str:
    metrics_str = _build_return_metrics_string(tool_args)

    params = ["total_count=True"]

    if "groupby_property" in tool_args and tool_args["groupby_property"]:
        params.append(
            f"group_by=GroupByAggregate(prop='{tool_args['groupby_property']}')"
        )

    filter_string = (
        _build_filter_string(tool_args) if combined_filter is not None else ""
    )
    if filter_string and filter_string != "None":
        if "\n" in filter_string:
            indented_filter = filter_string.replace("\n", "\n    ")
            params.append(f"filters={indented_filter}")
        else:
            params.append(f"filters={filter_string}")

    if metrics_str != "None":
        params.append(f"return_metrics={metrics_str}")

    # Format with proper indentation
    formatted_params = ",\n    ".join(params)

    return f"""collection.aggregate.over_all(
    {formatted_params}
)"""


def _build_return_metrics_string(tool_args: dict) -> str:
    metrics_types = [
        "integer_property_aggregations",
        "float_property_aggregations",
        "text_property_aggregations",
        "boolean_property_aggregations",
        "date_property_aggregations",
    ]

    metric_strs = []
    for agg_type in metrics_types:
        if agg_type in tool_args:
            for agg in tool_args[agg_type]:
                prop_name = agg.property_name
                metrics = [m.upper() for m in agg.metrics]

                if not prop_name:
                    continue

                if agg_type.startswith("integer"):
                    # Map to correct integer metric names
                    metric_mapping = {
                        "MEAN": "mean",
                        "SUM": "sum_",
                        "MAX": "maximum",
                        "MIN": "minimum",
                        "COUNT": "count",
                        "MEDIAN": "median",
                        "MODE": "mode",
                    }
                    metric_names = [metric_mapping.get(m, m.lower()) for m in metrics]
                    metric_bools = [f"{m}=True" for m in metric_names]
                    metric_strs.append(
                        f"Metrics('{prop_name}').integer({', '.join(metric_bools)})"
                    )

                elif agg_type.startswith("float"):
                    # Map to correct float metric names
                    metric_mapping = {
                        "MEAN": "mean",
                        "SUM": "sum_",
                        "MAX": "maximum",
                        "MIN": "minimum",
                        "COUNT": "count",
                        "MEDIAN": "median",
                        "MODE": "mode",
                    }
                    metric_names = [metric_mapping.get(m, m.lower()) for m in metrics]
                    metric_bools = [f"{m}=True" for m in metric_names]
                    metric_strs.append(
                        f"Metrics('{prop_name}').number({', '.join(metric_bools)})"
                    )

                elif agg_type.startswith("text"):
                    # Map to correct text metric names
                    metric_mapping = {
                        "COUNT": ["count"],
                        "TOP_OCCURRENCES": [
                            "top_occurrences_count",
                            "top_occurrences_value",
                        ],
                    }
                    metric_names = [
                        metric_name
                        for m in metrics
                        for metric_name in metric_mapping.get(m, m.lower())  # type: ignore
                    ]
                    metric_bools = [f"{m}=True" for m in metric_names]
                    limit_part = ""
                    if (
                        "min_occurrences" in tool_args[agg_type]
                        and tool_args[agg_type].min_occurrences
                    ):
                        limit_part = f", limit={tool_args[agg_type].min_occurrences}"

                    metric_strs.append(
                        f"Metrics('{prop_name}').text({', '.join(metric_bools)}{limit_part})"
                    )

                elif agg_type.startswith("boolean"):
                    metric_mapping = {
                        "PERCENTAGE_TRUE": "percentage_true",
                        "PERCENTAGE_FALSE": "percentage_false",
                        "TOTAL_TRUE": "total_true",
                        "TOTAL_FALSE": "total_false",
                        "COUNT": "count",
                    }
                    metric_names = [metric_mapping.get(m, m.lower()) for m in metrics]
                    metric_bools = [f"{m}=True" for m in metric_names]
                    metric_strs.append(
                        f"Metrics('{prop_name}').boolean({', '.join(metric_bools)})"
                    )

                elif agg_type.startswith("date"):
                    # Map to correct date metric names
                    metric_mapping = {
                        "MEAN": "mean",
                        "MAX": "maximum",
                        "MIN": "minimum",
                        "MEDIAN": "median",
                        "MODE": "mode",
                    }
                    metric_names = [metric_mapping.get(m, m.lower()) for m in metrics]
                    metric_bools = [f"{m}=True" for m in metric_names]
                    metric_strs.append(
                        f"Metrics('{prop_name}').date({', '.join(metric_bools)})"
                    )

    if metric_strs != []:
        return "[\n        " + ",\n        ".join(metric_strs) + "\n    ]"
    else:
        return "None"
