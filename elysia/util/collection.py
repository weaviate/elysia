import ast
import datetime
from typing import Any, List

from weaviate.classes.config import DataType
from weaviate.classes.query import Filter, Sort
from weaviate.types import UUID

from elysia.util.parsing import format_datetime

data_mapping = {
    "text": DataType.TEXT,
    "int": DataType.INT,
    "number": DataType.NUMBER,
    "bool": DataType.BOOL,
    "date": DataType.DATE,
    "object": DataType.OBJECT,
    "text[]": DataType.TEXT_ARRAY,
    "int[]": DataType.INT_ARRAY,
    "number[]": DataType.NUMBER_ARRAY,
    "bool[]": DataType.BOOL_ARRAY,
    "date[]": DataType.DATE_ARRAY,
    "object[]": DataType.OBJECT_ARRAY,
}


async def retrieve_all_collection_names(client):
    all_collections = await client.collections.list_all()
    return [
        collection
        for collection in all_collections
        if not collection.startswith("ELYSIA_")
    ]


def get_collection_data_types(client, collection_name: str):
    properties = client.collections.get(collection_name).config.get().properties
    return {property.name: property.data_type[:] for property in properties}


async def async_get_collection_data_types(client, collection_name: str):
    config = client.collections.get(collection_name).config
    config = await config.get()
    properties = config.properties
    return {property.name: property.data_type[:] for property in properties}


def get_collection_weaviate_data_types(client, collection_name: str):
    data_types = get_collection_data_types(client, collection_name)
    return {k: data_mapping[v] for k, v in data_types.items()}


async def async_get_collection_weaviate_data_types(client, collection_name: str):
    data_types = await async_get_collection_data_types(client, collection_name)
    return {k: data_mapping[v] for k, v in data_types.items()}


SERIALIZABLE_TYPES = (str, list, dict, float, int, bool)


def _convert_value(value: Any) -> Any:
    """Convert a single value to a serializable format."""
    if isinstance(value, str) and value.startswith("[") and value.endswith("]"):
        return ast.literal_eval(value)
    if isinstance(value, datetime.datetime):
        return format_datetime(value)
    if isinstance(value, UUID):
        return str(value)
    if isinstance(value, dict):
        return convert_weaviate_object(value)
    if isinstance(value, list):
        return convert_weaviate_list(value)
    if not isinstance(value, SERIALIZABLE_TYPES):
        return str(value)
    return value


def convert_weaviate_list(list_object: list) -> list:
    """Recursively convert list items to serializable formats."""
    return [_convert_value(value) for value in list_object]


def convert_weaviate_object(dict_object: dict) -> dict:
    """Recursively convert dict values to serializable formats."""
    return {key: _convert_value(value) for key, value in dict_object.items()}


async def paginated_collection(
    client,
    collection_name: str,
    query: str = "",
    sort_on: str | None = None,
    ascending: bool = False,
    filter_config: dict[
        str, Any
    ] = {},  # e.g. {"type":"all", "filters": [{"field":"issue_state", "operator":"not_equal", "value":"open"}]}
    page_size: int = 10,
    page_number: int = 1,
):
    collection = client.collections.get(collection_name)

    if (page_size * (page_number - 1) + page_size) > 99_999:
        raise ValueError(
            "Page size exceeds Weaviate's limit of 100,000 objects for using offset."
        )

    filter_type = filter_config.get("type", "all")
    filters_list = filter_config.get("filters", [])
    filters = [f["field"] for f in filters_list]
    filter_operators = [f["operator"] for f in filters_list]
    filter_values = [f["value"] for f in filters_list]

    if len(filters) == 0:

        if query != "":
            response = await collection.query.bm25(
                query=query,
                limit=page_size,
                offset=page_size * (page_number - 1),
            )
        elif sort_on is not None:
            response = await collection.query.fetch_objects(
                sort=Sort.by_property(name=sort_on, ascending=ascending),
                limit=page_size,
                offset=page_size * (page_number - 1),
            )
        else:
            response = await collection.query.fetch_objects(
                limit=page_size, offset=page_size * (page_number - 1)
            )

    else:
        # Build filters using operator lookup
        operator_methods = {
            "equal": "equal",
            "greater_or_equal": "greater_or_equal",
            "greater_than": "greater_than",
            "less_or_equal": "less_or_equal",
            "less_than": "less_than",
            "not_equal": "not_equal",
        }

        all_filters = []
        for filter_name, filter_operator, filter_value in zip(
            filters, filter_operators, filter_values
        ):
            if filter_operator not in operator_methods:
                raise ValueError(f"Invalid filter operator: {filter_operator}")
            prop_filter = Filter.by_property(filter_name)
            method = getattr(prop_filter, operator_methods[filter_operator])
            all_filters.append(method(filter_value))

        if filter_type == "all":
            main_filter = Filter.all_of(all_filters)
        elif filter_type == "any":
            main_filter = Filter.any_of(all_filters)
        else:
            raise ValueError(f"Invalid filter type: {filter_type}")

        if query != "":
            response = await collection.query.bm25(
                query=query,
                filters=main_filter,
                limit=page_size,
                offset=page_size * (page_number - 1),
            )
        elif sort_on is not None:
            response = await collection.query.fetch_objects(
                filters=main_filter,
                sort=Sort.by_property(name=sort_on, ascending=ascending),
                limit=page_size,
                offset=page_size * (page_number - 1),
            )
        else:
            response = await collection.query.fetch_objects(
                filters=main_filter,
                limit=page_size,
                offset=page_size * (page_number - 1),
            )

    objects = [
        {
            **convert_weaviate_object(o.properties),
            "uuid": str(o.uuid),
        }
        for o in response.objects
    ]

    return objects
