import datetime
import json
import uuid
from typing import Any
from types import GenericAlias

from weaviate.collections.classes.aggregate import (
    AggregateDate,
    AggregateGroupByReturn,
    AggregateNumber,
    AggregateReturn,
    AggregateText,
)


def objects_dict_to_str(objects: list) -> str:
    out = ""
    for item in objects:
        if isinstance(item, dict):
            inner = "["
            for key, value in item.items():
                try:
                    inner += json.dumps({key: value}) + ", "
                except TypeError:
                    pass
            inner += "]"
            out += inner + "\n"

        elif isinstance(item, list):
            out += objects_dict_to_str(item)

        elif isinstance(item, str):
            out += item + "\n"

    return out


def format_datetime(dt: datetime.datetime | None) -> str:
    if dt is None:
        return ""

    output = dt.isoformat("T")
    plus = output.find("+")
    if plus != -1:
        return output[:plus] + "Z"
    else:
        return output + "Z"


def format_dict_to_serialisable(d: dict[str, Any], remove_unserialisable: bool = False):
    if remove_unserialisable:
        keys_to_remove = []

    for key, value in d.items():
        if isinstance(value, datetime.datetime):
            d[key] = format_datetime(value)

        elif isinstance(value, dict):
            format_dict_to_serialisable(value, remove_unserialisable)

        elif isinstance(value, list):
            items_to_remove = []
            for i, item in enumerate(value):
                if isinstance(item, dict):
                    format_dict_to_serialisable(item, remove_unserialisable)

                elif isinstance(item, datetime.datetime):
                    d[key][i] = format_datetime(item)

                elif isinstance(item, uuid.UUID):
                    d[key][i] = str(item)

                elif remove_unserialisable and not isinstance(
                    item, (str, int, float, bool, list, dict)
                ):
                    items_to_remove.append(i)

            for index in sorted(items_to_remove, reverse=True):
                del d[key][index]

        elif isinstance(value, uuid.UUID):
            d[key] = str(value)

        elif isinstance(value, type):
            d[key] = value.__name__

        elif isinstance(value, GenericAlias):
            d[key] = str(value)

        elif remove_unserialisable and not isinstance(
            d, (str, int, float, bool, list, dict)
        ):
            keys_to_remove.append(key)

    if remove_unserialisable:
        for key in keys_to_remove:
            del d[key]


def remove_whitespace(text: str) -> str:
    return " ".join(text.split())


def _extract_aggregate_values(prop, aggregations: list[tuple[str, str]]) -> list[dict]:
    """Extract non-None aggregation values from a property."""
    values = []
    for attr_name, agg_name in aggregations:
        value = getattr(prop, attr_name, None)
        if value is not None:
            values.append({"value": value, "field": None, "aggregation": agg_name})
    return values


def format_aggregation_property(prop):
    if isinstance(prop, AggregateText):
        return {
            "type": "text",
            "values": [
                {"value": occ.count, "field": occ.value, "aggregation": "count"}
                for occ in prop.top_occurrences
            ],
        }

    if isinstance(prop, AggregateNumber):
        aggregations = [
            ("count", "count"),
            ("maximum", "maximum"),
            ("mean", "mean"),
            ("median", "median"),
            ("minimum", "minimum"),
            ("mode", "mode"),
            ("sum_", "sum"),
        ]
        return {"type": "number", "values": _extract_aggregate_values(prop, aggregations)}

    if isinstance(prop, AggregateDate):
        aggregations = [
            ("count", "count"),
            ("maximum", "maximum"),
            ("median", "median"),
            ("minimum", "minimum"),
            ("mode", "mode"),
        ]
        return {"type": "date", "values": _extract_aggregate_values(prop, aggregations)}

    return {"type": "unknown", "values": []}


def format_aggregation_response(response):
    out = {}
    if isinstance(response, AggregateGroupByReturn):

        for result in response.groups:
            field = result.grouped_by.prop
            out[field] = {"type": "text", "values": []}

        for result in response.groups:
            field = result.grouped_by.prop

            if result.total_count is not None:
                out[field]["values"].append(
                    {
                        "value": result.total_count,
                        "field": result.grouped_by.value,
                        "aggregation": "count",
                    }
                )

            for key, prop in result.properties.items():
                formatted_props = format_aggregation_property(prop)
                if "groups" not in out[field]:
                    out[field]["groups"] = {}

                if result.grouped_by.value not in out[field]["groups"]:
                    out[field]["groups"][result.grouped_by.value] = {}

                out[field]["groups"][result.grouped_by.value][key] = formatted_props

    elif isinstance(response, AggregateReturn):

        for field, field_properties in response.properties.items():
            out[field] = {}
            props = format_aggregation_property(field_properties)
            for key, prop in props.items():
                out[field][key] = prop

    return out
