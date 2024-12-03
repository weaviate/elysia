import json
import datetime

from weaviate.collections.classes.aggregate import (
    GroupByAggregate,
    AggregateGroup,
    AggregateReturn,
    AggregateGroupByReturn,
    AggregateText,
    AggregateNumber,
    AggregateBoolean,
    AggregateDate,
    AggregateInteger
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

def format_datetime(dt: datetime.datetime) -> str:
    dt = dt.isoformat("T")
    return dt[:dt.find("+")] + "Z"

def remove_whitespace(text: str) -> str:
    return " ".join(text.split())


# def update_current_message(current_message: str, new_message: str) -> str:
#     """
#     Given a current message and a new message, return the current message updated with the new message and the update itself.
#     """

#     if current_message == "":
#         return new_message, new_message

#     if new_message.startswith(current_message):
#         message_update = new_message[len(current_message):]
#     else:
#         message_update = new_message
    
#     if current_message.endswith(".") or current_message.endswith("?") or current_message.endswith("!"):
#         full_message = current_message + " " + message_update
#     else:
#         full_message = current_message + ". " + message_update
        
#     return full_message, message_update

def backup_update_current_message(current_message: str, new_message: str):
    
    if current_message == "":
        return new_message, new_message

    sentences = new_message.split(". ")

    if len(sentences) == len(current_message.split(". ")):
        return current_message, ""

    return current_message + " " + sentences[-1], sentences[-1]

def update_current_message(current_message: str, new_message: str):
    # find new sentence
    if "<NEW>" not in new_message:
        return backup_update_current_message(current_message, new_message)
    
    new_sentence = new_message.split("<NEW>")[1]
    new_sentence = new_sentence[:new_sentence.find("</NEW>")]

    return current_message + " " + new_sentence, new_sentence

def format_aggregation_property(prop):

    if isinstance(prop, AggregateText):
        out = {"type": "text", "values": [], "groups": {}}
        for top_occurence in prop.top_occurrences:
            out["values"].append({"value": top_occurence.count, "field": top_occurence.value, "aggregation": "count"})
        return out
    
    elif isinstance(prop, AggregateNumber):
        out = {"type": "number", "values": [], "groups": {}}

        if prop.count is not None:
            out["values"].append({"value": prop.count, "field": None, "aggregation": "count"})

        if prop.maximum is not None:
            out["values"].append({"value": prop.maximum, "field": None, "aggregation": "maximum"})

        if prop.mean is not None:
            out["values"].append({"value": prop.mean, "field": None, "aggregation": "mean"})

        if prop.median is not None:
            out["values"].append({"value": prop.median, "field": None, "aggregation": "median"})

        if prop.minimum is not None:
            out["values"].append({"value": prop.minimum, "field": None, "aggregation": "minimum"})

        if prop.mode is not None:
            out["values"].append({"value": prop.mode, "field": None, "aggregation": "mode"})

        if prop.sum_ is not None:
            out["values"].append({"value": prop.sum_, "field": None, "aggregation": "sum"})

        return out

def format_aggregation_response(response):

    out = {}
    if isinstance(response, AggregateGroupByReturn):
        for result in response.groups:
            field = result.grouped_by.prop
            out[field] = {"type": "text", "values": [], "groups": {}}

        for result in response.groups:
            field = result.grouped_by.prop

            if result.total_count is not None:
                out[field]["values"].append({"value": result.total_count, "field": result.grouped_by.value, "aggregation": "count"})

            for key, prop in result.properties.items():
                out[field]["groups"][result.grouped_by.value] = {}

            for key, prop in result.properties.items():
                formatted_props = format_aggregation_property(prop)
                out[field]["groups"][result.grouped_by.value][key] = formatted_props

    elif isinstance(response, AggregateReturn):
        properties = response.properties
        for field, field_properties in properties.items():
            out[field] = {}
            props = format_aggregation_property(field_properties)
            for key, prop in props.items():
                out[field][key] = {}
                out[field][key] = prop

    elif isinstance(response, list):
        for item in response:
            out.append(format_aggregation_response(item))

    return out