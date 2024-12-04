import datetime
from weaviate.classes.config import DataType

from elysia.globals.weaviate_client import client
from elysia.util.parsing import format_datetime

def get_all_collection_names():
    return [k.lower() for k in list(client.collections.list_all().keys()) if k.lower().startswith("example_verba")]

def get_collection_data_types(collection_name: str):
    properties = client.collections.get(collection_name).config.get().properties
    return {property.name: property.data_type[:] for property in properties}

def get_collection_weaviate_data_types(collection_name: str):
    data_types = get_collection_data_types(collection_name)
    data_mapping = {
        "text": DataType.TEXT,
        "int": DataType.INT,
        "number": DataType.NUMBER,
        "bool": DataType.BOOL,
        "date": DataType.DATE,
        "text[]": DataType.TEXT_ARRAY,
        "int[]": DataType.INT_ARRAY,
        "number[]": DataType.NUMBER_ARRAY,
        "bool[]": DataType.BOOL_ARRAY,
        "date[]": DataType.DATE_ARRAY
    }
    return {k: data_mapping[v] for k, v in data_types.items()}


def get_collection_data(collection_name: str, lower_bound: int = 0, upper_bound: int = -1, convert_datetime: bool = True):
    collection = client.collections.get(collection_name)

    if upper_bound == -1:
        upper_bound = len(collection)

    items = []
    for i, item in enumerate(collection.iterator()):
        if i >= lower_bound and i < upper_bound:
            if convert_datetime:
                for key, value in item.properties.items():
                    if isinstance(value, datetime.datetime):
                        item.properties[key] = format_datetime(value)
                    elif (
                        not isinstance(value, str) and 
                        not isinstance(value, list) and 
                        not isinstance(value, dict) and 
                        not isinstance(value, float) and 
                        not isinstance(value, int) and 
                        not isinstance(value, bool)
                    ):
                        item.properties[key] = str(value)
                    
                    if isinstance(item.properties[key], str) and item.properties[key].startswith("[") and item.properties[key].endswith("]"):
                        item.properties[key] = eval(item.properties[key])
            items.append(item.properties)
    return items