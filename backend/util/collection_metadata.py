
from backend.globals.weaviate_client import client

def get_all_collection_names():
    return [k.lower() for k in list(client.collections.list_all().keys()) if k.lower().startswith("example_verba")]

def get_collection_data_types(collection_name: str):
    properties = client.collections.get(collection_name).config.get().properties
    return {property.name: property.data_type[:] for property in properties}

def get_collection_data(collection_name: str, lower_bound: int = 0, upper_bound: int = -1):
    collection = client.collections.get(collection_name)

    if upper_bound == -1:
        upper_bound = len(collection)

    items = []
    for i, item in enumerate(collection.iterator()):
        if i >= lower_bound and i < upper_bound:
            items.append(item.properties)
    return items