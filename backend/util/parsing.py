import json

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
    
        elif isinstance(item, str):
            out += item + "\n"

    return out