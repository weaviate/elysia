import json
import datetime

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

def update_current_message(current_message: str, new_message: str):
    
    if current_message == "":
        return new_message, new_message

    sentences = new_message.split(". ")

    if len(sentences) == len(current_message.split(". ")):
        return current_message, ""

    return current_message + " " + sentences[-1], sentences[-1]