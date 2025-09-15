specific_return_types = {
    "conversation": (
        "Full conversations, including all messages and message authors, with timestamps and context of other messages in the conversation. "
        "This type can only be selected if there is a field that uniquely identifies what conversation each message belongs to, e.g. a 'Conversation ID', "
        "as well as a field that uniquely identifies each message within the conversation, e.g. a 'Message ID'."
    ),
    "message": (
        "Individual messages, only including the author of each individual message and timestamp, "
        "without surrounding context of other messages by different people. "
        "If the 'conversation' field is suitable, then this is also suitable by definition."
    ),
    "ticket": ("Support tickets, similar to Github issues or similar."),
    "product": (
        "Products items, so usually involving descriptions, prices, ratings, reviews, etc, but not always. "
        "Contains an image field, and space for plenty of metadata."
    ),
    "document": (
        "Text-based information, optionally with a title, author, date, and content, but not always. "
        "Ideal for any text-based information."
    ),
}

all_return_types = {
    **specific_return_types,
    "generic": (
        "Any other type of information that does not fit into the more specific categories. "
        "Contains fields for a range of different types of information, and is a good option for a wide range of data if no other display type is available. "
    ),
    "table": (
        "A table of information, with rows and columns. Used for displaying all of the data in a structured way. "
        "This is a fall-back option if not other display type is available. "
        "Alternatively, if the data or query requires a more analytical insight, this could be a good option."
    ),
}

conversation = {
    "content": "the content or text of the message, what was written. string",
    "author": "the author of the message. string",
    "timestamp": "the timestamp of the message in any format. datetime/string/other",
    "conversation_id": "the id of the conversation that the message belongs to. integer/string/other",
    "message_id": "the id of the message itself, within the conversation. integer/string/other",
}

message = {
    "content": "the content or text of the message, what was written. string",
    "author": "the author of the message. string",
    "timestamp": "the timestamp of the message in any format. datetime/string/other",
    "conversation_id": "the id of the conversation that the message belongs to. integer/string/other",
    "message_id": "the id or index of the message, used to either identify a message within a conversation or the message itself. integer/string/other",
}

ticket = {
    "title": "the title of the ticket. string",
    "subtitle": "the subtitle of the ticket. string",
    "author": "the author of the ticket. string",
    "content": "the text of the ticket. string",
    "created_at": "the timestamp of the original creation time/date of the ticket in any format. datetime/string/other",
    "updated_at": "the timestamp of the last update time/date of the ticket in any format. datetime/string/other",
    "url": "the url of the ticket. string",
    "status": "the status of the ticket. string",
    "id": "the id of the ticket. integer/string/other",
    "tags": "the tags of the ticket. list[string/other]",
    "comments": "either the comments of the ticket, or the number of comments. list[string/dict/other] / integer",
}

product = {
    "name": "the name of the product. string",
    "description": "the description of the product. string",
    "price": "the price of the product. float/integer/other",
    "category": "the category of the product. string",
    "subcategory": "the subcategory of the product. string",
    "collection": "the collection that the product belongs to. string",
    "rating": "the rating of the product. float/integer/other",
    "reviews": "the reviews of the product, or number of reviews. list[string/dict/other] / integer",
    "tags": "the tags of the product. list[string/other]",
    "url": "the url of the product. string",
    "image": "the image of the product. string/other",
    "brand": "the brand of the product. string",
    "id": "the id of the product. integer/string/other",
    "colors": "the color(s) of the product. list[string/other] / string",
    "sizes": "the size(s) of the product. list[string/other] / string",
}

document = {
    "title": "the title of the document. string",
    "author": "the author or username or creator of the document. string",
    "date": "any date or time format. datetime/string/other",
    "content": "the textual content of the document. string/other",
    "category": "some string describing the category of the document, e.g. type of something. string",
}

generic = {
    "title": "the title of the information. string",
    "subtitle": "the subtitle of the information. string",
    "content": "the content of the information. string/other",
    "url": "the url of the information. string",
    "id": "the id of the information. integer/string/other",
    "author": "the author of the information. string/other",
    "timestamp": "the timestamp of the information in any format. datetime/string/other",
    "tags": "the tags of the information. list[string/other]",
    "category": "some string describing the category of the information, e.g. type of something. string",
    "subcategory": "some string describing a nested level of category of the data. string",
}

types_dict: dict[str, dict[str, str]] = {
    "conversation": conversation,
    "message": message,
    "ticket": ticket,
    "product": product,
    "generic": generic,
    "document": document,
    "table": {},
}
