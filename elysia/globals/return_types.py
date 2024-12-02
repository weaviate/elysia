specific_return_types={
    "conversation": "retrieve full conversations, including all messages and message authors, with timestamps and context of other messages in the conversation.",
    "message": "retrieve individual messages, only including the author of each individual message and timestamp, without surrounding context of other messages by different people.",
    "ticket": "retrieve individual tickets, including all fields of the ticket.",
    "ecommerce": "retrieve individual products, including all fields of the product."
}

all_return_types={
    "conversation": "retrieve full conversations, including all messages and message authors, with timestamps and context of other messages in the conversation.",
    "message": "retrieve individual messages, only including the author of each individual message and timestamp, without surrounding context of other messages by different people.",
    "ticket": "retrieve individual tickets, including all fields of the ticket.",
    "ecommerce": "retrieve individual products, including all fields of the product.",
    "generic": "retrieve any other type of information that does not fit into the other categories."
}

conversation =  {
    "content": "the content or text of the message, what was written. string",
    "author": "the author of the message. string",
    "timestamp": "the timestamp of the message in any format. datetime/string/other",
    "conversation_id": "the id of the conversation that the message belongs to. integer/string/other",
    "message_id": "the id of the message itself, within the conversation. integer/string/other"
}

message = {
    "content": "the content or text of the message, what was written. string",
    "author": "the author of the message. string",
    "timestamp": "the timestamp of the message in any format. datetime/string/other",
    "conversation_id": "the id of the conversation that the message belongs to. integer/string/other",
    "message_id": "the id of the message itself, within the conversation. integer/string/other"
}

ticket = {
    "title": "the title of the ticket. string",
    "subtitle": "the subtitle of the ticket. string",
    "content": "the text of the ticket. string",
    "created_at": "the timestamp of the original creation time/date of the ticket in any format. datetime/string/other",
    "updated_at": "the timestamp of the last update time/date of the ticket in any format. datetime/string/other",
    "url": "the url of the ticket. string",
    "status": "the status of the ticket. string",
    "id": "the id of the ticket. integer/string/other",
    "tags": "the tags of the ticket. list[string/other]",
    "comments": "either the comments of the ticket, or the number of comments. list[string/dict/other] / integer"
}

ecommerce = {
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
    "sizes": "the size(s) of the product. list[string/other] / string"
}

generic = {
    "title": "the title of the information. string",
    "subtitle": "the subtitle of the information. string",
    "content": "the content of the information. string/other",
    "url": "the url of the information. string",
    "id": "the id of the information. integer/string/other",
    "author": "the author of the information. string/other",
    "timestamp": "the timestamp of the information in any format. datetime/string/other",
    "tags": "the tags of the information. list[string/other]"
}

types_dict = {
    "conversation": conversation,
    "message": message,
    "ticket": ticket,
    "ecommerce": ecommerce,
    "generic": generic
}