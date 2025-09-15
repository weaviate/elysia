# Returning Objects

## Result 

For more detail, [see the `Result` description in the reference](../Reference/Objects.md#elysia.objects.Result).

A `Result` is a class with a `to_json()` and `to_frontend()` method which returns the objects defined within the class, and their metadata, to either the decision tree environment and an attached frontend, respectively. By default, any `Result` objects yielded within a tool will automatically have these objects assigned and sent.

I.e.
```python
# top of file
from elysia.objects import Result

# existing tool code
yield Result(
    objects = [...],
    metadata = {...},
    payload_type = "<your_type_here>",
    name = "<name_to_go_in_environment>",
    mapping = {...}
)
# continued tool code
```

### Parameters

**Objects**

The `Result` class stores `objects`, which are a list of dictionaries containing anything that should be added to the environment or sent to the frontend. These can be stored however you like, but an Elysia-aware frontend expects the objects to have a certain format - the fields of the dictionaries need to be specific to the type of object being sent.

For example, an Elysia frontend knows what the payload of a 'document' object should be, and there are fields such as `'title'`, `'content'` and `'author'` which it expects. But the objects returned from a retrieval might not have the fields line up exactly like this - maybe it has fields called `'document_header'`, `'text_content'` and `'writer'` instead.

**Metadata**

Results also can have metadata attached to them, which is a single dictionary that contains any information about this particular list of objects that exist across all objects, rather than on an individual level. For example, if we have retrieved some documents using a search query, the metadata can contain what that search query was and what collection was searched, whereas `objects` are the objects themselves.

**Payload Type**

The `payload_type` parameter is used to tell the frontend what to expect. Within Elysia, there are some predefined types ([see here](../Reference/PayloadTypes.md)) that the built-in frontend is aware of. However, you can specify your own payload types if you are building a different frontend that is aware of different payload types.

**Mapping**

Within the `Result` there is also a `mapping` field which is a dictionary which maps from these fields to the fields contained within the objects themselves. For example, if we had an object with fields `"document_header"`, `"text_content"` and `"writer"`, then you may want to specify the `mapping` as
```python
{
    "title": "document_header",
    "content": "text_content",
    "author": "writer
}
```
So that on the frontend, when a document is displayed, the correct field values appear in the relevant fields.

**Name**

To index within the tree's environment, a `name` string identifier must be used to identify what type of objects are added to the environment. E.g., in the retrieval setting, this could be the collection name queried.


### Frontend 

When the default `.to_frontend(user_id, conversation_id, query_id)` method is called, it first calls `to_json(mapping=True)`, which returns a list of all objects that are mapped to their frontend-specific values, and then returns an augmented payload:

```python
{
    "type": "result",
    "user_id": str,
    "conversation_id": str,
    "query_id": str,
    "id": str,
    "payload": dict = {
        "type": str,
        "objects": list[dict],
        "metadata": dict,
    },
}
```

The outer level `type` field will default to `"result"`, by definition of the `Result` class. The `"payload"` field contains the unique `payload_type` (created on initialisation of the `Result`), as well as the objects/metadata within the `Result` object. Other fields, such as `user_id` and so forth, are inputs to the function which are automatically assigned when the decision tree processes the results.

### LLM Message

The `Result` class also has a `llm_message` parameter (or `.llm_parse()` method) which can be used to display custom information to both the decision agent LLM as well as any tools which use the `tasks_completed` field in `tree_data`.

**llm_message:**

The message can be formatted using placeholders given by:

- `{payload_type}`: The payload type of the object created at initialisation
- `{name}`: The name of the object
- `{num_objects}`: The number of objects in the `Result`

Additionally, any key in the metadata dictionary can be used.

E.g., on initialising the `Result`, you can pass `llm_message = "Retrieved {num_objects} from {collection_name}"` if you have `collection_name` in the metadata.


## Custom Objects

### Using Result Directly

If you want specific payload types or similar, you can directly use `Result` and simply change the initialisation parameters for your own use-case. E.g. in the tool call, if we are dealing cards randomly, you can use

```python
yield Result(
    objects = [
        {"card_title": "Jack of Clubs", "card_value": 11},
        {"card_title": "8 of Diamonds", "card_value": 8}
    ], 
    metadata = {"deck_size": 52},
    payload_type = "playing_cards",
    name="dealt_cards",
    llm_message="Dealt {num_objects} cards out of a possible {deck_size}."
)
```
Here, the `llm_message` will be input to the `tasks_completed` field of the `tree_data`, and when input to any tools using that field, this message will be displayed alongside what prompt was used and the tool called.

Also, the frontend payload will use the `playing_cards` payload type, and the decision tree processing the result will automatically create the correct frontend payload.

### Defining a New Subclass

You can create your own subclass of the `Result` class to customise your objects specifically for your use-case. This can be used when you may want some custom rules when the `.to_json`, `.to_frontend`, or `.llm_parse` methods are automatically used.

For example, if we just want to 

```python
class Card(Result):
    def __init__(
        self,
        objects: list[dict],
        metadata: dict = {},
        name: str = "card",
        mapping: dict | None = None,
        llm_message: str | None = None,
        unmapped_keys: list[str] = [],
    ):
        Result.__init__(
            self,
            objects=objects,
            payload_type="playing_card",
            metadata=metadata,
            name=name,
            mapping=mapping,
            llm_message=llm_message,
            unmapped_keys=unmapped_keys,
        )

    def to_json(self, mapping: bool = False) -> list[dict]:

        output_objects = self.objects
        for card in output_objects:
            suit = "♥️♦️♣️♠️"[hash(str(card)) % 4] if "card_title" in card else "🃏"
            card["suit_emoji"] = suit
            card["is_lucky"] = card.get("card_value", 0) > 10
        
        # Apply mapping if requested
        if mapping and self.mapping:
            output_objects = self.do_mapping(output_objects)
        
        return output_objects

    def llm_parse(self):
        out = f"""
        The first object in the hand was {self.objects[0]}.
        There were {len(self.objects)} cards in the hand in total.
        """
        if "deck_size" in self.metadata:
            out += f"There were a possible {self.metadata['deck_size']} cards to be dealt"
        return out

```

Here, the `to_json` method was overwritten to add a bit of flavour to the objects, which didn't exist in the original objects retrieved.
The `llm_parse` method was overwritten so it could use information from the objects themselves, which was not available using the generic placeholders.
