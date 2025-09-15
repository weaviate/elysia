
Whenever a [`Result`](../Reference/Objects.md#elysia.objects.Result) object is _yielded_ from an Elysia tool or decision agent, two things happen:

1. Any objects and metadata in the `Result` are automatically added to Elysia's [Environment](../Advanced/environment.md) for future use in the decision tree.
2. The `.to_frontend()` method of the `Result` parses the objects and metadata to a frontend-acceptable format and are yielded outside of the tree (to send payloads to a connected frontend).

Similarly, an [`Update`](../Reference/Objects.md#elysia.objects.Update) class also yields a payload outside of the tree, but does not add any objects to the environment.

All payloads that are sent from the decision tree to the frontend have the same structure:
```python
{
	"type": str, 
	"id": str,
	"user_id": str,
	"conversation_id": str, 
	"query_id": str,
	"payload": dict
}
```

Where the `"payload"` dictionary always contains:

```python
{
    "type": str, 
    "metadata": dict
    "objects": list[dict],
    ... # additional fields
}
```
where the additional fields are based on the `"type"` of the output. The dictionaries in the `list[dict]` of the `objects` is normally unique to each `"type"` that is returned, but will always include a `_REF_ID` field containing a unique identifier for its place in the Elysia environment. 

For example, any objects returned by the Elysia query tool will be mapped to specific fields that the frontend is 'aware' of. Items in the Weaviate collection that are returned are not known how to be displayed by the frontend as the fields are unique to the user's collection. So instead they are mapped to frontend-specific fields that are decided in advance by the [preprocessing step](../setting_up.md#preprocessing-collections) before they are returned outside of the tree.

- 