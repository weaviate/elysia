# Environment

The environment is a persistent object across all actions, tools and decisions performed within the Elysia decision tree. It can be used to store global information, such as retrieved objects or information that needs to be seen across all tools and actions.

## Overview

The 'Environment' variable contains all objects returned from any tools called by the Elysia decision tree. In essence, it is a dictionary which is keyed by two variables:

- `tool_name` (str): the name of the tool used to add this field to the environment
- `name` (str): a subkey of `tool_name`, a unique `name` associated to the returns from that tool. E.g. a collection name from a retrieval.

And when these items are accessed, it is a *list of dictionaries*, where each dictionary contains two subfields:

- `objects` (list[dict])
- `metadata` (dict)

where each item in `objects` is a list of objects retrieved during the call of that tool. Each set of objects has its own corresponding metadata. 

For example, if Elysia calls the 'Query' tool, then the `tool_name` is `"query"` and the `name` is the name of the collection queried. Each list of objects has metadata associated with the query used to retrieve the data. So each list of objects has unique metadata.

<details closed>
<summary>Environment Example</summary>

Below is an example of what the environment looks like, after the tools `query` and `aggregate` have been called within this tree session.
```python
{
    "query": {
        "message_result": [
            {
                "objects": [
                    {"message_id": 1, "message_content": "Hi this is an example message about frogs!"},
                    {"message_id": 2, "message_content": "Hi this is also an example message about reindeer!"},
                ], 
                "metadata": {
                    "collection_name": "example_email_messages_collection",
                    "query_search_term": "animals"
                }
            },
        ]
    },
    "aggregate": {
        "pet_food_result": [
            {
                "objects": [
                    {
                        "average_price": 45.99, 
                        "product_count": 150, 
                    }
                ],
                "metadata": {
                    "collection_name": "pet_food",
                    "group_by": {"field": "animal", "value": "frog"} 
                }
            }
        ]
    }
}
```
This is just an example and not exactly how the structure within Elysia's inbuilt query and aggregate tools behave (they have much more information and would be harder to follow).

Note the levels of indexing the environment. 
- The outer most level is the tool name that yielded the result (`"query"` and `"aggregate"`).
- The next level is a `name` parameter associated with the `Result` that was yielded (`"message_result"` for query and `"pet_food_result"` for aggregate).
- After the `name` key, there is a list of dictionaries. This list corresponds to a different result that was yielded within the same tool/name combination.
- Each element of the list underneath `name` contains an `objects` and `metadata`, where the metadata is shared amongst all objects in this element.


</details>


## Interacting with the Environment

For a full breakdown of all the methods, [see the Environment reference page](../Reference/Objects.md#elysia.tree.objects.Environment).

### Automatic Assignment

When yielding a `Result` object from a Tool, the result's `to_json()` method will return a list of dictionaries which automatically gets created or appended to the objects field in `environment[tool_name][name]`. The metadata are added at the same point.

This calls the `.add()` method on the environment using the `Result` object.


<details closed>
<summary>Environment Example (cont. pt. 1)</summary>

From the `aggregate` tool, we can yield and initialise a `Result` back to the decision tree, where it is processed by the tree logic:

```python
yield Result(
    name="pet_food_result",
    objects = [
        {
            "average_price": 12.52, 
            "product_count": 33,
        }
    ],
    metadata = {
        "collection_name": "pet_food",
        "group_by": {"field": "animal", "value": "reindeer"} 
    }
)
```

And the updated environment looks like:

```python
{
    "query": {
        "message_result": [...]
    },
    "aggregate": {
        "pet_food_result": [
            {
                "objects": [
                    {
                        "average_price": 45.99, 
                        "product_count": 150, 
                    }
                ],
                "metadata": {
                    "collection_name": "pet_food",
                    "group_by": {"field": "animal", "value": "frog"} 
                }
            },
            {
                "objects": [
                    {
                        "average_price": 12.52, 
                        "product_count": 33,
                    }
                ],
                "metadata": {
                    "collection_name": "pet_food",
                    "group_by": {"field": "animal", "value": "reindeer"} 
                }
            }
        ]
    }
}
```
Notice how a new entry was not added to either the first or second level of the environment dictionary, but was instead appended to the existing entries under `aggregate -> pet_food_result`
</details>

### `.add()` and `.add_objects()`

[See the reference page.](../Reference/Objects.md#elysia.tree.objects.Environment.add)

When calling a tool, you can specifically add a `Result` object to the environment via 
```python
environment.add(tool_name, Result)
```
The corresponding `to_json()` method in the `Result` is used to obtain the objects which get added.

You can also have more control over which objects get added specifically by using
```python
environment.add_objects(tool_name, name, objects, metadata)
```
where `objects` is a list of dictionaries, `metadata` is a dictionary and `tool_name` and `name` are string identifiers.

<details closed>
<summary>Environment Example (cont. pt. 2)</summary>
If we were to do
```python
frog_result = Result(
    objects = [
        {
            "animal": "frog",
            "description": "Green and slimy"
        }
    ],
    name="animal_description"
)
environment.add(tool_name="descriptor", result=frog_result)
```
Then the environment would be updated to 
```python
{
    "query": {
        "message_result": [...]
    },
    "aggregate": {
        "pet_food_result": [...]
    }
    "descriptor": {
        "animal_description": [
            {
                "objects": [
                    {
                        "animal": "frog",
                        "description": "Green and slimy"
                    }
                ],
                "metadata": {}
            }
        ]
    }
}
```
Even though we never interfaced with a tool called `descriptor`.
</details>

### `.replace()`

[See the reference page.](../Reference/Objects.md#elysia.tree.objects.Environment.replace)

Change an item in the environment with another item 
```python
environment.replace(tool_name, name, objects, metadata, index)
```
either replace the entire list of items (objects + metadatas) or a single item at a particular index. `index` will only replace a particular item at that location, or if `None` (default) will replace the entire list.

<details closed>
<summary>Environment Example (cont. pt. 3)</summary>
If we were to change the results from the `"descriptor"` to something else,
```python
environment.replace(
    tool_name="descriptor", 
    name="animal_description",
    objects = [{"animal": "reindeer", "description": "Has a red nose"}]
)
```
Then the environment would be updated to 
```python
{
    "query": {
        "message_result": [...]
    },
    "aggregate": {
        "pet_food_result": [...]
    }
    "descriptor": {
        "animal_description": [
            {
                "objects": [
                    {
                        "animal": "reindeer",
                        "description": "Has a red nose"
                    }
                ],
                "metadata": {}
            }
        ]
    }
}
```
</details>

### `.find()`

[See the reference page.](../Reference/Objects.md#elysia.tree.objects.Environment.find)

You can use
```python
environment.find(tool_name, name, index)
```
Which is an easy way to retrieve objects from the environment associated with the `tool_name` and `name` string identifiers.
`index` is a parameter which finds the corresponding location of an item for these identifiers. Defaults to `None`, in which case it returns a list of all items.

This is essentially just an alias to `environment.environment[tool_name][name][index]`.

### `.remove()`

[See the reference page.](../Reference/Objects.md#elysia.tree.objects.Environment.remove)

Items (objects + metadata) in the environment can be removed via
```python
environment.remove(tool_name, name, index)
```
which uses the `tool_name` and `name` string identifiers to find the corresponding item.
The `index` parameter will remove the objects and metadata associated only with that position in the list. E.g., if `index=-1`, then the most recent entry in the list will be deleted. This defaults to `None`, in which case the entire set of objects for `tool_name` and `name` are removed. You can, of course, check the length of items beforehand via `len(environment.find(tool_name, name))` and use that to define the index. Will raise an `IndexError` if that index does not exist.

## The Hidden Environment

Within the environment there is also a dictionary, `environment.hidden_environment`, designed to be used as a store of data that is not shown to the LLM.
You can save any type of object within this dictionary as it does not need to be converted to string to converted to LLM formatting.

For example, this could be used to save raw retrieval objects that are not converted to their simple object properties, so you can still access the metadata output from the retrieval method that you otherwise wouldn't save inside the object metadata.

## Some Quick Usecases

- You may want to create a tool that only runs when the environment is non empty, so the `run_if_true` method of the tool [(see here for details)](advanced_tool_construction.md#run_if_true) returns `not tree_data.environment.is_empty()`.
- Your tool may not want to return any objects to the frontend, so instead of returning specific `Result` objects, you could modify the environment via `.add_objects()`, `.replace()` and `.remove()`. This stores 'private' variables that are not seen by the user unless they can manually inspect the environment.