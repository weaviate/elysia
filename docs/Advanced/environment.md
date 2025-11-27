# Environment

The environment is a persistent object across all actions, tools and decisions performed within the Elysia decision tree. It can be used to store global information, such as retrieved objects or information that needs to be seen across all tools and actions.

!!! note "Breakdown"
    For example, you ask Elysia for "recent messages from the Communications collection", which prompts the decision agent to use the built-in `query` tool. Information is retrieved from the tool, and these individual JSON objects need some way of being referenced by the decision agent _after the tool is complete_. 
    
    Most agentic systems will add this to the conversation history, i.e. just append the tool output to the next turn in conversation, then the agent picks up where it left off. Instead, **Elysia dynamically builds an internal environment** which is constantly referenced by the decision agent. This means previous long-context decisions aren't necessary to be stored in the conversation history, and the environment can be modified if needed and passed around between different LLM calls.


## Overview

For a detailed view at the `Environment` class and its methods, [see the Environment reference page](../Reference/Objects.md#elysia.tree.objects.Environment).

In essence, it contains a dictionary which stores JSON objects organised by the tools that added them to the environment, as well as associated metadata used to key the information. The top level key is the *tool name*, and under each tool name is a list of `EnvironmentItem`s, which have two attributes; `objects` and `metadata`.

- `objects` (list[dict]): contains multiple JSON objects (such as the results of a retrieval)
- `metadata` (dict): contains information that relates to each of the objects as a whole (such as the name of the collection that was queried)

<details closed>
<summary>Example</summary>

You ask Elysia for "the two most recent messages from the Communications collection", which calls the query tool. Then when the tool is run, the following objects are retrieved:

```json
[
    {
        "author": "John",
        "content": "Hey Jane, it's John"
    },
    {
        "author": "Jane",
        "content": "Hey John, good to hear from you."
    }
]
```
When these are yielded out of the tool (more on that later), the environment will look like:
```python
{
    "query": [
        EnvironmentItem(
            objects = [
                {
                    "author": "John",
                    "content": "Hey Jane, it's John"
                },
                {
                    "author": "Jane",
                    "content": "Hey John, good to hear from you."
                }
            ],
            metadata = {
                "collection_name": "Communications",
                "query_used": {"search_term": None, "sort_by": "created_at", "sort_order": "desc"}
            }
        )
    ]
}
```
In the actual tool run, `metadata` from the query tool would contain more information. If another run of the `query` tool completed with the _same metadata_, then additional objects would be added under the same `EnvironmentItem`. However, if the query was completed with different metadata, such as from a different collection, then a separate `EnvironmentItem` would be added underneath `"query"` with new metadata.

</details>


## Interacting with the Environment within Tools

For a full breakdown of all the methods, [see the Environment reference page](../Reference/Objects.md#elysia.tree.objects.Environment).

There are two ways to interact with the environment within tools:
1. Yielding a `Result` object or a subclass of `Result` inside the tool (as an async generator), which is **automatic assignment**
2. Using the **environment methods** to get, add, edit, and remove items from the environment

### Automatic Assignment with Frontend Updates

When yielding a `Result` object from a Tool, the result's `to_json()` method will return a list of dictionaries (the *objects*) which become the `.objects` attribute of the `EnvironmentItem` object. The metadata are added at the same point (`result.metadata`).

Behind the scenes, this calls the `.add()` method on the environment on the `Result` object directly, when the decision tree receives the result after it is yielded out of the tool.

**In addition** to modifying the environment, yielding a `Result` object will also send a payload from the decision tree (when using its `.async_run` method, which is an async generator function), with the objects and metadata of the result. This is used to update any attached frontend with this data.

For example, if you just used the manual methods such as `.add()` or `.add_objects()`, then the frontend will not be updated with this data. But yielding a result will send a payload to the frontend with the data, so it can display it.


<details closed>
<summary>Example</summary>

We create a tool called `aggregate` that calculates summary statistics on some data. Within the tool, we initialise and yield a `Result` back to the decision tree:

```python
yield Result(
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
    ..., # previous items in the environment
    "aggregate": [
        EnvironmentItem(
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
    ]
}
```
</details>

### Environment Methods

By manipulating the environment object itself you can completely customise how the environment looks. Your tool can change the internal context of the decision tree, add new information, or remove old information.

For example, you can update a re-occuring description of something in context whenever the tool is run with updated results.

#### Accessing the Environment Variable

Within a tool, **you can access the environment through the `tree_data`** variable via `tree_data.environment`, which gives you direct access to [Environment](../Reference/Objects.md#elysia.tree.objects.Environment). The `tree_data` variable is passed into every tool call.

E.g., in your tool call (via the wrapper), you can access the environment like so:
```python
from elysia import tool
from elysia.tree.objects import TreeData

@tool
def my_tool(tree_data: TreeData):
    environment = tree_data.environment
    ...
```

or within your `Tool` class, the `__call__` method must accept `tree_data: TreeData` as a parameter:
```python
class MyTool(Tool):
    
    ...

    def __call__(self, tree_data: TreeData, ...):
        environment = tree_data.environment
        ...
```
see [advanced tool construction](advanced_tool_construction.md) for more details.

The methods in `Environment` are detailed in [this reference page](../Reference/Objects.md#elysia.tree.objects.Environment), but some examples are detailed here:

#### `.add()` and `.add_objects()`

[See the reference page.](../Reference/Objects.md#elysia.tree.objects.Environment.add)

When calling a tool, you can specifically add a `Result` object to the environment via 
```python
environment.add(tool_name, Result)
```
The corresponding `to_json()` method in the `Result` is used to obtain the objects which get added.

You can also have more control over which objects get added specifically by using
```python
environment.add_objects(tool_name, objects, metadata)
```
where `objects` is a list of dictionaries, `metadata` is a dictionary and `tool_name` is a string identifier (by default, the name of the tool that added this data, but in using `.add_objects`, you can specify a different name if you wish).

<details closed>
<summary>Example (.add)</summary>

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
    ... # previous items in the environment
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
Even though we never interfaced with a tool called `descriptor`. This almost the same as yielding the `Result` object directly.
</details>

<details closed>
<summary>Example (.add_objects)</summary>

To replicate the same functionality as above (but without any custom `Result` objects), we can do
```python
environment.add_objects(
    tool_name="descriptor", 
    objects = [
        {
            "animal": "frog",
            "description": "Green and slimy"
        }
    ],
    metadata = {}
)
```
Then the environment would be updated to 
```python
{
    ... # previous items in the environment
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
This is a simpler method than `.add()`, as it does not require a `Result` object.
</details>

#### `.append()`

[See the reference page.](../Reference/Objects.md#elysia.tree.objects.Environment.append)

You can use
```python
environment.append(tool_name, objects, metadata, metadata_key, metadata_value)
```

This appends objects to _existing_ lists in the environment, rather than adding new `EnvironmentItem`s or creating new lists (which is `.add()` or `.add_objects()`).

<details closed>
<summary>Example</summary>
Say you have

```python
{
    "query": [
        EnvironmentItem(
            objects = [
                {
                    "animal": "frog",
                    "price": 3.99
                }
            ],
            metadata = {
                "collection_name": "pet_food", 
                "query_search_term": "animals"
            }
        )
        ... # more items in the environment under "query"
    ]
}
```
and you want to append an object that matches the same `"collection_name"` of `"pet_food"`, you can do:

```python
environment.append(tool_name="query", objects=[{"animal": "reindeer", "price": 12.99}], metadata_key="collection_name", metadata_value="pet_food")
```
Then the environment would be updated to:
```python
{
    "query": [
        EnvironmentItem(
            objects = [
                {
                    "animal": "frog",
                    "price": 3.99
                },
                {
                    "animal": "reindeer",
                    "price": 12.99
                }
            ],
            metadata = {
                "collection_name": "pet_food", 
                "query_search_term": "animals"
            }
        ),
        ...
    ]
}
```
</details>

#### `.get()` and `.get_objects()`

See the reference page for [get](../Reference/Objects.md#elysia.tree.objects.Environment.get) and [get_objects](../Reference/Objects.md#elysia.tree.objects.Environment.get_objects).

```python
environment.get(
    tool_name: str
)
```
In short, this retrieves items from the environment. The `.get()` method is a simple indexer of the environment dictionary, and `.get(tool_name)` is equivalent to `environment.environment.get(tool_name, None)`. It returns a list of `EnvironmentItem` objects (or `None` if the `tool_name` is not found in the environment).

```python
environment.get_objects(
    tool_name: str,
    metadata: dict | None = None,
    metadata_key: str | None = None,
    metadata_value: Any | None = None,
)
```

The `.get_objects()` method is a more complex indexer of the environment. You can specify a set of `metadata` (a dictionary) to filter the objects of the environment. It will return _all objects_ for a single `tool_name` that match this metadata exactly.

<details closed>
<summary>Example (using `metadata`)</summary>

If our environment is
```python
{
    ..., # previous items in the environment
    "aggregate": [
        EnvironmentItem(
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
    ]
}
```
Then we can do
```python
environment.get_objects(
    tool_name="aggregate",
    metadata={
        "collection_name": "pet_food",
        "group_by": {"field": "animal", "value": "reindeer"} 
    }
)
```
This will return the list of `EnvironmentItem` objects that match this metadata exactly.
</details>

Similarly, you could specify only a single key of the metadata (`metadata_key`) and the expected value this key should be (`metadata_value`) and it will return all objects for a single `tool_name` that match this key, value pair exactly.

<details closed>
<summary>Example (using `metadata_key` and `metadata_value`)</summary>

If our environment is
```python
{
    "search": [
        EnvironmentItem(
            objects = [
                {
                    "price": 8.43,
                    "animal": "reindeer",
                },
                {
                    "price": 1.49,
                    "animal": "giraffe",
                },
                {
                    "price": 2.99,
                    "animal": "frog",
                },
                {
                    "price": 2.13,
                    "animal": "reindeer",
                }
            ],
            metadata = {
                "collection_name": "pet_food",
                "filters": [ {"field": "price", "value": 10, "operator": "<"}]
            }
        ),
        EnvironmentItem(
            objects = [
                {
                    "average_price": 5.28, 
                    "product_count": 2,
                }
            ],
            metadata = {
                "collection_name": "pet_food",
                "group_by": {"field": "animal", "value": "reindeer"} 
            }
        )
    ]
}
```
Then we have two tools that have interacted with the same collection. They share a metadata key `collection_name`, so to retrieve both of these items we can do
```python
environment.get_objects(
    tool_name="search",
    metadata_key="collection_name",
    metadata_value="pet_food"
)
```
This will return the list of `EnvironmentItem` objects that match this metadata key-value pair, both objects above.
</details>

#### `.replace()`

[See the reference page.](../Reference/Objects.md#elysia.tree.objects.Environment.replace)

Change an item in the environment with another item 
```python
environment.replace(tool_name, objects, metadata, metadata_key, metadata_value)
```

This will replace the item in the environment that matches both the `tool_name`, and either the `metadata_key` and `metadata_value` pair, or the `metadata` dictionary, with the new `objects`. This will remove all existing `objects` and replace them with the new ones.

<details closed>
<summary>Example</summary>
If our environment is

```python
{
    ..., # previous items in the environment
    "locations": [
        EnvironmentItem(
            objects = [
                {
                    "name": "Goblin Cave",
                    "description": "A dark gloomy cave", 
                    "current": True,
                },
                {
                    "name": "Elysia Town",
                    "description": "A bustling town full of people",
                    "current": False,
                }
            ],
            metadata = {
                "id": "basic_locations",
                "level": 1,
            }
        )
    ]
}
```
Then we can change the `"locations"` objects by:
```python
environment.replace(
    tool_name="locations", 
    metadata_key="id",
    metadata_value="basic_locations",
    objects = [
        {
            "name": "New City",
            "description": "A big city", 
            "current": True,
        },
    ]
)
```
And this changes _every object_ in the environment that matches the `tool_name` and `metadata_key` and `metadata_value` pair. You can similarly use a full `metadata` to match the metadata exactly.

If, for example, you wanted to only change certain properties, you could first `.get_objects()` and then modify them, and `.replace()` them back:

```python
locations = environment.get_objects(
    tool_name="locations",
    metadata_key="id",
    metadata_value="basic_locations"
)
locations[0]["current"] = False
locations[0]["current"] = True
environment.replace(
    tool_name="locations", 
    metadata_key="id",
    metadata_value="basic_locations",
    objects = locations
)
```

</details>

#### `.remove()`

[See the reference page.](../Reference/Objects.md#elysia.tree.objects.Environment.remove)

Objects in the environment can be removed (the `.objects` attribute of an individual environment item) using the `.remove()` method.
```python
environment.remove(tool_name, metadata, metadata_key, metadata_value)
```
which uses the `tool_name` and either the full `metadata` dictionary, or the `metadata_key` and `metadata_value` pair to find the corresponding item.

The actual `EnvironmentItem` still exists, with its metadata, but the `.objects` attribute is an empty list.

#### `.is_empty()`

[See the reference page.](../Reference/Objects.md#elysia.tree.objects.Environment.is_empty)

This method returns `True` if the environment is empty, and `False` otherwise. This includes objects removed from the `.remove()` method (empty lists). So if you had an environment of only empty lists of objects, it will still return `True`.

## The Hidden Environment

Within the environment there is also a dictionary, `environment.hidden_environment`, designed to be used as a store of data that is not shown to the LLM.
You can save any type of object within this dictionary as it does not need to be converted to string to converted to LLM formatting.

For example, this could be used to save raw retrieval objects that are not converted to their simple object properties, so you can still access the metadata output from the retrieval method that you otherwise wouldn't save inside the object metadata.

## Some Quick Usecases

- You may want to create a tool that only runs when the environment is non empty, so the `run_if_true` method of the tool [(see here for details)](advanced_tool_construction.md#run_if_true) returns `not tree_data.environment.is_empty()`.
- Your tool may not want to return any objects to the frontend, so instead of returning specific `Result` objects, you could modify the environment via `.add_objects()`, `.replace()` and `.remove()`. This stores 'private' variables that are not seen by the user unless they can manually inspect the environment.