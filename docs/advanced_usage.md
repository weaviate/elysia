# Customising the Elysia Decision Tree

If you haven't followed [the basic guide](basic.md) yet, check that out first before you go into more detail here.

[See the setup page for details on setting up models and API keys](setting_up.md).

## Changing the Style, Agent Description or End Goal

You can initialise these parameters at the startup of the tree, for example
```python
from elysia import Tree
tree = Tree(
    style = "...",
    agent_description = "...",
    end_goal = "..."
)
```
Each of these options is described below.

### Style

You can customise the style of Elysia's text responses by either initialising the decision tree with the `style` argument, 
or creating a tree and then modifying the style later.
```python
tree = Tree()
tree.change_style("Always speak in rhyming couplets.")
```

You should see the changes in Elysia's textual responses!
```python
response, _ = tree("Hi Elysia, how are you?")
print(response)
```
```
## Hello there, it's Elly, I'm doing fine, I hope you are as well, in this rhyme!
```


### Agent Description

The `agent_description` parameter in Elysia assigns the agent a unique identity, which you can use to customise the objective of the agent. 

Change this via `change_agent_description`, e.g.
```python
tree.change_agent_description("You are a travel agent, specialised in creating unique travel plans for customers that interact with you.")
```

### End Goal

The `end_goal` parameter describes to Elysia what criteria it will use to decide when the decision tree should end. Normally, this is something similar to 'When you have either achieved all the user has asked of you, or you can no longer do any actions that have not failed already, and you have no other options to explore.'. But this could be unique to your use case, for example,
```python
tree.change_end_goal("The user has been recommended a hotel, travel options as well as activities to do in the local area. Or, you have exhausted all options. Or, you have asked the user for more clarification about their request.")
```

## Creating a Title

You can run `tree.create_conversation_title()` to use the base LM in the tree to create a title for the conversation (which uses the conversation history as inspiration). This saves the title as the `tree.conversation_title` attribute, which you can also directly overwrite or access if needed. An async version of this method is available at `tree.create_conversation_title_async()`.

## Loading and Saving Trees

### Locally

You can export a decision tree to a JSON serialisable dictionary object via [`tree.export_to_json()`](Reference/Tree.md#elysia.tree.tree.Tree.export_to_json). This saves certain aspects, including:

- The Tree Data, which includes:
    - The environment, including all retrieved objects from previous tool runs or otherwise
    - The collection metadata, if processed
- Class variables used to initialise the tree
- The config options, such as the settings (including the model choices) and the style, agent description and end goal.
- The branch initialisation, which will be re-run when the tree is loaded.

The tree can be loaded from the dictionary via running the [`Tree.import_from_json(json_data=...)`](Reference/Tree.md#elysia.tree.tree.Tree.import_from_json) method, passing as the first argument `json_data` which is the output from `tree.export_to_json()`.

**Note that any custom tools or branches added to the decision tree are not saved and need to be manually re-added, in the same way that your tree was originally initialised.**

### With Weaviate

Also included are two similar functions for saving and loading a decision tree within a Weaviate instance. To save a tree in a Weaviate instance, you can use [`tree.export_to_weaviate(collection_name, client_manager)`](Reference/Tree.md#elysia.tree.tree.Tree.export_to_weaviate). You can specify the collection that you will be saving to via the `collection_name` argument. You can specify the Weaviate cluster information by passing a [`ClientManager`](Reference/Client.md#elysia.util.client). If you do not provide a ClientManager, it will automatically create one from the specification set in the environment variables. It will save the tree according to the `conversation_id` used to initialise the tree (which is randomly generated via a UUID if not set).

To load a tree from Weaviate, you can use the class method [`Tree.import_from_weaviate(collection_name, conversation_id, client_manager)`](Reference/Tree.md#elysia.tree.tree.Tree.import_from_weaviate). You must use the correct `conversation_id` to load a tree. If you do not know the conversation ID, you can view all available conversation IDs saved to a Weaviate collection via
```python
from elysia.tree.util import get_saved_trees_weaviate
get_saved_trees_weaviate()
```
Which will return a dictionary whose keys correspond to the available conversation IDs, and whose values are the titles as strings of the conversations ([if one was created via `tree.create_conversation_title()`](#creating-a-title)).

**Note that any custom tools or branches added to the decision tree are not saved and need to be manually re-added, in the same way that your tree was originally initialised.**