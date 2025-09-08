
# Advanced Tool Construction Overview

This page details how to create more custom and flexible tools in Elysia, by inheriting the `Tool` class and adding it to the decision tree via the `.add_tool` of the `Tree` object.

To see an easier method of creating tools, see the [Creating a Tool](../creating_tools.md) guide.

This page will detail all relevant information for tool construction, to get started with an example, see
- [A basic text response example](#example-text-response-basic)
- [A more complex example dealing cards to Elysia](#example-dealing-cards-randomly-from-a-deck-intermediate)

## Initialisation

A tool must be initialised with
```python
    def __init__(self, logger: Logger | None = None, **kwargs):
        super().__init__(
            name=...,
            description=...,
            status=..., # optional
            inputs=..., # optional
            end=..., # optional
            **kwargs # required
        )
```

- `name`: A short, one or two word name of the tool.
- `description`: A detailed description of what the tool will do and what it will accomplish. This is how the LLM decides whether to call the tool or not, so it is important that this is comprehensive and detailed.
- `status` (optional): A short 'update' message that is displayed whilst the tool is running.
- `inputs` (optional): A dictionary of inputs to your tool, which the LLM will decide on, which conform to the following structure:
    ```python
    {
        input_name: {
            "description": str,
            "type": Any,
            "default": Any,
            "required": bool
        },
        ...
    }
    ```
    You can have as many inputs as you want, but similar to the description field, the descriptions here need to be informative so that the LLM knows exactly what to choose.
- `end` (optional): A bool denoting whether the tool is capable of ending the entire decision tree. For example, a `text_response` tool can end the process, but a `query` tool cannot. This is because a query tool returns some information which is then parsed by the decision tree _afterwards_, to see if the retrieved information was worthwhile. Note that setting `end=True` does not guarantee that after this tool is finished running, the decision process ends, it only allows the model to choose that performing this action _can_ end the tree.
- `**kwargs` (required)

The `logger` can be automatically assigned to the initialisation of the tool and is passed by default into the Elysia decision tree. Save this as `self.logger = logger` to use it in the tool call later.

## Tool Call

The tool should have an _async_ `__call__` method,
```python
    async def __call__(
        tree_data: TreeData,
        inputs: dict,
        base_lm: dspy.LM,
        complex_lm: dspy.LM,
        client_manager: ClientManager,
        **kwargs
    ):
        # tool call here
```
which has the following inputs:

- `tree_data`, an object of type `TreeData` which contains some information about the state of the decision making process at this point. This is likely the most relevant data to use in your tool calls, if the tool will affect the decision tree in some way. Here, you can access the environment (via `tree_data.environment`), the tasks completed dictionary (`tree_data.tasks_completed`), the collection metadata (`tree_data.collection_data`) and more - [see here for all data you can access](../Reference/Objects.md#elysia.tree.objects.TreeData).
- `inputs`: the inputs previously defined, formatted as 
    ```python
    {
        input_name_1: value,
        input_name_2: value,
        ...
    }
    ```
    which were given by the LLM (or reverted to their default values, if the LLM chose nothing for a particular input).
- `base_lm`: a `dspy.LM` object that can be used to interface with LLMs within the tool. This model is the same picked in the `elysia.configure(base_model="...", base_provider="...")` call. You can use this directly, e.g.
    ```python
    base_lm("hello, world!")
    ```
    or via a DSPy signature or module.
- `complex_lm`: same as above for the complex LM specified.
- `client_manager`: the interface to the Weaviate cluster you are connected to.

The `__call__` method is automatically run when the LLM decision agent chooses to use that tool. You can use any of these inputs within your tool method and the code will be executed.

Within the `__call__` method of the tool, you will want to interact with the decision tree in some way. There are multiple ways of doing this, either via returning various objects that Elysia defines within `elysia.objects`, or by interacting with the environment.

### Returning Objects

If you return an Elysia specific object, they will be returned to the decision tree and automatically parsed in different ways which automatically add the relevant objects to the environment, and send any payloads to the frontend.

Within your tool's call method, you may want to `yield` different objects to bring them back to the tree.

- Any class that inherits from the [`Update` class](../Reference/Objects.md#elysia.objects.Update) will send updates to the frontend, such as a status message.
- Any class that inherits from the [`Result` class](../Reference/Objects.md#elysia.objects.Result) have their corresponding objects added to the tree's environment, which the decision agent will 'look at', so that it can continue making decisions and respond accordingly to the user. Then, if applicable, relevant payloads will be sent to the frontend.


#### Status

A **[`Status`](../Reference/Objects.md#elysia.objects.Status)** message is initialised with a single string argument, this displays on the frontend or the progress bar a unique message.


#### Result

Running inside of the call something like:
```python
    yield Result(
        objects = [
            {
                "title": "Example Result",
                "content": "This is just an example of a result"
            }
        ]
    )
```
will mean that this particular object gets added to the Tree's 'Environment', and the LLM can look at this to make further decisions. This will also automatically parse this object as a payload to a frontend, if one is connected.

The arguments for the `Result` are:
 - `objects`: a list of dictionaries that contain your specific objects. Currently, the keys of the dictionary do not matter, but if you want to display these items on the frontend, they need to conform to specific keys ([see later](#displaying-objects-frontend-only))
 - `metadata`: a dictionary of metadata items. You can use this to separate global information from object-specific information.
 - `payload_type`: a string describing the type of objects you are giving.
 - `mapping`: a dictionary mapping frontend-aware fields to the fields in `objects` ([see here](../Reference/PayloadTypes.md)).

[See the custom objects page for more detail.](custom_objects.md)

## Interacting with the Environment

[See here a full description of the methods that you can use to interact with the environment](environment.md).

In short, the environment can be modified either by yielding `Result` objects, or by calling the environment methods explicitly. 
You can do so via calling the `.add()`, `.add_objects()`, `.replace()` or `.remove()` from the `tree_data`.

**Note:** If you add items to the environment and also yield a `Result` object with the same items, there will likely be duplicate items in the environment.

## Displaying Objects (Frontend Only)

You can yield a `Result` to the frontend, and by specifying the `payload_type`, the frontend will be aware of the type of object sent. The payload type currently must be one of the objects in [the reference page](../Reference/PayloadTypes.md), and you must also either conform to the field structure for each type or provide a `mapping` that maps from the expected fields to the fields in the objects.

To display your objects without any mappings or display types, you can specify the payload type as `table`.

## Easy LLM calls with Elysia Chain of Thought

An easy way to access attributes from the tree (if you are calling an LLM within the tool) is to use the custom `ElysiaChainOfThought` DSPy module with specific arguments. This automatically adds information from the `tree_data` to an LLM prompt as inputs in a DSPy signature, as well as some specific outputs deemed useful within the decision tree environment (and a chain of thought reasoning field output field).

To call this, you can do, for example
```python
from elysia.util.elysia_chain_of_thought import ElysiaChainOfThought
my_module = ElysiaChainOfThought(
    MyCustomSignature, # a dspy signature needing to be defined
    tree_data=tree_data, # tree_data input from the tool
    message_update: bool = True,
    environment: bool = False,
    collection_schemas: bool = False,
    tasks_completed: bool = False,
    collection_names: list[str] = [],
)
```
By setting the boolean flags for the different variables, you can control the inputs and outputs assigned, whereas some inputs are always included (such as user prompt).

To use the augmented module via `ElysiaChainOfThought`, call the `.aforward()` method of the new module, passing all your *new* inputs as keyword arguments. You do not need to include keyword arguments for the other inputs, like the `environment` or `user_prompt`, they are automatically added, e.g.
```python
my_module.aforward(input1=..., input2=..., lm=...)
```
The `lm` parameter can be inherited from the tool inputs, i.e. `base_lm` or `complex_lm`. Or you can define your own LMs via `dspy.LM`.

[See the description for more details](../Reference/Util.md#elysia.util.elysia_chain_of_thought)

## Adding Tools to the Tree

To add a Tool to be evaluated in the tree, just run `.add_tool`. For example

```python
.add_tool(TextResponse)
```
This will add the `TextResponse` tool to the root branch, by default (the base of the decision tree).

Elysia sometimes has *branches* in the decision tree, which can be created via `add_branch`. If you want to add a tool to a particular branch, specify the `branch_id`, e..g if we have a branch called "responses", then

```python 
.add_tool(TextResponse, branch_id="responses")
```

You can add tools on top of existing tools. Assume that the decision tree has the `multi_branch` structure, so that at the root node there are two options: `search` and `text_response`. The `text_response` option is a single tool, whereas the `search` option is in fact a branch with two options: `query` and `aggregate`.

If you wanted to add a tool called `CheckOutput` to be run *after* the query tool, then you can do:
```python
.add_tool(CheckOutput, branch_id="search", from_tool_ids = ["query"])
```
which will add the `CheckOutput` tool to the line `search -> query`, resulting in `search -> query -> check_output`.

Note that the `search` branch still has two options, but if the decision LLM chooses to do the `query` tool, then the `check_output` tool is available for choice after querying. Also note that if a tool has no inputs and is alone in a decision node (it is the only option for the LLM to pick), the LLM decision will be skipped and the node will be automatically added. You can add more nodes to after the `query` tool and then the decision LLM will now resume operations at that node.


## Self Healing Errors

You can yield an **[`Error`](../Reference/Objects.md#elysia.objects.Error)** object to 'return' an error from the tool to the decision tree. These errors are saved within the tree data and automatically added to the decision nodes as well as any LLM calls made with `ElysiaChainOfThought` called within that tool. The LLM is 'informed' about these errors via an input to the prompts. The LLM can choose to continue calling the tool again, in spite of the error (if it seems fixable), or it can use the information to end the conversation and inform the user of an error, or to try a different tool that will not error.

The `Error` object is initialised with a single string argument, which should be informative and descriptive.

*Note that this does not raise an error within Python, it is used to 'inform' the LLM that a potentially preventable error has occurred somewhere within the tool.*

**For example**, the Query tool built into Elysia will yield `Error` objects if the LLM creates a query which fails to run in Weaviate, such as not having the correct filter type for a particular property. The decision agent will read the error, and perhaps try to call the query tool again. Upon seeing the previous error in the error history, the query LLM agent should see that it should instead use a different filter property type, and correct itself.

## Advanced Tool Methods

### `run_if_true`

You can optionally choose to add another method to your Tool - `run_if_true`. This is a method that will be checked at the _start_ of every decision tree, for every tool that has this method. If you don't wish to use this method, then simply do not define one.

The `run_if_true` method returns two arguments (`tuple[bool, dict]`):

- a boolean value indicating whether the tool should be called *straight away*,
- a dictionary of `inputs` for if this tool gets called.

If `run_if_true` returns `True`, then the `__call__` method of your tool will be called and carried out regardless of if the LLM wishes to use this tool or not. It is a hardcoded rule to run the tool. Some potential examples of using this include:

- The `run_if_true` method can count the number of tokens in the environment, and if the environment is getting too large, it runs the tool. Then the `__call__` method will be shrinking the environment in some way (e.g. using an LLM or just taking one particular item from it).
- If the user is asking about a particular subject, e.g. if the `user_prompt` (inside of `tree_data`) contains a specific word, then you could augment the `tree_data` to include some more specific information.

```python
async def run_if_true(
    self,
    tree_data,
    base_lm,
    complex_lm,
    client_manager,
) -> tuple[bool, dict]:
    ...
```

Like the `__call__` and `is_tool_available` methods, this method has access to [the tree data](../Reference/Objects.md#elysia.tree.objects.TreeData) object, as well as some language models used by the tree and the [ClientManager](../Reference/Client.md#elysia.util.client.ClientManager), to use a Weaviate client.


[See the reference for more details.](../Reference/Objects.md#elysia.objects.Tool.run_if_true)

### `is_tool_available`

This method should return `True` if the tool is available to be used by the LLM. It should return `False` if the LLM should not have access to it. This can depend on the environment. For example, you can use `tree_data.environment.is_empty()` and the tool is only accessible if the environment is empty. Likewise you can use `not tree_data.environment.is_empty()` for it only to be available if the environment has something in it.

```python
async def is_tool_available(
    self,
    tree_data,
    base_lm,
    complex_lm,
    client_manager,
) -> bool:
    """A brief reason when this tool will become available goes here."""
    ...
```

Like the `__call__` and `run_if_true` methods, this method has access to [the tree data](../Reference/Objects.md#elysia.tree.objects.TreeData) object, as well as some language models used by the tree and the [ClientManager](../Reference/Client.md#elysia.util.client.ClientManager), to use a Weaviate client.

You should give a brief reason in the docstring of `is_tool_available` as to when it will become available, so that the LLM can perform actions towards completing this goal if it judges the tool to be useful to the current prompt.

[See the reference for more details.](../Reference/Objects.md#elysia.objects.Tool.is_tool_available) 

## Example: Text Response (basic)

Consider the generic text response tool that Elysia will use if the conversation ends without a sufficient answer.

```python
import dspy
from elysia.objects import Response, Tool
from elysia.tree.objects import TreeData
from elysia.util.client import ClientManager
from elysia.tools.text.prompt_templates import TextResponsePrompt
from elysia.util.elysia_chain_of_thought import ElysiaChainOfThought

class TextResponse(Tool):
    def __init__(self, **kwargs):
        super().__init__(
            name="final_text_response",
            description="",
            status="Writing response...",
            inputs={},
            end=True,
        )

    async def __call__(
        self,
        tree_data: TreeData,
        inputs: dict,
        base_lm: dspy.LM,
        complex_lm: dspy.LM,
        client_manager: ClientManager | None = None,
        **kwargs
    ):
        text_response = ElysiaChainOfThought(
            TextResponsePrompt,
            tree_data=tree_data,
            environment=True,
            tasks_completed=True,
            message_update=False,
        )

        output = await text_response.aforward(
            lm=base_lm,
        )

        yield Response(text=output.response)
```

The tool is simple, it is initialised and the descriptions are added to the Tool. Then the `__call__` method simply runs the text_response agent. Whilst the `TextResponsePrompt` is not shown here, it is a simple input -> output call, where different parts of the `tree_data` are used as inputs to the LLM to give it context before answering. The relevant information from the `tree_data` are automatically inserted into the prompt via the `ElysiaChainOfThought` custom DSPy module.


**Note:** If using DSPy within your tool, make sure to always call `aforward` method on the module so that it can be used async.


## Example: Dealing Cards Randomly from a Deck (Intermediate)

Let's create a tool that deals cards, adds them to the environment and displays them on the frontend.

Just for fun, these cards, when they are dealt, change the Elysia conversation somewhat, by some modifiers we will define ourselves.

```python
import random
from elysia import Tool
from elysia.tools import Ecommerce

class DealCards(Tool):
    def __init__(self, **kwargs):
        super().__init__(
            name="deal_cards",
            description="""
            This tool should always be run at the start of any interaction with the user.
            It defines any modifiers that get added to the conversation, from a random pool.
            It does so by 'dealing cards' to the user as if they were a player of a card game.
            These then will be displayed.
            Call this tool when specifically asked for, or at the start of any conversation.
            """,
            status="Dealing cards...",
            inputs={
                "num_cards": {
                    "description": "The number of cards to deal",
                    "type": int,
                    "default": 3,
                    "required": False,
                }
            },
            end=False,
        )
        self.logger = kwargs.get("logger", None)

    def select_random_cards(self, num_cards=3):
        possible_cards = [
            {
                "title": "The Jumbled",
                "effect": "Sometimes, the Elysia agent will say words in the wrong order.",
                "rarity": 3,
                "image": "https://i.imgur.com/KdGeZTp.png",
            },
            {
                "title": "The Comedian",
                "effect": "At the end of every sentence, the Elysia agent will tell a joke.",
                "rarity": 2,
                "image": "https://i.imgur.com/I8yVXHa.png",
            },
            {
                "title": "The Sarcastic",
                "effect": "Most interactions end with the Elysia agent making a sarcastic remark.",
                "rarity": 1,
                "image": "https://i.imgur.com/oFkwt1M.png",
            },
            {
                "title": "The Bro",
                "effect": "Elysia must now use the word 'bro' a lot more, and apply similar slang everywhere.",
                "rarity": 1,
                "image": "https://i.imgur.com/J6dLbTZ.png",
            },
            {
                "title": "The Philosopher",
                "effect": "The Elysia agent will now try to philosophise at every opportunity.",
                "rarity": 3,
                "image": "https://i.imgur.com/D6VSitF.png",
            },
        ]
        return random.choices(
            possible_cards, weights=[1 / card["rarity"] for card in possible_cards], k=3
        )

    async def __call__(self, tree_data, inputs, base_lm, complex_lm, client_manager, **kwargs):
        self.logger.info(f"Dealing {inputs['num_cards']} cards")
        cards = self.select_random_cards(inputs["num_cards"])

        yield Ecommerce(
            objects=cards,
            mapping={
                "description": "effect",
                "name": "title",
                "price": "rarity",
                "image": "image",
            },
            metadata={
                "num_cards": inputs["num_cards"],
            },
            name="cards",
            llm_message="""
            Cards have been successfully dealt for this prompt! 
            Dealt {num_cards} out of a possible 5.
            Look at them in the environment to apply their modifiers to the conversation.
            Pay attention to what these cards do, and how they affect the conversation.
            You should apply the modifiers together, in a combination, not only one at a time.
            """,
        )
```

Let's break down the different components of this tool.

1. In the `__init__`, we gave the tool name, a hefty description as well as a single input - the number of cards to deal. Make sure you provide the `**kwargs` argument also.
2. There is a custom method that randomly chooses `num_cards` out of 5 possible cards, hand-written.
3. 
    - The `__call__` method, when the tool gets chosen, simply calls the `select_random_cards` method with the input that has come from the decision agent. 
    - Then it yields an `Ecommerce` object (placeholder) which will display the card. 
    - Since the `Ecommerce` object has pre-defined fields, to choose which of the card's fields go where, the `mapping` places the card `effect` in the Ecommerce `description`, the card `title` in place of the `name` field, the `rarity` becomes the `price` and the image field name is the same, but it is mapped anyway.
    - The `llm_message` argument of the Ecommerce `Result`, describes what happens to the LLM whenever this tool is completed. This `llm_message` is persistent through further calls in Elysia, it will remain there for all future events in this conversation. In this case, it re-iterates the point that the cards add custom modifiers, and shows how many cards were dealt to the user at this point.

We could add more features to this card, for example, modifying the `tree_data.environment` object to find any existing cards in the environment (with the name "cards") and overwriting them with the new deal.

