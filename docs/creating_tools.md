# Creating a Tool

You can use the custom tool decorator within Elysia to very simply add a tool to the tree. For example:

```python
from elysia import tool

@tool
async def add(x: int, y: int) -> int:
    """
    Return the sum of two numbers.
    """
    return x + y
```

The docstring of the function serves as the tool description, and it's important this is as detailed as possible. This tool can be added to an Elysia `Tree` via

```python 
from elysia import Tree
tree = Tree()
tree.add_tool(add)
```

Then when calling the tree, the decision agent should use the tool if it recognises it as the best tool for the task.

```python
response, objects = tree("What is 1238213 + 1238213?")
print(response)
```
```
'I will calculate the sum for you using the add tool. The sum of 1238213 + 1238213 is 2476426.'
```

And this is all you need to do to add a tool to Elysia! Some things to note:

- Your tool must be an async function (must be defined via `async def` instead of `def`).
- `tree.add_tool(add)` added this tool to the _root_ decision node (at the base of the tree). If you are using a tree with multiple branches, you can specify which branch it is added to via `tree.add_tool(add, branch_id=...)` where `...` should be replaced with the `branch_id`.
- You can add a tool to the tree automatically via customising the decorator function, e.g.
    ```python
    @tool(tree=tree, branch_id="base")
    async def add(x: int, y: int) -> int:
        return x + y    
    ```
    which will automatically add it to a pre-defined tree (`tree`) at branch ID `"base"`.
- Type hinting (e.g. declaring `x: int` and `y: int`) helps the LLM choose the correct input types to the function.


## More Detail

Elysia works by adding objects to its internal environment. For example, when we called the `add` function above, it automatically added a `Result` type object to the Elysia tree environment. Any objects directly returned by the function will be added to the environment under a generic set of keys. To have more control over this, you can create your tool as an async generator function, which yields objects. For example, let's extend our basic calculator a bit further:

```python
@tool
async def calculate_two_numbers(x: int, y: int):
    """
    This function calculates the sum, product, and difference of two numbers.
    """
    yield {
        "sum": x + y,
        "product": x * y,
        "difference": x - y,
    }
    yield f"I just performed some calculations on {x} and {y}."
```

This now returns _two_ items to the decision tree, a string and a dictionary. There is no limit to the amount of objects you can yield. 

When a string is returned, it automatically becomes a _response_ from Elysia, so it will be displayed to the user as if the agent is talking back to them. When any other type of item is yielded, it becomes a `Result` type object which means it becomes part of the tree's environment. Yielding or returning a dictionary means you can customise what specific object is added to the environment. Returning or yielding a list of dictionaries will add multiple objects to the `Result`. You can also return or yield one or more `Result` objects directly.

## Assigning Inputs

The decision agent LLM is responsible for choosing the correct inputs to the tool. Any inputs added to the declaration of your tool will be automatically chosen by the LLM. Let's extend the calculator even more:

```python
from math import prod
@tool
async def perform_mathematical_operations(numbers: list[int | float], operation: str = "sum"):
    """
    This function calculates a mathematical operation on the `numbers` list.
    The `numbers` input must be a list of integers or floats.
    The `operation` input must be one of: "sum" or "product". These are the only options.
    """

    if operation == "sum":
        yield sum(numbers)
    elif operation == "product":
        yield prod(numbers)

    yield f"I just performed a {operation} on {numbers}."    
```

Now the LLM should choose the operation in addition to the numbers. We also extended it so that the values can be a list of integers or floats, not just two numbers. The default argument, indicated by `operation: str = "sum"`, give the decision agent awareness of what the default argument for that particular input is - and it is no longer a required input and can be ignored, in which case the default argument is used. Note how the tool description details the descriptions of each input. [In more advanced tool construction](Advanced/advanced_tool_construction.md), you can assign descriptions to each input separately.


## Advanced Features


If your tool may error, then you can return or yield a custom Elysia `Error` object which will not cause a halt in the execution of the program. Instead, the error message will be logged in the decision tree for which the decision agent can judge whether the error is avoidable on another run of the tool. For example, if our decision agent tries to choose the wrong `operation` in the above `perform_mathematical_operations` tool, we can do something like this:
```python
from elysia import Error
@tool
async def perform_mathematical_operations(numbers: list[int | float], operation: str = "sum"):
    """
    This function calculates a mathematical operation on the `numbers` list.
    The `numbers` input must be a list of integers or floats.
    The `operation` input must be one of: "sum" or "product". These are the only options.
    """

    if operation == "sum":
        yield sum(numbers)
    elif operation == "product":
        yield prod(numbers)
    else:
        # This will return an error back to the decision tree
        yield Error(f"You picked the input {operation}, but it was not in the available operations: 'sum' or 'product'")
        return # Then return out of the tool early

    yield f"I just performed a {operation} on {numbers}."    
```

Finally, tools can interact with Elysia's environment, LMs and the Weaviate client through specific inputs to the function. To use the [`TreeData` class](Reference/Objects.md#elysia.tree.objects.TreeData), you can use the argument `tree_data` in the function signature (for which you can access the [Elysia environment](Advanced/environment.md)). Likewise, for the base LM you can use `base_lm`, for the complex LM you can use `complex_lm` and for the [Client Manager](Reference/Client.md) you can use `client_manager`. For example:

```python
@tool
async def some_tool(
    tree_data, base_lm, complex_lm, client_manager, # these inputs are automatically assigned as Elysia variables
    x: str, y: int # these inputs are not assigned automatically and get assigned by the decision agent
):
    # do something
    pass
```

All optional arguments you can pass to the `@tool` decorator are:

- `tree` ([`Tree`](Reference/Tree.md#elysia.tree.tree.Tree)): the tree that you will automatically add the tool to.
- `branch_id` (`str`): the ID of the branch on the tree to add the tool to.
- `status` (`str`): a custom message to display whilst the tool is running.
- `end` (`bool`): when `True`, this tool can be the end of the conversation if the decision agent decides it should end after the completion of this tool.
