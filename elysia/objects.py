import uuid
import inspect
import ast
from typing import Any, AsyncGenerator, Callable, TYPE_CHECKING, overload

if TYPE_CHECKING:
    from elysia.tree.tree import Tree


class ToolMeta(type):
    """Metaclass that extracts tool metadata from __init__ method."""

    _tool_name: str | None = None
    _tool_description: str | None = None
    _tool_inputs: dict | None = None
    _tool_end: bool | None = None

    @staticmethod
    def _convert_ast_dict(ast_dict: ast.Dict) -> dict:
        out = {}
        for i in range(len(ast_dict.keys)):
            k: ast.Constant = ast_dict.keys[i]  # type: ignore
            v: ast.Dict | ast.List | ast.Constant = ast_dict.values[i]  # type: ignore

            if isinstance(v, ast.Dict):
                out[k.value] = ToolMeta._convert_ast_dict(v)
            elif isinstance(v, ast.List):
                out[k.value] = ToolMeta._convert_ast_list(v)
            elif isinstance(v, ast.Constant):
                out[k.value] = v.value
        return out

    @staticmethod
    def _convert_ast_list(ast_list: ast.List) -> list:
        out = []
        for v in ast_list.values:  # type: ignore
            if isinstance(v, ast.Dict):
                out.append(ToolMeta._convert_ast_dict(v))
            elif isinstance(v, ast.List):
                out.append(ToolMeta._convert_ast_list(v))
            elif isinstance(v, ast.Constant):
                out.append(v.value)
        return out

    def __new__(cls, name, bases, namespace):
        new_class = super().__new__(cls, name, bases, namespace)

        # Skip the base Tool class itself
        if name == "Tool":
            return new_class

        # Extract metadata from __init__ method if it exists
        if "__init__" in namespace:
            try:
                init_method = namespace["__init__"]
                source = inspect.getsource(init_method)
                tree = ast.parse(source.strip())

                # Find the super().__init__() call and extract arguments
                for node in ast.walk(tree):
                    if (
                        isinstance(node, ast.Call)
                        and isinstance(node.func, ast.Attribute)
                        and node.func.attr == "__init__"
                        and isinstance(node.func.value, ast.Call)
                        and isinstance(node.func.value.func, ast.Name)
                        and node.func.value.func.id == "super"
                    ):

                        # Extract keyword arguments
                        for keyword in node.keywords:
                            if keyword.arg == "name" and isinstance(
                                keyword.value, ast.Constant
                            ):
                                new_class._tool_name = keyword.value.value
                            elif keyword.arg == "description" and isinstance(
                                keyword.value, ast.Constant
                            ):
                                new_class._tool_description = keyword.value.value
                            elif keyword.arg == "inputs" and isinstance(
                                keyword.value, ast.Dict
                            ):
                                new_class._tool_inputs = ToolMeta._convert_ast_dict(
                                    keyword.value
                                )
                            elif keyword.arg == "end" and isinstance(
                                keyword.value, ast.Constant
                            ):
                                new_class._tool_end = keyword.value.value
                        break
            except Exception:
                # If parsing fails, just continue without setting class attributes
                pass

        return new_class


class Tool(metaclass=ToolMeta):
    """
    The generic Tool class, which will be a superclass of any tools used by Elysia.
    """

    @classmethod
    def get_metadata(cls) -> dict[str, str | bool | dict | None]:
        """Get tool metadata without instantiation."""
        return {
            "name": getattr(cls, "_tool_name", None),
            "description": getattr(cls, "_tool_description", None),
            "inputs": getattr(cls, "_tool_inputs", None),
            "end": getattr(cls, "_tool_end", False),
        }

    def __init__(
        self,
        name: str,
        description: str,
        status: str = "",
        inputs: dict = {},
        end: bool = False,
        **kwargs,
    ):
        """
        Args:
            name (str): The name of the tool. Required.
            description (str): A detailed description of the tool to give to the LLM. Required.
            status (str): A status update message to display while the tool is running. Optional, defaults to "Running {name}...".
            inputs (dict): A dictionary of inputs for the tool, with the following structure:
                ```python
                {
                    input_name: {
                        "description": str,
                        "type": str,
                        "default": str,
                        "required": bool
                    }
                }
                ```
            end (bool): Whether the tool is an end tool. Optional, defaults to False.
        """
        self.name = name
        self.description = description
        self.inputs = inputs
        self.end = end

        if status == "":
            self.status = f"Running {self.name}..."
        else:
            self.status = status

    def get_default_inputs(self) -> dict:
        return {
            key: value["default"] if "default" in value else None
            for key, value in self.inputs.items()
        }

    async def run_if_true(
        self,
        tree_data,
        base_lm,
        complex_lm,
        client_manager,
    ) -> tuple[bool, dict]:
        """
        This method is called to check if the tool should be run automatically.
        If this returns `True`, then the tool is immediately called at the start of the tree.
        Otherwise it does not appear in the decision tree.
        You must also return a dictionary of inputs for the tool, which will be used to call the tool if `True` is returned.
        If the inputs are None or an empty dictionary, then the default inputs will be used.

        This must be an async function.

        Args:
            tree_data (TreeData): The tree data object.
            base_lm (LM): The base language model, a dspy.LM object.
            complex_lm (LM): The complex language model, a dspy.LM object.
            client_manager (ClientManager): The client manager, a way of interfacing with a Weaviate client.

        Returns:
            bool: True if the tool should be run automatically, False otherwise.
            dict: A dictionary of inputs for the tool, with the following structure:
                ```python
                {
                    input_name: input_value
                }
                ```
        """
        return False, {}

    async def is_tool_available(
        self,
        tree_data,
        base_lm,
        complex_lm,
        client_manager,
    ) -> bool:
        """
        This method is called to check if the tool is available.
        If this returns `True`, then the tool is available to be used by the LLM.
        Otherwise it does not appear in the decision tree.

        The difference between this and `run_if_true` is that `run_if_true` will _always_ run the __call__ method if it returns `True`,
        even if the LLM does not choose to use the tool.

        This must be an async function.

        Args:
            tree_data (TreeData): The tree data object.
            base_lm (LM): The base language model, a dspy.LM object.
            complex_lm (LM): The complex language model, a dspy.LM object.
            client_manager (ClientManager): The client manager, a way of interfacing with a Weaviate client.

        Returns:
            bool: True if the tool is available, False otherwise.
        """
        return True

    async def __call__(
        self, tree_data, inputs, base_lm, complex_lm, client_manager, **kwargs
    ) -> AsyncGenerator[Any, None]:
        """
        This method is called to run the tool.

        This must be an async generator, so objects must be yielded and not returned.

        Args:
            tree_data (TreeData): The data from the decision tree, includes the environment, user prompt, etc.
                See the `TreeData` class for more details.
            base_lm (LM): The base language model, a dspy.LM object.
            complex_lm (LM): The complex language model, a dspy.LM object.
            client_manager (ClientManager): The client manager, a way of interfacing with a Weaviate client.
        """
        yield None


@overload
def tool(
    function: Callable,
    *,
    status: str | None = None,
    end: bool = False,
    tree: "Tree | None" = None,
    branch_id: str | None = None,
) -> Tool: ...


@overload
def tool(
    function: None = None,
    *,
    status: str | None = None,
    end: bool = False,
    tree: "Tree | None" = None,
    branch_id: str | None = None,
) -> Callable[[Callable], Tool]: ...


def tool(
    function: Callable | None = None,
    *,
    status: str | None = None,
    end: bool = False,
    tree: "Tree | None" = None,
    branch_id: str | None = None,
) -> Callable[[Callable], Tool] | Tool:
    """
    Create a tool from a python function.
    Use this decorator to create a tool from a function.
    The function must be an async function or async generator function.

    Args:
        function (Callable): The function to create a tool from.
        status (str | None): The status message to display while the tool is running.
            Optional, defaults to None, which will use the default status message "Running {tool_name}...".
        end (bool): Whether the tool can be at the end of the decision tree.
            Set to True when this tool is allowed to end the conversation.
            Optional, defaults to False.
        tree (Tree | None): The tree to add the tool to.
            Optional, defaults to None, which will not add the tool to the tree.
        branch_id (str | None): The branch of the tree to add the tool to.
            Optional, defaults to None, which will add the tool to the root branch.

    Returns:
        (Tool): The tool object which can be added to the tree (via `tree.add_tool(...)`).
    """

    def decorator(function: Callable) -> Tool:
        async_function = inspect.iscoroutinefunction(function)
        async_generator_function = inspect.isasyncgenfunction(function)

        if not async_function and not async_generator_function:
            raise TypeError(
                "The provided function must be an async function or async generator function."
            )

        if "inputs" in list(function.__annotations__.keys()):
            raise TypeError(
                "The `inputs` argument is reserved in tool functions, please choose another name."
            )

        sig = inspect.signature(function)
        defaults_mapping = {
            k: v.default
            for k, v in sig.parameters.items()
            if v.default is not inspect.Parameter.empty
        }

        def list_to_list_of_dicts(result: list) -> list[dict]:
            objects = []
            for obj in result:
                if isinstance(obj, dict):
                    objects.append(obj)
                elif isinstance(obj, int | float | bool):
                    objects.append(
                        {
                            "tool_result": obj,
                        }
                    )
                elif isinstance(obj, list):
                    objects.append(list_to_list_of_dicts(obj))
                else:
                    objects.append(
                        {
                            "tool_result": obj,
                        }
                    )
            return objects

        def return_mapping(result, inputs: dict):
            if isinstance(result, Result | Text | Update | Status | Error):
                return result
            elif isinstance(result, str):
                return Response(result)
            elif isinstance(result, int | float | bool):
                return Result(
                    objects=[
                        {
                            "tool_result": result,
                        }
                    ],
                    metadata={
                        "tool_name": function.__name__,
                        "inputs_used": inputs,
                    },
                )
            elif isinstance(result, dict):
                return Result(
                    objects=[result],
                    metadata={
                        "tool_name": function.__name__,
                        "inputs_used": inputs,
                    },
                )
            elif isinstance(result, list):
                return Result(
                    objects=list_to_list_of_dicts(result),
                    metadata={
                        "tool_name": function.__name__,
                        "inputs_used": inputs,
                    },
                )
            else:
                return Result(
                    objects=[
                        {
                            "tool_result": result,
                        }
                    ],
                    metadata={
                        "tool_name": function.__name__,
                        "inputs_used": inputs,
                    },
                )

        class ToolClass(Tool):
            def __init__(self, **kwargs):
                self._original_function = function
                self._original_function_args = {
                    arg: None
                    for arg in function.__code__.co_varnames[
                        : function.__code__.co_argcount
                    ]
                }
                for arg in function.__annotations__:
                    if arg in self._original_function_args:
                        self._original_function_args[arg] = function.__annotations__[
                            arg
                        ]
                super().__init__(
                    name=function.__name__,
                    description=function.__doc__ or "",
                    status=(
                        status
                        if status is not None
                        else f"Running {function.__name__}..."
                    ),
                    inputs={
                        input_key: {
                            "description": "",
                            "type": (
                                "Not specified" if input_value is None else input_value
                            ),
                            "default": defaults_mapping.get(input_key, None),
                            "required": defaults_mapping.get(input_key, None)
                            is not None,
                        }
                        for input_key, input_value in self._original_function_args.items()
                        if input_key
                        not in [
                            "tree_data",
                            "base_lm",
                            "complex_lm",
                            "client_manager",
                            "return",
                        ]
                    },
                    end=end,
                )

            async def __call__(
                self, tree_data, inputs, base_lm, complex_lm, client_manager, **kwargs
            ):

                if async_function:
                    tool_output = [
                        await function(
                            **inputs,
                            **{
                                k: v
                                for k, v in {
                                    "tree_data": tree_data,
                                    "base_lm": base_lm,
                                    "complex_lm": complex_lm,
                                    "client_manager": client_manager,
                                    **kwargs,
                                }.items()
                                if k in self._original_function_args
                            },
                        )
                    ]
                elif async_generator_function:
                    results = []
                    async for result in function(
                        **inputs,
                        **{
                            k: v
                            for k, v in {
                                "tree_data": tree_data,
                                "base_lm": base_lm,
                                "complex_lm": complex_lm,
                                "client_manager": client_manager,
                                **kwargs,
                            }.items()
                            if k in self._original_function_args
                        },
                    ):
                        results.append(result)
                    tool_output = results

                for result in tool_output:
                    mapped_result = return_mapping(result, inputs)
                    yield mapped_result

        tool_class = ToolClass()

        if tree is not None:
            tree.add_tool(tool_class, branch_id=branch_id)

        return tool_class

    if function is not None:
        return decorator(function)
    else:
        return decorator


class Return:
    """
    A Return object is something that is returned to the frontend.
    This kind of object is frontend-aware.
    """

    def __init__(self, frontend_type: str, payload_type: str):
        self.frontend_type = frontend_type  # frontend identifier
        self.payload_type = payload_type  # backend identifier


class Text(Return):
    """
    Frontend Return Type 1: Text
    Objects is usually a one-element list containing a dict with a "text" key only.
    But is not limited to this.
    """

    def __init__(
        self,
        payload_type: str,
        objects: list[dict],
        metadata: dict = {},
        display: bool = True,
    ):
        Return.__init__(self, "text", payload_type)
        self.objects = objects
        self.metadata = metadata
        self.text = self._concat_text(self.objects)
        self.display = display

    def _concat_text(self, objects: list[dict]):
        text = ""
        for i, obj in enumerate(objects):
            if "text" in obj:
                spacer = ""
                if (
                    not obj["text"].endswith(" ")
                    and not obj["text"].endswith("\n")
                    and i != len(objects) - 1
                ):
                    spacer += " "

                if (
                    i != len(objects) - 1
                    and "text" in objects[i + 1]
                    and objects[i + 1]["text"].startswith("* ")
                    and not obj["text"].endswith("\n")
                ):
                    spacer += "\n"

                text += obj["text"] + spacer

        text = text.replace("_REF_ID", "")
        text = text.replace("REF_ID", "")
        text = text.replace("  ", " ")
        return text

    def to_json(self):
        return {
            "type": self.payload_type,
            "metadata": self.metadata,
            "objects": self.objects,
        }

    async def to_frontend(
        self,
        user_id: str,
        conversation_id: str,
        query_id: str,
    ):
        if not self.display:
            return

        return {
            "type": self.frontend_type,
            "id": self.frontend_type[:3] + "-" + str(uuid.uuid4()),
            "user_id": user_id,
            "conversation_id": conversation_id,
            "query_id": query_id,
            "payload": self.to_json(),
        }


class Response(Text):
    def __init__(self, text: str, **kwargs):
        Text.__init__(self, "response", [{"text": text}], **kwargs)


class Update(Return):
    """
    Frontend Return Type 2: Update
    An update is something that is not _displayed_ on the frontend, but gives information to the frontend.
    E.g. a warning, error, status message, etc.
    """

    def __init__(self, frontend_type: str, object: dict):
        Return.__init__(self, frontend_type, "update")
        self.object = object

    def to_json(self):
        return self.object

    async def to_frontend(self, user_id: str, conversation_id: str, query_id: str):
        return {
            "type": self.frontend_type,
            "user_id": user_id,
            "conversation_id": conversation_id,
            "query_id": query_id,
            "id": self.frontend_type[:3] + "-" + str(uuid.uuid4()),
            "payload": self.to_json(),
        }


class Status(Update):
    """
    Status message to be sent to the frontend for real-time updates in words.
    """

    def __init__(self, text: str):
        self.text = text
        Update.__init__(self, "status", {"text": text})


class Warning(Update):
    """
    Warning message to be sent to the frontend.
    """

    def __init__(self, text: str):
        self.text = text
        Update.__init__(self, "warning", {"text": text})


class Completed(Update):
    """
    Completed message to be sent to the frontend (tree is complete all recursions).
    """

    def __init__(self):
        Update.__init__(self, "completed", {})


class Result(Return):
    """
    Result objects are returned to the frontend.
    These are displayed on the frontend.
    E.g. a table, a chart, a text response, etc.

    You can yield a `Result` directly, and specify the `type` and `name` of the result.
    """

    def __init__(
        self,
        objects: list[dict],
        metadata: dict = {},
        payload_type: str = "default",
        name: str = "default",
        mapping: dict | None = None,
        llm_message: str | None = None,
        unmapped_keys: list[str] = ["_REF_ID"],
        display: bool = True,
    ):
        """
        Args:
            objects (list[dict]): The objects attached to this result.
            payload_type: (str): Identifier for the type of result.
            metadata (dict): The metadata attached to this result,
                for example, query used, time taken, any other information not directly within the objects
            name (str): The name of the result, e.g. this could be the name of the collection queried.
                Used to index the result in the environment.
            mapping (dict | None): A mapping of the objects to the frontend.
                Essentially, if the objects are going to be displayed on the frontend, the frontend has specific keys that it wants the objects to have.
                This is a dictionary that maps from frontend keys to the object keys.
            llm_message (str | None): A message to be displayed to the LLM to help it parse the result.
                You can use the following placeholders that will automatically be replaced with the correct values:

                - {type}: The type of the object
                - {name}: The name of the object
                - {num_objects}: The number of objects in the object
                - {metadata_key}: Any key in the metadata dictionary
            unmapped_keys (list[str]): A list of keys that are not mapped to the frontend.
            display (bool): Whether to display the result on the frontend when yielding this object.
                Defaults to `True`.
        """
        Return.__init__(self, "result", payload_type)
        self.objects = objects
        self.metadata = metadata
        self.name = name
        self.mapping = mapping
        self.llm_message = llm_message
        self.unmapped_keys = unmapped_keys
        self.display = display

    def __len__(self):
        return len(self.objects)

    def format_llm_message(self):
        """
        llm_message is a string that is used to help the LLM parse the output of the tool.
        It is a placeholder for the actual message that will be displayed to the user.

        You can use the following placeholders:

        - {payload_type}: The type of the object
        - {name}: The name of the object
        - {num_objects}: The number of objects in the object
        - {metadata_key}: Any key in the metadata dictionary
        """
        if self.llm_message is None:
            return ""

        return self.llm_message.format_map(
            {
                "payload_type": self.payload_type,
                "name": self.name,
                "num_objects": len(self),
                **self.metadata,
            }
        )

    def do_mapping(self, objects: list[dict]):

        if self.mapping is None:
            return objects

        output_objects = []
        for obj in objects:
            output_objects.append(
                {
                    key: obj[self.mapping[key]]
                    for key in self.mapping
                    if self.mapping[key] != ""
                }
            )
            output_objects[-1].update(
                {key: obj[key] for key in self.unmapped_keys if key in obj}
            )

        return output_objects

    def to_json(self, mapping: bool = False):
        """
        Convert the result to a list of dictionaries.

        Args:
            mapping (bool): Whether to map the objects to the frontend.
                This will use the `mapping` dictionary created on initialisation of the `Result` object.
                If `False`, the objects are returned as is.
                Defaults to `False`.

        Returns:
            (list[dict]): A list of dictionaries, which can be serialised to JSON.
        """
        from elysia.util.parsing import format_dict_to_serialisable

        assert all(
            isinstance(obj, dict) for obj in self.objects
        ), "All objects must be dictionaries"

        if mapping and self.mapping is not None:
            output_objects = self.do_mapping(self.objects)
        else:
            output_objects = self.objects

        for object in output_objects:
            format_dict_to_serialisable(object)

        return output_objects

    async def to_frontend(
        self,
        user_id: str,
        conversation_id: str,
        query_id: str,
    ):
        """
        Convert the result to a frontend payload.
        This is a wrapper around the `to_json` method.
        But the objects and metadata are inside a `payload` key, which also includes a `type` key,
        being the frontend identifier for the type of payload being sent.
        (e.g. `ticket`, `product`, `message`, etc.)

        The outside of the payload also contains the user_id, conversation_id, and query_id.

        Args:
            user_id (str): The user ID.
            conversation_id (str): The conversation ID.
            query_id (str): The query ID.

        Returns:
            (dict): The frontend payload, which is a dictionary with the following structure:
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
        """
        if not self.display:
            return

        objects = self.to_json(mapping=True)
        if len(objects) == 0:
            return

        payload = {
            "type": self.payload_type,
            "objects": objects,
            "metadata": self.metadata,
        }

        return {
            "type": self.frontend_type,
            "user_id": user_id,
            "conversation_id": conversation_id,
            "query_id": query_id,
            "id": self.frontend_type[:3] + "-" + str(uuid.uuid4()),
            "payload": payload,
        }

    def llm_parse(self):
        """
        This method is called when the result is displayed to the LLM.
        It is used to display custom information to the LLM about the result.
        If `llm_message` was defined, then the llm message is formatted using the placeholders.
        Otherwise a default message is used.

        Returns:
            (str): The formatted llm message.
        """

        if self.llm_message is not None:
            return self.format_llm_message()
        else:
            # Default message
            return f"Displayed: A {self.payload_type} object with {len(self.objects)} objects."


# Retrieval/Aggregation/Anything we used code to get data
class Retrieval(Result):
    """
    Store of returned objects from a query/aggregate/any displayed results.
    """

    def __init__(
        self,
        objects: list[dict],
        metadata: dict = {},
        payload_type: str = "default",
        name: str | None = None,
        mapping: dict | None = None,
        unmapped_keys: list[str] = [
            "uuid",
            "ELYSIA_SUMMARY",
            "collection_name",
            "_REF_ID",
        ],
        display: bool = True,
    ) -> None:
        if name is None and "collection_name" in metadata:
            result_name = metadata["collection_name"]
        elif name is None:
            result_name = "default"
        else:
            result_name = name

        Result.__init__(
            self,
            objects=objects,
            payload_type=payload_type,
            metadata=metadata,
            name=result_name,
            mapping=mapping,
            unmapped_keys=unmapped_keys,
            display=display,
        )

    def add_summaries(self, summaries: list[str] = []) -> None:
        for i, obj in enumerate(self.objects):
            if i < len(summaries):
                obj["ELYSIA_SUMMARY"] = summaries[i]
            else:
                obj["ELYSIA_SUMMARY"] = ""

    def llm_parse(self) -> str:
        out = ""
        count = len(self.objects)

        if "collection_name" in self.metadata:
            if count != 0:
                out += f"\nQueried collection: '{self.metadata['collection_name']}' and returned {count} objects, "
            else:
                out += f"\nQueried collection: '{self.metadata['collection_name']}' but no objects were returned."
                out += f" Since it had no objects, judge the query that was created, and evaluate whether it was appropriate for the collection, the user prompt, and the data available."
                out += f" If it seemed innappropriate, you can choose to try again if you think it can still be completed (or there is more to do)."
        if "return_type" in self.metadata:
            out += f"\nReturned with type '{self.metadata['return_type']}', "
        if "output_type" in self.metadata:
            out += f"\noutputted '{'itemised summaries' if self.metadata['output_type'] == 'summary' else 'original objects'}'\n"
        if "query_text" in self.metadata:
            out += f"\nSearch terms: '{self.metadata['query_text']}'"
        if "query_type" in self.metadata:
            out += f"\nType of query: '{self.metadata['query_type']}'"
        if "impossible" in self.metadata:
            out += f"\nImpossible prompt: '{self.metadata['impossible']}'"
            if "collection_name" in self.metadata:
                out += f"\nThis attempt at querying the collection: {self.metadata['collection_name']} was deemed impossible."
            if "impossible_reason" in self.metadata:
                out += f"\nReasoning for impossibility: {self.metadata['impossible_reason']}"
        if "query_output" in self.metadata:
            out += f"\nThe query used was:\n{self.metadata['query_output']}"
        return out

    async def to_frontend(
        self,
        user_id: str,
        conversation_id: str,
        query_id: str,
    ) -> dict | None:
        objects = self.to_json(mapping=True)
        if len(objects) == 0:
            return

        payload = {
            "type": self.payload_type,
            "objects": objects,
            "metadata": self.metadata,
        }

        if "code" in self.metadata:
            payload["code"] = self.metadata["code"]

        return {
            "type": self.frontend_type,
            "user_id": user_id,
            "conversation_id": conversation_id,
            "query_id": query_id,
            "id": self.frontend_type[:3] + "-" + str(uuid.uuid4()),
            "payload": payload,
        }


class Error(Update):
    """
    Error objects are used to communicate errors to the decision agent/tool calls.
    When yielded, Error objects are automatically saved inside the TreeData object.
    When calling the same tool again, the saved Error object is automatically loaded into any tool calls made with the same tool name.
    All errors are shown to the decision agent to help decide whether the tool should be called again (retried), or a different tool should be called.
    """

    def __init__(self, feedback: str = "", error_message: str = ""):
        """
        Args:
            feedback (str): The feedback to display to the decision agent.
                Usually this will be the error message, but it could be formatted more specifically.
        """
        self.feedback = feedback
        self.error_message = error_message

        if feedback == "":
            self.feedback = "An unknown issue occurred."

        Update.__init__(
            self,
            "self_healing_error",
            {"feedback": self.feedback, "error_message": self.error_message},
        )
