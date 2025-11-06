from typing import Any
from logging import Logger
from datetime import datetime
from pydantic import BaseModel, Field

from elysia.config import Settings
from elysia.config import settings as environment_settings
from elysia.objects import Result
from elysia.util.client import ClientManager
from elysia.util.parsing import format_dict_to_serialisable, remove_whitespace
from copy import deepcopy
from weaviate.classes.query import Filter


class EnvironmentItem(BaseModel):
    metadata: dict
    objects: list[dict]

    def to_json(self):
        return self.model_dump()

    @classmethod
    def from_json(cls, json_data: dict):
        return cls(**json_data)


class Environment:
    """
    Store of all objects across different types of queries and responses.

    The environment is how the tree stores all objects across different types of queries and responses.
    This includes things like retrieved objects, retrieved metadata, retrieved summaries, etc.

    This is persistent across the tree, so that all agents are aware of the same objects.

    The environment is formatted and keyed as follows:
    ```python
    {
        <tool_name>: {
            list[dict] = [
                {
                    "metadata": dict,
                    "objects": list[dict],
                },
                ...
            ]
        }
    }
    ```

    Where each <tool_name> is a string denoting the name of the tool that the result belongs to,
    e.g. if the result came from a tool called `"query"`, then <tool_name> is `"query"`.
    Each list under <tool_name> is a dictionary with both `metadata` and `objects` keys.
    Each separate item in the list is a separate result from the tool (i.e. called at different times or multiple yields from the same tool).

    `"metadata"` is the metadata of the result,
    e.g. the time taken to retrieve the result, the query used, etc.
    This is customisable by the tool result itself.

    `"objects"` is the list of objects retrieved from the result. This is a list of dictionaries, where each dictionary is an object.
    It is important that `objects` should be list of dictionaries.
    e.g. each object that was returned from a retrieval, where the fields of each dictionary are the fields of the object returned.

    You can use various methods to add, remove, replace, and find objects in the environment. See the methods below for more information.

    Within the environment, there is a variable called `hidden_environment`, which is a dictionary of key-value pairs.
    This is used to store information that is not shown to the LLM, but is instead a 'store' of data that can be used across tools.
    """

    def __init__(
        self,
        environment: dict[str, list[EnvironmentItem]] | None = None,
        hidden_environment: dict[str, Any] = {},
    ):
        if environment is None:
            environment = {}
        self.environment = environment
        self.hidden_environment = hidden_environment

    def clear(self, include_hidden: bool = False):
        """
        Clears the environment.
        """
        self.environment = {}
        if include_hidden:
            self.hidden_environment = {}

    def is_empty(self, tool_name: str | None = None):
        """
        Check if the environment is empty.

        If the `.remove` method has been used, this is accounted for (e.g. empty lists count towards an empty environment).

        Args:
            tool_name (str | None): Optional. The name of the tool to check if it is empty.
                If provided, only the tool with the given name is checked.
                If `None`, the entire environment is checked.
                Defaults to `None`.
        """

        if tool_name:
            if tool_name not in self.environment:
                return True
            else:
                empty = True
                for item in self.environment[tool_name]:
                    if len(item.objects) > 0:
                        empty = False
                        break
                return empty
        else:
            empty = True
            for tool_name in self.environment.keys():
                for item in self.environment[tool_name]:
                    if len(item.objects) > 0:
                        empty = False
                        break
                if not empty:
                    break
            return empty

    def add(self, tool_name: str, result: Result, include_duplicates: bool = False):
        """
        Adds a result to the environment.
        Is called automatically by the tree when a result is returned from an agent.

        You can also add a result to the environment manually by using this method.
        In this case, you must be adding a 'Result' object (which has an implicit 'name' attribute used to key the environment).
        If you want to add something manually, you can use the `add_objects` method.

        Each item is added with a `_REF_ID` attribute, which is a unique identifier used to identify the object in the environment.
        If duplicate objects are detected, they are added with a duplicate `_REF_ID` entry detailing that they are a duplicate,
        as well as the `_REF_ID` of the original object.

        Args:
            tool_name (str): The name of the tool called that the result belongs to.
            result (Result): The result to add to the environment.
            include_duplicates (bool): Optional. Whether to include duplicate objects in the environment.
                Defaults to `False`, which still adds the duplicate object, but with a duplicate `_REF_ID` entry (and no repeating properties).
                If `True`, the duplicate object is added with a new `_REF_ID` entry, and the repeated properties are added to the object.
        """
        objects = result.to_json()
        metadata = result.metadata
        self.add_objects(tool_name, objects, metadata, include_duplicates)

    def add_objects(
        self,
        tool_name: str,
        objects: list[dict],
        metadata: dict = {},
        include_duplicates: bool = False,
    ):
        """
        Adds an object to the environment.
        Is not called automatically by the tree, so you must manually call this method.
        This is useful if you want to add an object to the environment manually that doesn't come from a Result object.

        Each item is added with a `_REF_ID` attribute, which is a unique identifier used to identify the object in the environment.
        If duplicate objects are detected, they are added with a duplicate `_REF_ID` entry detailing that they are a duplicate,
        as well as the `_REF_ID` of the original object.

        Args:
            tool_name (str): The name of the tool called that the result belongs to.
            objects (list[dict]): The objects to add to the environment.
            metadata (dict): Optional. The metadata of the objects to add to the environment.
                Defaults to an empty dictionary.
            include_duplicates (bool): Optional. Whether to include duplicate objects in the environment.
                Defaults to `False`, which still adds the duplicate object, but with a duplicate `_REF_ID` entry (and no repeating properties).
                If `True`, the duplicate object is added with a new `_REF_ID` entry, and the repeated properties are added to the object.

        """
        if tool_name not in self.environment:
            self.environment[tool_name] = [
                EnvironmentItem(
                    metadata=metadata,
                    objects=objects,
                )
            ]
        else:
            found_item = False
            for item in self.environment[tool_name]:
                if item.metadata == metadata:
                    item.objects.extend(objects)
                    found_item = True
                    break
            if not found_item:
                self.environment[tool_name].append(
                    EnvironmentItem(
                        metadata=metadata,
                        objects=objects,
                    )
                )

    def append(
        self,
        tool_name: str,
        objects: list[dict],
        metadata: dict | None = None,
        metadata_key: str | None = None,
        metadata_value: Any | None = None,
    ):
        """
        Appends an object to a list of objects in the environment.
        Appends under the `"objects"` key of a given item in the environment.
        The item is specified by the metadata of the object. If no metadata is provided, the item is specified by the metadata key and value.

        If no item is found, an error is raised.
        Args:
            tool_name (str): The name of the tool called that the result belongs to.
            metadata (dict | None): The metadata of the objects to append to the environment.
            metadata_key (str | None): The key of the metadata to append to the environment.
            metadata_value (Any | None): The value of the metadata to append to the environment.
            objects (list[dict]): The objects to append to the environment.

        For example, if you have
        ```python
        {
            "query": [
                {
                    "metadata": {"collection_name": "pet_food", "query_search_term": "animals"},
                    "objects": [{"animal": "frog"}]
                }
                ...
            ]
        }
        ```
        and you want to append an object that matches the same `"collection_name"` of `"pet_food"`, you can do:

        ```python
        environment.append(tool_name="query", objects=[{"animal": "reindeer"}], metadata_key="collection_name", metadata_value="pet_food")
        ```
        Then the environment would be updated to:
        ```python
        {
            "query": [
                {
                    "metadata": {"collection_name": "pet_food", "query_search_term": "animals"},
                    "objects": [{"animal": "frog"}, {"animal": "reindeer"}]
                }
                ...
            ]
        }
        ```
        """
        if (
            metadata is not None
            and metadata_key is not None
            and metadata_value is not None
        ):
            raise ValueError(
                "Cannot provide both metadata and metadata_key/metadata_value"
            )
        elif metadata is None and (metadata_key is None or metadata_value is None):
            raise ValueError(
                "Must provide either metadata or metadata_key/metadata_value"
            )
        elif metadata_key is not None and metadata_value is None:
            raise ValueError("Must provide metadata_value if metadata_key is provided")
        elif metadata_key is None and metadata_value is not None:
            raise ValueError("Must provide metadata_key if metadata_value is provided")

        if metadata:
            for item in self.environment[tool_name]:
                if item.metadata == metadata:
                    item.objects.extend(objects)
                    return
            raise ValueError(
                f"No item found in the environment for the given metadata: {metadata}"
            )

        elif metadata_key:
            for item in self.environment[tool_name]:
                if (
                    metadata_key in item.metadata
                    and item.metadata[metadata_key] == metadata_value
                ):
                    item.objects.extend(objects)
                    return

            raise ValueError(
                f"No item found in the environment for the given metadata: {metadata_key} = {metadata_value}"
            )

    def remove(
        self,
        tool_name: str,
        metadata: dict | None = None,
        metadata_key: str | None = None,
        metadata_value: Any | None = None,
    ):
        """
        Replaces the list of objects for the given `tool_name` with an empty list.
        Specifying either `metadata` or the `metadata_key` and `metadata_value` parameters will specify the item to remove.
        If no item is found, all items under the given `tool_name` are removed.
        Args:
            tool_name (str): The name of the tool called that the result belongs to.
            metadata (dict | None): The metadata of the object to remove.
            metadata_key (str | None): The key of the metadata to remove.
            metadata_value (Any | None): The value of the metadata to remove.
        """
        if (
            metadata is not None
            and metadata_key is not None
            and metadata_value is not None
        ):
            raise ValueError(
                "Cannot provide both metadata and metadata_key/metadata_value"
            )
        elif metadata_key is not None and metadata_value is None:
            raise ValueError("Must provide metadata_value if metadata_key is provided")
        elif metadata_key is None and metadata_value is not None:
            raise ValueError("Must provide metadata_key if metadata_value is provided")

        if metadata:
            for item in self.environment[tool_name]:
                if item.metadata == metadata:
                    item.objects = []
                    return
            raise ValueError(
                f"No item found in the environment for the given metadata: {metadata}"
            )
        elif metadata_key:
            for item in self.environment[tool_name]:
                if (
                    metadata_key in item.metadata
                    and item.metadata[metadata_key] == metadata_value
                ):
                    item.objects = []
                    return
            raise ValueError(
                f"No item found in the environment for the given metadata: {metadata_key} = {metadata_value}"
            )
        else:
            self.environment[tool_name] = []

    def replace(
        self,
        tool_name: str,
        objects: list[dict],
        metadata: dict | None = None,
        metadata_key: str | None = None,
        metadata_value: Any | None = None,
    ):
        """
        Replaces the list of objects for the given `tool_name` with the given list of objects.
        Specifying either `metadata` or the `metadata_key` and `metadata_value` parameters will specify the item to replace.
        Args:
            tool_name (str): The name of the tool called that the result belongs to.
            objects (list[dict]): The objects to replace the existing objects with.
            metadata (dict | None): The metadata of the objects to replace the existing objects with.
            metadata_key (str | None): The key of the metadata to replace the existing objects with.
            metadata_value (Any | None): The value of the metadata to replace the existing objects with.
        """

        if (
            metadata is not None
            and metadata_key is not None
            and metadata_value is not None
        ):
            raise ValueError(
                "Cannot provide both metadata and metadata_key/metadata_value"
            )
        elif metadata_key is not None and metadata_value is None:
            raise ValueError("Must provide metadata_value if metadata_key is provided")
        elif metadata_key is None and metadata_value is not None:
            raise ValueError("Must provide metadata_key if metadata_value is provided")
        elif metadata is None and (metadata_key is None or metadata_value is None):
            raise ValueError(
                "Must provide either metadata or metadata_key/metadata_value"
            )
        elif tool_name and tool_name not in self.environment:
            raise ValueError(f"Tool name {tool_name} not found in environment")

        if metadata:
            for item in self.environment[tool_name]:
                if item.metadata == metadata:
                    item.objects = objects
                    return
            raise ValueError(
                f"No item found in the environment for the given metadata: {metadata}"
            )
        elif metadata_key:
            for item in self.environment[tool_name]:
                if (
                    metadata_key in item.metadata
                    and item.metadata[metadata_key] == metadata_value
                ):
                    item.objects = objects
                    return
            raise ValueError(
                f"No item found in the environment for the given metadata: {metadata_key} = {metadata_value}"
            )

    def get_objects(
        self,
        tool_name: str,
        metadata: dict | None = None,
        metadata_key: str | None = None,
        metadata_value: Any | None = None,
    ) -> list[dict] | None:
        """
        Get the objects for the given `tool_name` associated with the given `metadata` or `metadata_key` and `metadata_value`.
        Args:
            tool_name (str): The name of the tool called that the result belongs to.
            metadata (dict | None): The metadata of the object to get.
            metadata_key (str | None): The key of the metadata to get.
            metadata_value (Any | None): The value of the metadata to get.
        Returns:
            list[dict] | None: The list of objects for the given `tool_name`. `None` if the `tool_name` is not found in the environment.
        """
        if (
            metadata is not None
            and metadata_key is not None
            and metadata_value is not None
        ):
            raise ValueError(
                "Cannot provide both metadata and metadata_key/metadata_value"
            )
        elif metadata_key is not None and metadata_value is None:
            raise ValueError("Must provide metadata_value if metadata_key is provided")
        elif metadata_key is None and metadata_value is not None:
            raise ValueError("Must provide metadata_key if metadata_value is provided")
        elif metadata is None and (metadata_key is None or metadata_value is None):
            raise ValueError(
                "Must provide either metadata or metadata_key/metadata_value"
            )

        if tool_name not in self.environment:
            return None

        if metadata:
            for item in self.environment[tool_name]:
                if item.metadata == metadata:
                    return item.objects
            raise ValueError(
                f"No item found in the environment for the given metadata: {metadata}"
            )

        elif metadata_key:
            objects = []
            for item in self.environment[tool_name]:
                if (
                    metadata_key in item.metadata
                    and item.metadata[metadata_key] == metadata_value
                ):
                    objects.extend(item.objects)
            if not objects:
                raise ValueError(
                    f"No item found in the environment for the given metadata: {metadata_key} = {metadata_value}"
                )
            return objects

    def get(self, tool_name: str) -> list[EnvironmentItem] | None:
        """
        Get the objects for the given `tool_name`.

        Args:
            tool_name (str): The name of the tool called that the result belongs to.

        Returns:
            list[EnvironmentItem] | None: The list of items for the given `tool_name`. `None` if the `tool_name` is not found in the environment.
        """

        if tool_name not in self.environment:
            return None

        return self.environment[tool_name]

    def output_llm_metadata(self):
        out = ""
        for tool_name, items in self.environment.items():
            out += f"{tool_name}: {len(items)} results\n\t---\n"
            for item in items:
                out += f"\t{len(item.objects)} object{'s' if len(item.objects) != 1 else ''} with metadata:\n"
                for metadata_key, metadata_value in item.metadata.items():
                    out += f"\t\t{metadata_key}: {metadata_value}\n"
                out += "\t---\n"
        return out

    def to_json(self, remove_unserialisable: bool = False):
        """
        Converts the environment to a JSON serialisable format.
        Used to access specific objects from the environment.
        """
        env_copy = deepcopy(self.environment)
        hidden_env_copy = deepcopy(self.hidden_environment)

        for tool_name in env_copy:
            for item in env_copy[tool_name]:
                format_dict_to_serialisable(item.metadata, remove_unserialisable)
                for obj in item.objects:
                    format_dict_to_serialisable(obj, remove_unserialisable)

        format_dict_to_serialisable(hidden_env_copy)

        return {
            "environment": {
                tool_name: [item.to_json() for item in items]
                for tool_name, items in env_copy.items()
            },
            "hidden_environment": hidden_env_copy,
        }

    @classmethod
    def from_json(cls, json_data: dict):
        environment = {
            tool_name: [EnvironmentItem.from_json(item) for item in items]
            for tool_name, items in json_data.get("environment", {}).items()
        }
        return cls(
            environment=environment,
            hidden_environment=json_data.get("hidden_environment", {}),
        )


def datetime_reference():
    date: datetime = datetime.now()
    return {
        "current_datetime": date.isoformat(),
        "current_day_of_week": date.strftime("%A"),
        "current_time_of_day": date.strftime("%I:%M %p"),
    }


class Atlas(BaseModel):
    style: str = Field(
        default="No style provided.",
        description="The writing style of the agent. "
        "This is the way the agent writes, and the tone of the language it uses.",
    )
    agent_description: str = Field(
        default="No description provided.",
        description="The description of the process you are following. This is pre-defined by the user. "
        "This could be anything - this is the theme of the program you are a part of.",
    )
    end_goal: str = Field(
        default="No end goal provided.",
        description="A short description of your overall goal. "
        "Use this to determine if you have completed your task. "
        "However, you can still choose to end actions early, "
        "if you believe the task is not possible to be completed with what you have available.",
    )
    datetime_reference: dict = Field(
        default=datetime_reference(),
        description="Current context information (e.g., date, time) for decision-making",
    )


class CollectionData:
    """
    Store of data about the Weaviate collections that are used in the tree.
    These are the output of the `preprocess` function.

    You do not normally need to interact with this class directly.
    Instead, do so via the `TreeData` class, which has corresponding methods to get the data in a variety of formats.
    (Such as via the `output_full_metadata` method.)
    """

    def __init__(
        self,
        collection_names: list[str],
        metadata: dict[str, Any] = {},
        logger: Logger | None = None,
    ):
        self.collection_names = collection_names
        self.metadata = metadata
        self.logger = logger

    async def set_collection_names(
        self, collection_names: list[str], client_manager: ClientManager
    ):
        temp_metadata = {}

        # check if any of these collections exist in the full metadata (cached)
        collections_to_get = []
        for collection_name in collection_names:
            if collection_name not in self.metadata:
                collections_to_get.append(collection_name)

        self.removed_collections = []
        self.incorrect_collections = []

        metadata_name = "ELYSIA_METADATA__"

        async with client_manager.connect_to_async_client() as client:
            # check if the metadata collection exists
            if not await client.collections.exists(metadata_name):
                self.removed_collections.extend(collections_to_get)
            else:
                metadata_collection = client.collections.get(metadata_name)
                filters = (
                    Filter.any_of(
                        [
                            Filter.by_property("name").equal(collection_name)
                            for collection_name in collections_to_get
                        ]
                    )
                    if len(collections_to_get) >= 1
                    else None
                )
                metadata = await metadata_collection.query.fetch_objects(
                    filters=filters,
                    limit=9999,
                )
                metadata_map = {
                    metadata_obj.properties["name"]: metadata_obj.properties
                    for metadata_obj in metadata.objects
                }

                for collection_name in collections_to_get:

                    if not await client.collections.exists(collection_name):
                        self.incorrect_collections.append(collection_name)
                        continue

                    if collection_name not in metadata_map:
                        self.removed_collections.append(collection_name)
                    else:
                        properties = metadata_map[collection_name]
                        format_dict_to_serialisable(properties)  # type: ignore
                        temp_metadata[collection_name] = properties

        if len(self.removed_collections) > 0 and self.logger:
            self.logger.warning(
                "The following collections have not been pre-processed for Elysia. "
                f"{self.removed_collections}. "
                "Ignoring these collections for now."
            )

        if len(self.incorrect_collections) > 0 and self.logger:
            self.logger.warning(
                "The following collections cannot be found in this Weaviate cluster. "
                f"{self.incorrect_collections}. "
                "These are being ignored for now. Please check that the collection names are correct."
            )

        self.collection_names = [
            collection_name
            for collection_name in collection_names
            if collection_name not in self.removed_collections
            and collection_name not in self.incorrect_collections
        ]

        # add to cached full metadata
        self.metadata = {
            **self.metadata,
            **{
                collection_name: temp_metadata[collection_name]
                for collection_name in temp_metadata
                if collection_name not in self.metadata
            },
        }

        return self.collection_names

    def output_full_metadata(
        self, collection_names: list[str] | None = None, with_mappings: bool = False
    ):
        if collection_names is None:
            collection_names = self.collection_names

        if with_mappings:
            return {
                collection_name: self.metadata[collection_name]
                for collection_name in collection_names
            }
        else:
            return {
                collection_name: {
                    k: v
                    for k, v in self.metadata[collection_name].items()
                    if k != "mappings"
                }
                for collection_name in collection_names
            }

    def output_collection_summaries(self, collection_names: list[str] | None = None):
        if collection_names is None:
            return {
                collection_name: self.metadata[collection_name]["summary"]
                for collection_name in self.collection_names
            }
        else:
            return {
                collection_name: self.metadata[collection_name]["summary"]
                for collection_name in collection_names
            }

    def output_mapping_lists(self):
        return {
            collection_name: list(self.metadata[collection_name]["mappings"].keys())
            for collection_name in self.collection_names
        }

    def output_mappings(self):
        return {
            collection_name: self.metadata[collection_name]["mappings"]
            for collection_name in self.collection_names
        }

    def to_json(self):
        return {
            "collection_names": self.collection_names,
            "metadata": self.metadata,
        }

    @classmethod
    def from_json(cls, json_data: dict, logger: Logger | None = None):
        return cls(
            collection_names=json_data["collection_names"],
            metadata=json_data["metadata"],
            logger=logger,
        )


class TreeData:
    """
    Store of data across the tree.
    This includes things like conversation history, actions, decisions, etc.
    These data are given to ALL agents, so every agent is aware of the stage of the decision processes.

    This also contains functions that process the data into an LLM friendly format,
    such as a string with extra description.
    E.g. the number of trees completed is converted into a string (i/N)
         and additional warnings if i is close to N.

    The TreeData has the following objects:

    - collection_data (CollectionData): The collection metadata/schema, which contains information about the collections used in the tree.
        This is the store of data that is saved by the `preprocess` function, and retrieved on initialisation of this object.
    - atlas (Atlas): The atlas, described in the Atlas class.
    - user_prompt (str): The user's prompt.
    - conversation_history (list[dict]): The conversation history stored in the current tree, of the form:
        ```python
        [
            {
                "role": "user" | "assistant",
                "content": str,
            },
            ...
        ]
        ```
    - environment (Environment): The environment, described in the Environment class.
    - tasks_completed (list[dict]): The tasks completed as a list of dictionaries.
        This is separate from the environment, as it separates what tasks were completed in each prompt in which order.
    - num_trees_completed (int): The current level of the decision tree, how many iterations have been completed so far.
    - recursion_limit (int): The maximum number of iterations allowed in the decision tree.
    - errors (dict): A dictionary of self-healing errors that have occurred in the tree. Keyed by the function name that caused the error.

    In general, you should not initialise this class directly.
    But you can access the data in this class to access the relevant data from the tree (in e.g. tool construction/usage).
    """

    def __init__(
        self,
        collection_data: CollectionData,
        atlas: Atlas,
        user_prompt: str | None = None,
        conversation_history: list[dict] | None = None,
        environment: Environment | None = None,
        tasks_completed: list[dict] | None = None,
        num_trees_completed: int | None = None,
        recursion_limit: int | None = None,
        settings: Settings | None = None,
    ):
        if settings is None:
            self.settings = environment_settings
        else:
            self.settings = settings

        # -- Base Data --
        if user_prompt is None:
            self.user_prompt = ""
        else:
            self.user_prompt = user_prompt

        if conversation_history is None:
            self.conversation_history = []
        else:
            self.conversation_history = conversation_history

        if environment is None:
            self.environment = Environment()
        else:
            self.environment = environment

        if tasks_completed is None:
            self.tasks_completed = []
        else:
            self.tasks_completed = tasks_completed

        if num_trees_completed is None:
            self.num_trees_completed = 0
        else:
            self.num_trees_completed = num_trees_completed

        if recursion_limit is None:
            self.recursion_limit = 3
        else:
            self.recursion_limit = recursion_limit

        # -- Atlas --
        self.atlas = atlas

        # -- Collection Data --
        self.collection_data = collection_data
        self.collection_names = []

        # -- Errors --
        self.errors: dict[str, list[str]] = {}
        self.current_task = None

    def set_property(self, property: str, value: Any):
        self.__dict__[property] = value

    def update_string(self, property: str, value: str):
        if property not in self.__dict__:
            self.__dict__[property] = ""
        self.__dict__[property] += value

    def update_list(self, property: str, value: Any):
        if property not in self.__dict__:
            self.__dict__[property] = []
        self.__dict__[property].append(value)

    def update_dict(self, property: str, key: str, value: Any):
        if property not in self.__dict__:
            self.__dict__[property] = {}
        self.__dict__[property][key] = value

    def delete_from_dict(self, property: str, key: str):
        if property in self.__dict__ and key in self.__dict__[property]:
            del self.__dict__[property][key]

    def soft_reset(self):
        self.previous_reasoning = {}

    def _update_task(self, task_dict, key, value):
        if value is not None:
            if key in task_dict:
                # If key already exists, append to it
                if isinstance(value, str):
                    task_dict[key] += "\n" + value
                elif isinstance(value, int) or isinstance(value, float):
                    task_dict[key] += value
                elif isinstance(value, list):
                    task_dict[key].extend(value)
                elif isinstance(value, dict):
                    task_dict[key].update(value)
                elif isinstance(value, bool):
                    task_dict[key] = value
            elif key not in task_dict:
                # If key does not exist, create it
                task_dict[key] = value

    def update_tasks_completed(
        self, prompt: str, task: str, num_trees_completed: int, **kwargs
    ):
        # search to see if the current prompt already has an entry for this task
        prompt_found = False
        task_found = False
        iteration_found = False

        for i, task_prompt in enumerate(self.tasks_completed):
            if task_prompt["prompt"] == prompt:
                prompt_found = True
                for j, task_j in enumerate(task_prompt["task"]):
                    if task_j["task"] == task:
                        task_found = True
                        task_i = j  # position of the task in the prompt

                    if task_j["iteration"] == num_trees_completed:
                        iteration_found = True

        # If the prompt is not found, add it to the list
        if not prompt_found:
            self.tasks_completed.append({"prompt": prompt, "task": [{}]})
            self.tasks_completed[-1]["task"][0]["task"] = task
            self.tasks_completed[-1]["task"][0]["iteration"] = num_trees_completed
            for kwarg in kwargs:
                self._update_task(
                    self.tasks_completed[-1]["task"][0], kwarg, kwargs[kwarg]
                )
            return

        # If the prompt is found but the task is not, add it to the list
        if prompt_found and not task_found:
            self.tasks_completed[-1]["task"].append({})
            self.tasks_completed[-1]["task"][-1]["task"] = task
            self.tasks_completed[-1]["task"][-1]["iteration"] = num_trees_completed
            for kwarg in kwargs:
                self._update_task(
                    self.tasks_completed[-1]["task"][-1], kwarg, kwargs[kwarg]
                )
            return

        # task already exists in this query, but the iteration is new
        if prompt_found and task_found and not iteration_found:
            self.tasks_completed[-1]["task"].append({})
            self.tasks_completed[-1]["task"][-1]["task"] = task
            self.tasks_completed[-1]["task"][-1]["iteration"] = num_trees_completed
            for kwarg in kwargs:
                self._update_task(
                    self.tasks_completed[-1]["task"][-1], kwarg, kwargs[kwarg]
                )
            return

        # If the prompt is found and the task is found, update the task
        if prompt_found and task_found:
            for kwarg in kwargs:
                self._update_task(
                    self.tasks_completed[-1]["task"][task_i], kwarg, kwargs[kwarg]
                )

    def set_current_task(self, task: str):
        self.current_task = task

    def get_errors(self):
        if self.current_task == "elysia_decision_node":
            return self.errors
        elif self.current_task is None or self.current_task not in self.errors:
            return []
        else:
            return self.errors[self.current_task]

    def clear_error(self, task: str):
        if task in self.errors:
            self.errors[task] = []

    def tasks_completed_string(self):
        """
        Output a nicely formatted string of the tasks completed so far, designed to be used in the LLM prompt.
        This is where the outputs of the `llm_message` fields are displayed.
        You can use this if you are interfacing with LLMs in tools, to help it understand the context of the tasks completed so far.

        Returns:
            (str): A separated and formatted string of the tasks completed so far in an LLM-parseable format.
        """
        out = ""
        for j, task_prompt in enumerate(self.tasks_completed):
            out += f"<prompt_{j+1}>\n"
            out += f"Prompt: {task_prompt['prompt']}\n"

            for i, task in enumerate(task_prompt["task"]):
                out += f"<task_{i+1}>\n"

                if "action" in task and task["action"]:
                    out += (
                        f"Chosen action: {task['task']} (this does not mean it has been completed, "
                        "only that it was chosen, use the environment to judge if a task is completed)\n"
                    )
                else:
                    out += f"Chosen subcategory: {task['task']} (this action has not been completed, this is only a subcategory)"

                if "error" in task and task["error"]:
                    out += (
                        f" (UNSUCCESSFUL) There was an error during this tool call. "
                        "See the error messages for details. This action did not complete.\n"
                    )
                else:
                    out += f" (SUCCESSFUL)\n"
                    for key in task:
                        if key != "task" and key != "action":
                            out += f"{key.capitalize()}: {task[key]}\n"

                out += f"</task_{i+1}>\n"
            out += f"</prompt_{j+1}>\n"

        return out

    async def set_collection_names(
        self, collection_names: list[str], client_manager: ClientManager
    ):
        self.collection_names = await self.collection_data.set_collection_names(
            collection_names, client_manager
        )
        return self.collection_names

    def tree_count_string(self):
        out = f"{self.num_trees_completed+1}/{self.recursion_limit}"
        if self.num_trees_completed == self.recursion_limit - 1:
            out += " (this is the last decision you can make before being cut off)"
        if self.num_trees_completed >= self.recursion_limit:
            out += " (recursion limit reached, write your full chat response accordingly - the decision process has been cut short, and it is likely the user's question has not been fully answered and you either haven't been able to do it or it was impossible)"
        return out

    def output_collection_metadata(
        self, collection_names: list[str] | None = None, with_mappings: bool = False
    ):
        """
        Outputs the full metadata for the given collection names.

        Args:
            with_mappings (bool): Whether to output the mappings for the collections as well as the other metadata.

        Returns:
            dict (dict[str, dict]): A dictionary of collection names to their metadata.
                The metadata are of the form:
                ```python
                {
                    # summary statistics of each field in the collection
                    "fields": list = [
                        field_name_1: dict = {
                            "description": str,
                            "range": list[float],
                            "type": str,
                            "groups": dict[str, str],
                            "mean": float
                        },
                        field_name_2: dict = {
                            ... # same fields as above
                        },
                        ...
                    ],

                    # mapping_1, mapping_2 etc refer to frontend-specific types that the AI has deemed appropriate for this data
                    # then the dict is to map the frontend fields to the data fields
                    "mappings": dict = {
                        mapping_1: dict = {
                            "frontend_field_1": "data_field_1",
                            "frontend_field_2": "data_field_2",
                            ...
                        },
                        mapping_2: dict = {
                            ... # same fields as above
                        },
                        ...,
                    },

                    # number of items in collection (float but just for consistency)
                    "length": float,

                    # AI generated summary of the dataset
                    "summary": str,

                    # name of collection
                    "name": str,

                    # what named vectors are available and their properties (if any)
                    "named_vectors": list = [
                        {
                            "name": str,
                            "enabled": bool,
                            "source_properties": list,
                            "description": str # defaults to empty
                        },
                        ...
                    ],

                    # some config settings relevant for queries
                    "index_properties": {
                        "isNullIndexed": bool,
                        "isLengthIndexed": bool,
                        "isTimestampIndexed": bool,
                    },
                }
                ```
                If `with_mappings` is `False`, then the mappings are not included.
                Each key in the outer level dictionary is a collection name.

        """

        if collection_names is None:
            collection_names = self.collection_names

        return self.collection_data.output_full_metadata(
            collection_names, with_mappings
        )

    def output_collection_return_types(self) -> dict[str, list[str]]:
        """
        Outputs the return types for the collections in the tree data.
        Essentially, this is a list of the keys that can be used to map the objects to the frontend.

        Returns:
            (dict): A dictionary of collection names to their return types.
                ```python
                {
                    collection_name_1: list[str],
                    collection_name_2: list[str],
                    ...,
                }
                ```
                Each of these lists is a list of the keys that can be used to map the objects to the frontend.
        """
        collection_return_types = self.collection_data.output_mapping_lists()
        out = {
            collection_name: collection_return_types[collection_name]
            for collection_name in self.collection_names
        }
        return out

    def to_json(self, remove_unserialisable: bool = False):
        out = {
            k: v
            for k, v in self.__dict__.items()
            if k not in ["collection_data", "atlas", "environment", "settings"]
        }
        out["collection_data"] = self.collection_data.to_json()
        out["atlas"] = self.atlas.model_dump()
        out["environment"] = self.environment.to_json(remove_unserialisable)
        out["settings"] = self.settings.to_json()
        return out

    @classmethod
    def from_json(cls, json_data: dict):
        settings = Settings.from_json(json_data["settings"])
        logger = settings.logger
        collection_data = CollectionData.from_json(json_data["collection_data"], logger)
        atlas = Atlas.model_validate(json_data["atlas"])
        environment = Environment.from_json(json_data["environment"])

        tree_data = cls(
            collection_data=collection_data,
            atlas=atlas,
            environment=environment,
            settings=settings,
        )
        for item in json_data:
            if item not in ["collection_data", "atlas", "environment", "settings"]:
                tree_data.set_property(item, json_data[item])
        return tree_data
