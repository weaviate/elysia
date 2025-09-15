import asyncio
import datetime
import os
from typing import Any
import uuid
from dotenv import load_dotenv
from weaviate.util import generate_uuid5

# Load environment variables from .env file
load_dotenv(override=True)

from elysia.tree.tree import Tree
from elysia.util.client import ClientManager
from elysia.tree.util import delete_tree_from_weaviate
from elysia.api.utils.config import Config, BranchInitType
from elysia.config import Settings


class TreeManager:
    """
    Manages trees (different conversations) for a single user.
    Designed to be used with the Elysia API, or a manager for multiple Elysia decision trees.

    You can initialise the TreeManager with particular config options.
    Or, upon adding a tree, you can set a specific style, description and end goal for that tree.

    Each tree has separate elements: the tree itself, the last request time, and the asyncio event.
    """

    def __init__(
        self,
        user_id: str,
        config: Config | None = None,
        tree_timeout: datetime.timedelta | int | None = None,
    ):
        """
        Args:
            user_id (str): Required. A unique identifier for the user being managed within this TreeManager.
            config (Config | None): Optional. A config for all trees managed by this TreeManager.
                Defaults to a new config with default settings.
                If `settings` is not provided, it will be set to the default settings (smart_setup).
            tree_timeout (datetime.timedelta | int | None): Optional. A timeout for all trees managed by this TreeManager.
                Defaults to the value of the `TREE_TIMEOUT` environment variable.
                If an integer is passed, it will be interpreted as minutes.
                If set to 0, trees will not be automatically removed.
        """
        self.trees = {}
        self.user_id = user_id

        if tree_timeout is None:
            self.tree_timeout = datetime.timedelta(
                minutes=int(os.environ.get("TREE_TIMEOUT", 10))
            )
        elif isinstance(tree_timeout, int):
            self.tree_timeout = datetime.timedelta(minutes=tree_timeout)
        else:
            self.tree_timeout = tree_timeout

        if config is None:
            self.config = Config()
        else:
            self.config = config

        self.settings = self.config.settings

    def update_config(
        self,
        conversation_id: str | None = None,
        config_id: str | None = None,
        config_name: str | None = None,
        settings: dict[str, Any] | None = None,
        style: str | None = None,
        agent_description: str | None = None,
        end_goal: str | None = None,
        branch_initialisation: BranchInitType | None = None,
    ):
        if config_id is not None:
            self.config.id = config_id

        if config_name is not None:
            self.config.name = config_name

        if settings is not None:
            self.configure(conversation_id=conversation_id, replace=True, **settings)

        if style is not None:
            self.change_style(style, conversation_id)

        if agent_description is not None:
            self.change_agent_description(agent_description, conversation_id)

        if end_goal is not None:
            self.change_end_goal(end_goal, conversation_id)

        if branch_initialisation is not None:
            self.change_branch_initialisation(branch_initialisation, conversation_id)

    def add_tree(
        self,
        conversation_id: str,
        low_memory: bool = False,
    ):
        """
        Add a tree to the TreeManager.
        The decision tree can be initialised with specific config options, such as style, agent description and end goal.
        As well as a Settings object, which chooses options such as the LLM models, API keys and more.

        Args:
            conversation_id (str): Required. A unique identifier for the conversation.
            low_memory (bool): Optional. Whether to use low memory mode for the tree.
                Controls the LM history being saved in the tree, and some other variables.
                Defaults to False.
        """

        if not self.tree_exists(conversation_id):
            self.trees[conversation_id] = {
                "tree": Tree(
                    conversation_id=conversation_id,
                    user_id=self.user_id,
                    settings=self.settings,
                    style=self.config.style,
                    agent_description=self.config.agent_description,
                    end_goal=self.config.end_goal,
                    branch_initialisation=self.config.branch_initialisation,
                    low_memory=low_memory,
                    use_elysia_collections=self.config.use_elysia_collections,
                ),
                "last_request": datetime.datetime.now(),
                "event": asyncio.Event(),
            }
            self.trees[conversation_id]["event"].set()

    async def save_tree_weaviate(
        self, conversation_id: str, client_manager: ClientManager
    ):
        """
        Save a tree to Weaviate to collection ELYSIA_TREES__.
        Creates the collection if it doesn't exist.

        Args:
            conversation_id (str): The conversation ID which contains the tree.
            client_manager (ClientManager): The client manager to use for the tree.
        """
        tree: Tree = self.get_tree(conversation_id)
        await tree.export_to_weaviate("ELYSIA_TREES__", client_manager)

    async def check_tree_exists_weaviate(
        self, conversation_id: str, client_manager: ClientManager
    ):
        """
        Check if a tree exists in a Weaviate instance.
        The collection ELYSIA_TREES__ must exist, returns False if it doesn't.

        Args:
            conversation_id (str): The conversation ID which contains the tree.
            client_manager (ClientManager): The client manager to use for the tree.

        Returns:
            (bool): True if the tree exists in the Weaviate instance, False otherwise.
        """
        async with client_manager.connect_to_async_client() as client:
            if not await client.collections.exists("ELYSIA_TREES__"):
                return False

            collection = client.collections.get("ELYSIA_TREES__")
            uuid = generate_uuid5(conversation_id)
            return await collection.data.exists(uuid)

    async def load_tree_weaviate(
        self, conversation_id: str, client_manager: ClientManager
    ):
        """
        Load a tree from Weaviate.
        The conversation ID from the loaded tree is placed into the tree manager
        (possibly overwriting an existing tree with the same conversation ID).
        Then the tree itself is not returned - instead the list of frontend payloads
        that were yielded to the frontend by the tree is returned.

        Args:
            conversation_id (str): The conversation ID which contains the tree.
            client_manager (ClientManager): The client manager to use for the tree.

        Returns:
            (list): A list of dictionaries, each containing a frontend payload that was used to generate the tree.
                The list is ordered by the time the payload was originally sent to the frontend (at the time it was saved).
        """
        tree = await Tree.import_from_weaviate(
            "ELYSIA_TREES__", conversation_id, client_manager
        )
        if conversation_id not in self.trees:
            self.trees[conversation_id] = {
                "tree": None,
                "event": asyncio.Event(),
                "last_request": datetime.datetime.now(),
            }
        self.trees[conversation_id]["tree"] = tree
        self.trees[conversation_id]["event"].set()
        self.update_tree_last_request(conversation_id)
        return tree.returner.store

    async def delete_tree_weaviate(
        self, conversation_id: str, client_manager: ClientManager
    ):
        """
        Delete a tree from the stored trees in Weaviate.

        Args:
            conversation_id (str): The conversation ID of the tree to be deleted.
            client_manager (ClientManager): The client manager pointing to the Weaviate instance containing the tree.
        """
        await delete_tree_from_weaviate(
            conversation_id, "ELYSIA_TREES__", client_manager
        )

    def delete_tree_local(self, conversation_id: str):
        """
        Delete a tree from the TreeManager.

        Args:
            conversation_id (str): The conversation ID of the tree to be deleted.
        """
        if conversation_id in self.trees:
            del self.trees[conversation_id]

    def tree_exists(self, conversation_id: str):
        """
        Check if a tree exists in the TreeManager.

        Args:
            conversation_id (str): The conversation ID which may contain the tree.

        Returns:
            (bool): True if the tree exists, False otherwise.
        """
        return conversation_id in self.trees

    def get_tree(self, conversation_id: str):
        """
        Get a tree from the TreeManager.
        Will raise a ValueError if the tree is not found.

        Args:
            conversation_id (str): The conversation ID which contains the tree.

        Returns:
            (Tree): The tree associated with the conversation ID.
        """
        if conversation_id not in self.trees:
            raise ValueError(
                f"Tree {conversation_id} not found. Please initialise a tree first (by calling `add_tree`)."
            )

        return self.trees[conversation_id]["tree"]

    def get_event(self, conversation_id: str):
        """
        Get the asyncio.Event for a tree in the TreeManager.
        This is cleared when the tree is processing, and set when the tree is idle.
        This is used to block the API from sending multiple requests to the tree at once.

        Args:
            conversation_id (str): The conversation ID which contains the tree.

        Returns:
            (asyncio.Event): The event for the tree.
        """
        return self.trees[conversation_id]["event"]

    def configure(
        self, conversation_id: str | None = None, replace: bool = False, **kwargs: Any
    ):
        """
        Configure the settings for a tree in the TreeManager.

        Args:
            conversation_id (str | None): The conversation ID which contains the tree.
            replace (bool): Whether to override the current settings with the new settings.
                When this is True, all existing settings are removed, and only the new settings are used.
                Defaults to False.
            **kwargs (Any): The keyword arguments to pass to the Settings.configure() method.
        """
        if conversation_id is None:
            self.settings.configure(replace=replace, **kwargs)
            self.config.settings = self.settings
        else:
            self.get_tree(conversation_id).settings.configure(replace=replace, **kwargs)

    def change_style(self, style: str, conversation_id: str | None = None):
        """
        Change the style for a tree in the TreeManager.
        Or change the global style for all trees (if conversation_id is not supplied).
        And applies these changes to current trees with default settings.

        Args:
            style (str): The new style for the tree.
            conversation_id (str | None): Optional. The conversation ID which contains the tree.
                If not supplied, the style will be changed on all trees.
        """
        if conversation_id is None:
            self.config.style = style
            for conversation_id in self.trees:
                if not self.trees[conversation_id]["tree"]._config_modified:
                    self.trees[conversation_id]["tree"].change_style(style)
                    self.trees[conversation_id]["tree"]._config_modified = False
        else:
            if conversation_id not in self.trees:
                raise ValueError(f"Tree {conversation_id} not found")

            self.trees[conversation_id]["tree"].change_style(style)

    def change_agent_description(
        self, agent_description: str, conversation_id: str | None = None
    ):
        """
        Change the agent description for a tree in the TreeManager.
        Or change the global agent description for all trees (if conversation_id is not supplied).
        And applies these changes to current trees with default settings.

        Args:
            agent_description (str): The new agent description for the tree.
            conversation_id (str | None): Optional. The conversation ID which contains the tree.
                If not supplied, the agent description will be changed on all trees.
        """
        if conversation_id is None:
            self.config.agent_description = agent_description
            for conversation_id in self.trees:
                if not self.trees[conversation_id]["tree"]._config_modified:
                    self.trees[conversation_id]["tree"].change_agent_description(
                        agent_description
                    )
                    self.trees[conversation_id]["tree"]._config_modified = False
        else:
            if conversation_id not in self.trees:
                raise ValueError(f"Tree {conversation_id} not found")

            self.trees[conversation_id]["tree"].change_agent_description(
                agent_description
            )

    def change_end_goal(self, end_goal: str, conversation_id: str | None = None):
        """
        Change the end goal for a tree in the TreeManager.
        Or change the global end goal for all trees (if conversation_id is not supplied).
        And applies these changes to current trees with default settings.

        Args:
            end_goal (str): The new end goal for the tree.
            conversation_id (str | None): Optional. The conversation ID which contains the tree.
                If not supplied, the end goal will be changed on all trees.
        """
        if conversation_id is None:
            self.config.end_goal = end_goal
            for conversation_id in self.trees:
                if not self.trees[conversation_id]["tree"]._config_modified:
                    self.trees[conversation_id]["tree"].change_end_goal(end_goal)
                    self.trees[conversation_id]["tree"]._config_modified = False
        else:
            if conversation_id not in self.trees:
                raise ValueError(f"Tree {conversation_id} not found")

            self.trees[conversation_id]["tree"].change_end_goal(end_goal)

    def change_branch_initialisation(
        self, branch_initialisation: BranchInitType, conversation_id: str | None = None
    ):
        """
        Change the branch initialisation for a tree in the TreeManager.
        Or change the global branch initialisation for all trees (if conversation_id is not supplied).
        And applies these changes to current trees with default settings.

        Args:
            branch_initialisation (str): The new branch initialisation for the tree.
            conversation_id (str | None): Optional. The conversation ID which contains the tree.
                If not supplied, the branch initialisation will be changed on all trees.
        """
        if conversation_id is None:
            self.config.branch_initialisation = branch_initialisation
            for conversation_id in self.trees:
                if not self.trees[conversation_id]["tree"]._config_modified:
                    self.trees[conversation_id]["tree"].set_branch_initialisation(
                        branch_initialisation
                    )
                    self.trees[conversation_id]["tree"]._config_modified = False
        else:
            if conversation_id not in self.trees:
                raise ValueError(f"Tree {conversation_id} not found")

            self.trees[conversation_id]["tree"].set_branch_initialisation(
                branch_initialisation
            )

    async def process_tree(
        self,
        query: str,
        conversation_id: str,
        query_id: str | None = None,
        training_route: str = "",
        collection_names: list[str] = [],
        client_manager: ClientManager | None = None,
    ):
        """
        Process a tree in the TreeManager.
        This is an async generator which yields results from the tree.async_run() method.

        Args:
            query (str): Required. The user input/prompt to process in the decision tree.
            conversation_id (str): Required. The conversation ID which contains the tree.
            query_id (str): Optional. A unique identifier for the query.
                If not supplied, a random UUID will be generated.
            training_route (str): Optional. The training route, a string of the form "tool1/tool2/tool1" etc.
                See the `tree.async_run()` method for more details.
            collection_names (list[str]): Optional. A list of collection names to use in the query.
                If not supplied, all collections will be used.
            client_manager (ClientManager | None): Optional. A client manager to use for the query.
                If not supplied, a new ClientManager will be created.
        """

        if query_id is None:
            query_id = str(uuid.uuid4())

        tree: Tree = self.get_tree(conversation_id)
        self.update_tree_last_request(conversation_id)

        # wait for the tree to be idle
        await self.trees[conversation_id]["event"].wait()

        # clear the event, set it to working
        self.trees[conversation_id]["event"].clear()

        try:
            async for yielded_result in tree.async_run(
                query,
                collection_names=collection_names,
                client_manager=client_manager,
                query_id=query_id,
                training_route=training_route,
                close_clients_after_completion=False,
            ):
                yield yielded_result
                self.update_tree_last_request(conversation_id)

        finally:
            # set the event to idle
            self.trees[conversation_id]["event"].set()

    def check_tree_timeout(self, conversation_id: str):
        """
        Check if a tree has been idle for the last tree_timeout.

        Args:
            conversation_id (str): The conversation ID which contains the tree.

        Returns:
            (bool): True if the tree has been idle for the last tree_timeout, False otherwise.
        """
        # if tree not found, return True
        if conversation_id not in self.trees:
            return True

        # Remove any trees that have not been active in the last TREE_TIMEOUT minutes
        if (
            "last_request" in self.trees[conversation_id]
            and datetime.datetime.now() - self.trees[conversation_id]["last_request"]
            > self.tree_timeout
        ):
            return True

        return False

    def update_tree_last_request(self, conversation_id: str):
        self.trees[conversation_id]["last_request"] = datetime.datetime.now()

    def check_all_trees_timeout(self):
        """
        Check all trees in the TreeManager and remove any that have not been active in the last tree_timeout.
        """
        if self.tree_timeout == datetime.timedelta(minutes=0):
            return

        convs_to_remove = []
        for i, conversation_id in enumerate(self.trees):
            if self.check_tree_timeout(conversation_id):
                convs_to_remove.append(conversation_id)

        for conversation_id in convs_to_remove:
            del self.trees[conversation_id]
