import datetime
import os
import random
import json

from dotenv import load_dotenv
from typing import Any
from pathlib import Path
from logging import Logger

load_dotenv(override=True)

from elysia.api.services.tree import TreeManager
from elysia.objects import Update
from elysia.util.client import ClientManager
from elysia.api.core.log import logger
from elysia.api.utils.config import Config
from elysia.api.utils.config import FrontendConfig
from elysia.tree.util import get_saved_trees_weaviate


class TreeTimeoutError(Update):
    def __init__(self):
        super().__init__(
            "tree_timeout_error",
            {
                "text": "This conversation has been timed out due to inactivity. Please start a new conversation."
            },
        )


class UserTimeoutError(Update):
    def __init__(self):
        super().__init__(
            "user_timeout_error",
            {
                "text": "You have been timed out due to inactivity. Please start a new conversation."
            },
        )


async def load_frontend_config_from_file(
    user_id: str, logger: Logger
) -> FrontendConfig:
    # save frontend config locally
    try:
        elysia_package_dir = Path(__file__).parent.parent.parent  # Gets to elysia/
        config_dir = elysia_package_dir / "api" / "user_configs"
        config_file = config_dir / f"frontend_config_{user_id}.json"

        if not config_file.exists():
            return FrontendConfig(logger=logger)

        with open(config_file, "r") as f:
            fe_config = await FrontendConfig.from_json(json.load(f), logger)
            return fe_config

    except Exception as e:
        logger.error(f"Error in save_frontend_config_to_file: {str(e)}")
        return FrontendConfig(logger=logger)


class UserManager:
    """
    The UserManager is designed to manage, for each user:

    - TreeManager
    - ClientManager
    - FrontendConfig

    It contains methods for creating and updating these objects across a range of users, stored in a dictionary.
    It can be used as a dependency injection container for FastAPI.
    The user manager/tree manager is decoupled from the core of the Elysia package functionality.
    It is designed to be used to manage separate Elysia instances, set of trees (via tree managers), configs, etc.
    """

    def __init__(
        self,
        user_timeout: datetime.timedelta | int | None = None,
    ):
        """
        Args:
            user_timeout (datetime.timedelta | int | None): Optional.
                The length of time a user can be idle before being timed out.
                Defaults to 20 minutes or the value of the USER_TIMEOUT environment variable.
                If an integer is provided, it is interpreted as the number of minutes.
        """
        if user_timeout is None:
            self.user_timeout = datetime.timedelta(
                minutes=int(os.environ.get("USER_TIMEOUT", 20))
            )
        elif isinstance(user_timeout, int):
            self.user_timeout = datetime.timedelta(minutes=user_timeout)
        else:
            self.user_timeout = user_timeout

        self.manager_id = random.randint(0, 1000000)
        self.date_of_reset = None
        self.users = {}

    def user_exists(self, user_id: str):
        return user_id in self.users

    async def update_config(
        self,
        user_id: str,
        conversation_id: str | None = None,
        config_id: str | None = None,
        config_name: str | None = None,
        settings: dict[str, Any] | None = None,
        style: str | None = None,
        agent_description: str | None = None,
        end_goal: str | None = None,
        branch_initialisation: str | None = None,
    ):
        local_user = await self.get_user_local(user_id)
        local_user["tree_manager"].update_config(
            conversation_id,
            config_id,
            config_name,
            settings,
            style,
            agent_description,
            end_goal,
            branch_initialisation,
        )

        await local_user["client_manager"].reset_keys(
            wcd_url=local_user["tree_manager"].settings.WCD_URL,
            wcd_api_key=local_user["tree_manager"].settings.WCD_API_KEY,
            api_keys=local_user["tree_manager"].settings.API_KEYS,
        )

    async def update_frontend_config(
        self,
        user_id: str,
        config: dict[str, Any],
    ):
        local_user = await self.get_user_local(user_id)
        frontend_config: FrontendConfig = local_user["frontend_config"]
        await frontend_config.configure(**config)

    async def add_user_local(
        self,
        user_id: str,
        config: Config | None = None,
    ):
        """
        Add a user to the UserManager.

        Args:
            user_id (str): Required. The unique identifier for the user.
            config (Config): Required. The config for the user.
        """

        # add user if it doesn't exist
        if user_id not in self.users:
            self.users[user_id] = {}

            fe_config = await load_frontend_config_from_file(user_id, logger)

            self.users[user_id]["frontend_config"] = fe_config

            self.users[user_id]["tree_manager"] = TreeManager(
                user_id=user_id,
                config=config,
                tree_timeout=fe_config.config["tree_timeout"],
            )

            # client manager starts with env variables, when config is updated, api keys are updated
            self.users[user_id]["client_manager"] = ClientManager(
                logger=logger,
                client_timeout=fe_config.config["client_timeout"],
                settings=self.users[user_id]["tree_manager"].config.settings,
            )

    async def get_user_local(self, user_id: str):
        """
        Return a local user object.
        Will raise a ValueError if the user is not found.

        Args:
            user_id (str): Required. The unique identifier for the user.

        Returns:
            (dict): A local user object, containing a TreeManager ("tree_manager"),
                Frontend Config ("frontend_config") and ClientManager ("client_manager").
        """

        if user_id not in self.users:
            raise ValueError(
                f"User {user_id} not found. Please initialise a user first (by calling `add_user_local`)."
            )

        # update last request (adds last_request to user)
        await self.update_user_last_request(user_id)

        return self.users[user_id]

    async def get_tree(self, user_id: str, conversation_id: str):
        """
        Get a tree for a user.
        Will raise a ValueError if the user is not found, or the tree is not found.

        Args:
            user_id (str): Required. The unique identifier for the user.
            conversation_id (str): Required. The unique identifier for the conversation.

        Returns:
            (Tree): The tree.
        """
        local_user = await self.get_user_local(user_id)
        return local_user["tree_manager"].get_tree(conversation_id)

    async def check_all_trees_timeout(self):
        """
        Check all trees in all TreeManagers across all users and remove any that have not been active in the last tree_timeout.
        """
        for user_id in self.users:
            self.users[user_id]["tree_manager"].check_all_trees_timeout()

    def check_user_timeout(self, user_id: str):
        """
        Check if a user has been idle for the last user_timeout.

        Args:
            user_id (str): The user ID which contains the user.

        Returns:
            (bool): True if the user has been idle for the last user_timeout, False otherwise.
        """
        # if user not found, return True
        if user_id not in self.users:
            return True

        # Remove any trees that have not been active in the last user_timeout
        # if (
        #     "last_request" in self.users[user_id]
        #     and datetime.datetime.now() - self.users[user_id]["last_request"]
        #     > self.user_timeout
        # ):
        #     return True

        return False

    async def check_all_users_timeout(self):
        """
        Check all users in the UserManager and remove any that have not been active in the last user_timeout.
        """
        if self.user_timeout == datetime.timedelta(minutes=0):
            return

        for user_id in self.users:
            if self.check_user_timeout(user_id):
                del self.users[user_id]

    async def check_restart_clients(self):
        """
        Check all clients in all ClientManagers across all users and run the restart_client() method (for sync and async clients).
        The restart_client() methods will check if the client has been inactive for the last client_timeout minutes (set in init).
        """
        for user_id in self.users:
            if (
                "client_manager" in self.users[user_id]
                and self.users[user_id]["client_manager"].is_client
            ):
                await self.users[user_id]["client_manager"].restart_client()
                await self.users[user_id]["client_manager"].restart_async_client()

    async def close_all_clients(self):
        """
        Close all clients in all ClientManagers across all users.
        """
        for user_id in self.users:
            if "client_manager" in self.users[user_id]:
                await self.users[user_id]["client_manager"].close_clients()

    async def initialise_tree(
        self,
        user_id: str,
        conversation_id: str,
        low_memory: bool = False,
    ):
        """
        Initialises a tree for a user for an existing user at user_id.
        Requires a user to already exist in the UserManager, via `add_user_local`.
        This is a wrapper for the TreeManager.add_tree() method.

        Args:
            user_id (str): Required. The unique identifier for the user.
            conversation_id (str): Required. The unique identifier for a new conversation within the tree manager.
            low_memory (bool): Optional. Whether to use low memory mode for the tree.
                Controls the LM history being saved in the tree, and some other variables.
                Defaults to False.
        """
        # self.add_user_local(user_id)
        local_user = await self.get_user_local(user_id)
        tree_manager: TreeManager = local_user["tree_manager"]
        if not tree_manager.tree_exists(conversation_id):
            tree_manager.add_tree(
                conversation_id,
                low_memory,
            )
        return tree_manager.get_tree(conversation_id)

    async def save_tree(
        self,
        user_id: str,
        conversation_id: str,
        wcd_url: str | None = None,
        wcd_api_key: str | None = None,
    ):
        """
        Save a tree to a Weaviate instance (set in the frontend config).
        This is a wrapper for the TreeManager.save_tree_weaviate() method.

        Args:
            user_id (str): Required. The unique identifier for the user stored in the UserManager.
            conversation_id (str): Required. The unique identifier for the conversation for the user.
        """

        local_user = await self.get_user_local(user_id)
        tree_manager: TreeManager = local_user["tree_manager"]

        if wcd_url is None or wcd_api_key is None:
            save_location_client_manager = local_user[
                "frontend_config"
            ].save_location_client_manager
        else:
            save_location_client_manager = ClientManager(
                logger=logger,
                wcd_url=wcd_url,
                wcd_api_key=wcd_api_key,
            )

        await tree_manager.save_tree_weaviate(
            conversation_id, save_location_client_manager
        )

    async def check_tree_exists_weaviate(
        self,
        user_id: str,
        conversation_id: str,
        wcd_url: str | None = None,
        wcd_api_key: str | None = None,
    ):
        """
        Check if a tree exists in a Weaviate instance (set in the frontend config).

        Args:
            user_id (str): Required. The unique identifier for the user stored in the UserManager.
            conversation_id (str): Required. The unique identifier for the conversation for the user.
            wcd_url (str | None): Required. The URL of the Weaviate Cloud Database instance used to save the tree.
                Defaults to the value of the `wcd_url` setting in the frontend config.
            wcd_api_key (str | None): Required. The API key for the Weaviate Cloud Database instance used to save the tree.
                Defaults to the value of the `wcd_api_key` setting in the frontend config.

        Returns:
            (bool): True if the tree exists in the Weaviate instance, False otherwise.
        """

        local_user = await self.get_user_local(user_id)
        tree_manager: TreeManager = local_user["tree_manager"]

        if wcd_url is None or wcd_api_key is None:
            save_location_client_manager = local_user[
                "frontend_config"
            ].save_location_client_manager
        else:
            save_location_client_manager = ClientManager(
                logger=logger,
                wcd_url=wcd_url,
                wcd_api_key=wcd_api_key,
            )

        return await tree_manager.check_tree_exists_weaviate(
            conversation_id, save_location_client_manager
        )

    async def load_tree(
        self,
        user_id: str,
        conversation_id: str,
        wcd_url: str | None = None,
        wcd_api_key: str | None = None,
    ):
        """
        Load a tree from a Weaviate instance (set in the frontend config).
        This is a wrapper for the TreeManager.load_tree_weaviate() method.

        Args:
            user_id (str): Required. The unique identifier for the user stored in the UserManager.
            conversation_id (str): Required. The unique identifier for the conversation for the user.
            wcd_url (str | None): Required. The URL of the Weaviate Cloud Database instance used to save the tree.
                Defaults to the value of the `wcd_url` setting in the frontend config.
            wcd_api_key (str | None): Required. The API key for the Weaviate Cloud Database instance used to save the tree.
                Defaults to the value of the `wcd_api_key` setting in the frontend config.

        Returns:
            (list[dict]): A list of dictionaries, each containing a frontend payload that was used to generate the tree.
                The list is ordered by the time the payload was originally sent to the frontend (at the time it was saved).
        """

        local_user = await self.get_user_local(user_id)
        tree_manager: TreeManager = local_user["tree_manager"]

        if wcd_url is None or wcd_api_key is None:
            save_location_client_manager = local_user[
                "frontend_config"
            ].save_location_client_manager
        else:
            save_location_client_manager = ClientManager(
                logger=logger,
                wcd_url=wcd_url,
                wcd_api_key=wcd_api_key,
            )

        return await tree_manager.load_tree_weaviate(
            conversation_id, save_location_client_manager
        )

    async def delete_tree(
        self,
        user_id: str,
        conversation_id: str,
        wcd_url: str | None = None,
        wcd_api_key: str | None = None,
    ):
        """
        Delete a saved tree from a Weaviate instance (set in the frontend config).
        Also delete the tree from the local tree manager.
        This is a wrapper for the TreeManager.delete_tree_weaviate() method and the TreeManager.delete_tree_local() method.

        Args:
            user_id (str): Required. The unique identifier for the user stored in the UserManager.
            conversation_id (str): Required. The unique identifier for the conversation for the user.
            wcd_url (str | None): Required. The URL of the Weaviate Cloud Database instance used to save the tree.
                Defaults to the value of the `wcd_url` setting in the frontend config.
            wcd_api_key (str | None): Required. The API key for the Weaviate Cloud Database instance used to save the tree.
                Defaults to the value of the `wcd_api_key` setting in the frontend config.
        """

        local_user = await self.get_user_local(user_id)
        tree_manager: TreeManager = local_user["tree_manager"]

        if wcd_url is None or wcd_api_key is None:
            save_location_client_manager = local_user[
                "frontend_config"
            ].save_location_client_manager
        else:
            save_location_client_manager = ClientManager(
                logger=logger,
                wcd_url=wcd_url,
                wcd_api_key=wcd_api_key,
            )

        await tree_manager.delete_tree_weaviate(
            conversation_id, save_location_client_manager
        )
        tree_manager.delete_tree_local(conversation_id)

    async def get_saved_trees(
        self,
        user_id: str,
        wcd_url: str | None = None,
        wcd_api_key: str | None = None,
    ):
        """
        Get all saved trees from a Weaviate instance (set in the frontend config).

        Args:
            user_id (str): Required. The unique identifier for the user stored in the UserManager.
            wcd_url (str | None): Required. The URL of the Weaviate Cloud Database instance used to save the tree.
                Defaults to the value of the `wcd_url` setting in the frontend config.
            wcd_api_key (str | None): Required. The API key for the Weaviate Cloud Database instance used to save the tree.
                Defaults to the value of the `wcd_api_key` setting in the frontend config.

        Returns:
            (dict): A dictionary whose keys are the conversation IDs and whose values are dictionaries containing the title and last update time of the tree.
                E.g.
                ```
                {
                    "12345678-XXX-YYYY-ZZZZ": {
                        "title": "Query Request",
                        "last_update_time": "2025-06-07T10:06:47.376000Z"
                    }
                }
                ```
        """
        local_user = await self.get_user_local(user_id)

        if wcd_url is None or wcd_api_key is None:
            save_location_client_manager = local_user[
                "frontend_config"
            ].save_location_client_manager
        else:
            save_location_client_manager = ClientManager(
                logger=logger,
                wcd_url=wcd_url,
                wcd_api_key=wcd_api_key,
            )

        return await get_saved_trees_weaviate(
            "ELYSIA_TREES__", save_location_client_manager, user_id
        )

    async def update_user_last_request(self, user_id: str):
        self.users[user_id]["last_request"] = datetime.datetime.now()

    def check_tree_timeout(self, user_id: str, conversation_id: str):
        if user_id not in self.users:
            return True
        elif conversation_id not in self.users[user_id]["tree_manager"].trees:
            return True
        return False

    async def process_tree(
        self,
        query: str,
        user_id: str,
        conversation_id: str,
        query_id: str,
        training_route: str = "",
        collection_names: list[str] = [],
        save_trees_to_weaviate: bool | None = None,
        wcd_url: str | None = None,
        wcd_api_key: str | None = None,
    ):
        """
        Wrapper for the TreeManager.process_tree() method.
        Which itself is a wrapper for the Tree.async_run() method.
        This is an async generator which yields results from the tree.async_run() method.
        Automatically sends error payloads if the user or tree has been timed out.

        Args:
            query (str): Required. The user input/prompt to process in the decision tree.
            user_id (str): Required. The unique identifier for the user.
            conversation_id (str): Required. The conversation ID which contains the tree.
                This should be the same conversation ID as the one used to initialise the tree (see `initialise_tree`).
            query_id (str): Required. A unique identifier for the query.
            training_route (str): Optional. The training route, a string of the form "tool1/tool2/tool1" etc.
                See the `tree.async_run()` method for more details.
            collection_names (list[str]): Optional. A list of collection names to use in the query.
                If not supplied, all collections will be used.
            save_trees_to_weaviate (bool | None): Optional. Whether to save the trees to a Weaviate instance,
                after the process_tree() method has finished.
                Defaults to the value of the `save_trees_to_weaviate` setting in the frontend config.
            wcd_url (str | None): Required. The URL of the Weaviate Cloud Database instance used to save the tree.
                Defaults to the value of the `wcd_url` setting in the frontend config.
            wcd_api_key (str | None): Required. The API key for the Weaviate Cloud Database instance used to save the tree.
                Defaults to the value of the `wcd_api_key` setting in the frontend config.
        """

        if self.check_user_timeout(user_id):
            user_timeout_error = UserTimeoutError()
            error_payload = await user_timeout_error.to_frontend(
                user_id, conversation_id, query_id
            )
            yield error_payload
            return

        if self.check_tree_timeout(user_id, conversation_id):
            if await self.check_tree_exists_weaviate(user_id, conversation_id):
                await self.load_tree(user_id, conversation_id)
            else:
                tree_timeout_error = TreeTimeoutError()
                error_payload = await tree_timeout_error.to_frontend(
                    user_id, conversation_id, query_id
                )
                yield error_payload
                return

        local_user = await self.get_user_local(user_id)
        await self.update_user_last_request(user_id)

        tree_manager: TreeManager = local_user["tree_manager"]

        async for yielded_result in tree_manager.process_tree(
            query,
            conversation_id,
            query_id,
            training_route,
            collection_names,
            local_user["client_manager"],
        ):
            yield yielded_result
            await self.update_user_last_request(user_id)

        if save_trees_to_weaviate is None:
            frontend_config: FrontendConfig = local_user["frontend_config"]
            save_trees_to_weaviate = frontend_config.config["save_trees_to_weaviate"]

        if save_trees_to_weaviate:
            await self.save_tree(user_id, conversation_id, wcd_url, wcd_api_key)
