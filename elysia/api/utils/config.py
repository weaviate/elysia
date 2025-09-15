from logging import Logger
import datetime
import os
from typing import Literal, Optional
from uuid import uuid4

from elysia.config import Settings
from elysia.util.client import ClientManager

BranchInitType = Literal["default", "one_branch", "multi_branch", "empty"]


class Config:

    def __init__(
        self,
        id: str | None = None,
        name: str | None = None,
        settings: Settings | None = None,
        style: str | None = None,
        agent_description: str | None = None,
        end_goal: str | None = None,
        branch_initialisation: BranchInitType = "one_branch",
        use_elysia_collections: bool = True,
    ):

        if id is None:
            self.id = str(uuid4())
        else:
            self.id = id

        if name is None:
            self.name = "New Config"
        else:
            self.name = name

        if settings is None:
            self.settings = Settings().from_smart_setup()
        else:
            self.settings = settings

        if style is None:
            self.style = "Informative, polite and friendly."
        else:
            self.style = style

        if agent_description is None:
            self.agent_description = "You search and query Weaviate to satisfy the user's query, providing a concise summary of the results."
        else:
            self.agent_description = agent_description

        if end_goal is None:
            self.end_goal = (
                "You have satisfied the user's query, and provided a concise summary of the results. "
                "Or, you have exhausted all options available, or asked the user for clarification."
            )
        else:
            self.end_goal = end_goal

        self.branch_initialisation: BranchInitType = branch_initialisation
        self.use_elysia_collections: bool = use_elysia_collections

    def to_json(self):
        return {
            "id": self.id,
            "name": self.name,
            "settings": self.settings.to_json(),
            "style": self.style,
            "agent_description": self.agent_description,
            "end_goal": self.end_goal,
            "branch_initialisation": (
                self.branch_initialisation
                if self.branch_initialisation is not None
                else "one_branch"
            ),
            "use_elysia_collections": self.use_elysia_collections,
        }

    @classmethod
    def from_json(cls, json: dict):
        if "id" not in json and "config_id" in json:
            json["id"] = json["config_id"]

        return cls(
            id=json["id"],
            name=json["name"],
            settings=Settings.from_json(json["settings"]),
            style=json["style"],
            agent_description=json["agent_description"],
            end_goal=json["end_goal"],
            branch_initialisation=json["branch_initialisation"],
            use_elysia_collections=(
                json["use_elysia_collections"]
                if "use_elysia_collections" in json
                else True
            ),
        )


class FrontendConfig:
    """
    Handles frontend-specific config options.
    Anything that does not fit into the Elysia backend/python package, but relevant to the app is here.
    """

    def __init__(self, logger: Logger):
        """
        Args:
            logger (Logger): The logger to use for logging.
        """

        self.logger = logger

        client_timeout = int(os.getenv("CLIENT_TIMEOUT", 3))
        tree_timeout = int(os.getenv("TREE_TIMEOUT", 10))

        self.config: dict = {  # default values
            "save_trees_to_weaviate": True,
            "save_configs_to_weaviate": True,
            "client_timeout": client_timeout,
            "tree_timeout": tree_timeout,
        }
        self.save_location_wcd_url: str = os.getenv("WCD_URL", "")
        self.save_location_wcd_api_key: str = os.getenv("WCD_API_KEY", "")
        self.save_location_client_manager: ClientManager = ClientManager(
            wcd_url=self.save_location_wcd_url,
            wcd_api_key=self.save_location_wcd_api_key,
            logger=logger,
            client_timeout=self.config["client_timeout"],
        )

    def update_save_location(self, wcd_url: str, wcd_api_key: str):
        self.save_location_wcd_url = wcd_url
        self.save_location_wcd_api_key = wcd_api_key

    def get_save_location(self):
        return self.save_location_wcd_url, self.save_location_wcd_api_key

    def to_json(self):
        return {
            "save_trees_to_weaviate": self.config["save_trees_to_weaviate"],
            "save_configs_to_weaviate": self.config["save_configs_to_weaviate"],
            "save_location_wcd_url": self.save_location_wcd_url,
            "save_location_wcd_api_key": self.save_location_wcd_api_key,
            "client_timeout": self.config["client_timeout"],
            "tree_timeout": self.config["tree_timeout"],
        }

    async def configure(self, **kwargs):
        """
        Args:
            **kwargs:
                - save_trees_to_weaviate (bool): Whether to save trees to Weaviate.
                - save_config_to_weaviate (bool): Whether to save config to Weaviate.
                - tree_timeout (int): Optional.
                    The length of time in minutes a tree can be idle before being timed out.
                    Defaults to 10 minutes or the value of the TREE_TIMEOUT environment variable (integer, minutes).
                    If an integer is provided, it is interpreted as the number of minutes.
                - client_timeout (int): Optional.
                    The length of time in minutes a client can be idle before being timed out.
                    Defaults to 3 minutes or the value of the CLIENT_TIMEOUT environment variable (integer, minutes).
                    If an integer is provided, it is interpreted as the number of minutes.
                - save_location_wcd_url (str): Optional.
                    The URL of the Weaviate database to save trees/configs to.
                    Defaults to the value of the WCD_URL environment variable.
                - save_location_wcd_api_key (str): Optional.
                    The API key for the Weaviate database to save trees/configs to.
                    Defaults to the value of the WCD_API_KEY environment variable.
        """

        reload_client_manager = False
        if "save_location_wcd_url" in kwargs:
            self.save_location_wcd_url = kwargs["save_location_wcd_url"]
            reload_client_manager = True
        if "save_location_wcd_api_key" in kwargs:
            self.save_location_wcd_api_key = kwargs["save_location_wcd_api_key"]
            reload_client_manager = True
        if "save_trees_to_weaviate" in kwargs:
            self.config["save_trees_to_weaviate"] = kwargs["save_trees_to_weaviate"]
        if "save_configs_to_weaviate" in kwargs:
            self.config["save_configs_to_weaviate"] = kwargs["save_configs_to_weaviate"]
        if "client_timeout" in kwargs:
            self.config["client_timeout"] = kwargs["client_timeout"]
            self.save_location_client_manager.client_timeout = datetime.timedelta(
                minutes=self.config["client_timeout"]
            )
        if "tree_timeout" in kwargs:
            self.config["tree_timeout"] = kwargs["tree_timeout"]

        if reload_client_manager:
            await self.save_location_client_manager.reset_keys(
                wcd_url=self.save_location_wcd_url,
                wcd_api_key=self.save_location_wcd_api_key,
            )

    @classmethod
    async def from_json(cls, json: dict, logger: Logger):
        fe_config = cls(logger=logger)
        await fe_config.configure(**json)

        return fe_config
