from logging import Logger
import datetime
import os
from typing import Literal, Optional
from uuid import uuid4

from elysia.config import Settings
from elysia.util.client import ClientManager
from elysia.api.api_types import ToolItem, BranchInfo, ToolPreset
from pydantic import BaseModel

BranchInitType = Literal["default", "one_branch", "multi_branch", "empty"]

from elysia.api.utils.tools import get_presets_weaviate


class ToolPresetManager:
    def __init__(self):
        self.tool_presets = []

    def add(
        self,
        preset_id: str,
        name: str,
        order: list[ToolItem],
        branches: list[BranchInfo],
        default: bool,
    ):

        self.remove(preset_id)
        if default:
            for preset in self.tool_presets:
                preset.default = False

        self.tool_presets.append(
            ToolPreset(
                preset_id=preset_id,
                name=name,
                order=order,
                branches=branches,
                default=default,
            )
        )

    def remove(self, preset_id: str):
        self.tool_presets = [
            preset for preset in self.tool_presets if preset.preset_id != preset_id
        ]

    def get(self, preset_id: str):
        return next(
            (preset for preset in self.tool_presets if preset.preset_id == preset_id),
            None,
        )

    async def retrieve(self, user_id: str, client_manager: ClientManager):
        self.tool_presets = await get_presets_weaviate(user_id, client_manager)

    def to_json(self):
        if self.tool_presets is None:
            return None
        return [preset.model_dump() for preset in self.tool_presets]


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

        # Cloud connection settings
        self.save_location_wcd_url: str = os.getenv("WCD_URL", "")
        self.save_location_wcd_api_key: str = os.getenv("WCD_API_KEY", "")

        # Local connection settings
        self.save_location_weaviate_is_local: bool = (
            os.getenv("WEAVIATE_IS_LOCAL", "False") == "True"
        )
        self.save_location_local_weaviate_port: int = int(
            os.getenv("LOCAL_WEAVIATE_PORT", 8080)
        )
        self.save_location_local_weaviate_grpc_port: int = int(
            os.getenv("LOCAL_WEAVIATE_GRPC_PORT", 50051)
        )

        # Custom connection settings
        self.save_location_weaviate_is_custom: bool = (
            os.getenv("WEAVIATE_IS_CUSTOM", "False") == "True"
        )
        self.save_location_custom_http_host: str = os.getenv("CUSTOM_HTTP_HOST", "")
        self.save_location_custom_http_port: int = int(
            os.getenv("CUSTOM_HTTP_PORT", 8080)
        )
        self.save_location_custom_http_secure: bool = (
            os.getenv("CUSTOM_HTTP_SECURE", "False") == "True"
        )
        self.save_location_custom_grpc_host: str = os.getenv("CUSTOM_GRPC_HOST", "")
        self.save_location_custom_grpc_port: int = int(
            os.getenv("CUSTOM_GRPC_PORT", 50051)
        )
        self.save_location_custom_grpc_secure: bool = (
            os.getenv("CUSTOM_GRPC_SECURE", "False") == "True"
        )

        # Initialise the client manager
        self.save_location_client_manager: ClientManager = ClientManager(
            wcd_url=self.save_location_wcd_url,
            wcd_api_key=self.save_location_wcd_api_key,
            weaviate_is_local=self.save_location_weaviate_is_local,
            local_weaviate_port=self.save_location_local_weaviate_port,
            local_weaviate_grpc_port=self.save_location_local_weaviate_grpc_port,
            weaviate_is_custom=self.save_location_weaviate_is_custom,
            custom_http_host=self.save_location_custom_http_host,
            custom_http_port=self.save_location_custom_http_port,
            custom_http_secure=self.save_location_custom_http_secure,
            custom_grpc_host=self.save_location_custom_grpc_host,
            custom_grpc_port=self.save_location_custom_grpc_port,
            custom_grpc_secure=self.save_location_custom_grpc_secure,
            logger=logger,
            client_timeout=self.config["client_timeout"],
        )

    def to_json(self):
        return {
            "save_trees_to_weaviate": self.config["save_trees_to_weaviate"],
            "save_configs_to_weaviate": self.config["save_configs_to_weaviate"],
            "save_location_wcd_url": self.save_location_wcd_url,
            "save_location_wcd_api_key": self.save_location_wcd_api_key,
            "save_location_weaviate_is_local": self.save_location_weaviate_is_local,
            "save_location_local_weaviate_port": self.save_location_local_weaviate_port,
            "save_location_local_weaviate_grpc_port": self.save_location_local_weaviate_grpc_port,
            "save_location_weaviate_is_custom": self.save_location_weaviate_is_custom,
            "save_location_custom_http_host": self.save_location_custom_http_host,
            "save_location_custom_http_port": self.save_location_custom_http_port,
            "save_location_custom_http_secure": self.save_location_custom_http_secure,
            "save_location_custom_grpc_host": self.save_location_custom_grpc_host,
            "save_location_custom_grpc_port": self.save_location_custom_grpc_port,
            "save_location_custom_grpc_secure": self.save_location_custom_grpc_secure,
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
                - save_location_weaviate_is_local (bool): Optional.
                    Whether the Weaviate database is local.
                    Defaults to the value of the WEAVIATE_IS_LOCAL environment variable.
                - save_location_local_weaviate_port (int): Optional.
                    The port of the local Weaviate database.
                    Defaults to the value of the LOCAL_WEAVIATE_PORT environment variable.
                - save_location_local_weaviate_grpc_port (int): Optional.
                    The gRPC port of the local Weaviate database.
                    Defaults to the value of the LOCAL_WEAVIATE_GRPC_PORT environment variable.
                - save_location_weaviate_is_custom (bool): Optional.
                    Whether the Weaviate database is custom.
                    Defaults to the value of the WEAVIATE_IS_CUSTOM environment variable.
                - save_location_custom_http_host (str): Optional.
                    The HTTP host of the custom Weaviate database.
                    Defaults to the value of the CUSTOM_HTTP_HOST environment variable.
                - save_location_custom_http_port (int): Optional.
                    The HTTP port of the custom Weaviate database.
                    Defaults to the value of the CUSTOM_HTTP_PORT environment variable.
                - save_location_custom_http_secure (bool): Optional.
                    Whether the HTTP connection to the custom Weaviate database is secure.
                    Defaults to the value of the CUSTOM_HTTP_SECURE environment variable.
                - save_location_custom_grpc_host (str): Optional.
                    The gRPC host of the custom Weaviate database.
                    Defaults to the value of the CUSTOM_GRPC_HOST environment variable.
                - save_location_custom_grpc_port (int): Optional.
                    The gRPC port of the custom Weaviate database.
                    Defaults to the value of the CUSTOM_GRPC_PORT environment variable.
                - save_location_custom_grpc_secure (bool): Optional.
                    Whether the gRPC connection to the custom Weaviate database is secure.
                    Defaults to the value of the CUSTOM_GRPC_SECURE environment variable.
        """

        reload_client_manager = False
        if "save_location_wcd_url" in kwargs:
            self.save_location_wcd_url = kwargs["save_location_wcd_url"]
            reload_client_manager = True
        if "save_location_wcd_api_key" in kwargs:
            self.save_location_wcd_api_key = kwargs["save_location_wcd_api_key"]
            reload_client_manager = True
        if "save_location_weaviate_is_local" in kwargs:
            self.save_location_weaviate_is_local = kwargs[
                "save_location_weaviate_is_local"
            ]
            reload_client_manager = True
        if "save_location_local_weaviate_port" in kwargs:
            self.save_location_local_weaviate_port = kwargs[
                "save_location_local_weaviate_port"
            ]
            reload_client_manager = True
        if "save_location_local_weaviate_grpc_port" in kwargs:
            self.save_location_local_weaviate_grpc_port = kwargs[
                "save_location_local_weaviate_grpc_port"
            ]
            reload_client_manager = True

        if "save_location_weaviate_is_custom" in kwargs:
            self.save_location_weaviate_is_custom = kwargs[
                "save_location_weaviate_is_custom"
            ]
            reload_client_manager = True
        if "save_location_custom_http_host" in kwargs:
            self.save_location_custom_http_host = kwargs[
                "save_location_custom_http_host"
            ]
            reload_client_manager = True
        if "save_location_custom_http_port" in kwargs:
            self.save_location_custom_http_port = kwargs[
                "save_location_custom_http_port"
            ]
            reload_client_manager = True
        if "save_location_custom_http_secure" in kwargs:
            self.save_location_custom_http_secure = kwargs[
                "save_location_custom_http_secure"
            ]
            reload_client_manager = True
        if "save_location_custom_grpc_host" in kwargs:
            self.save_location_custom_grpc_host = kwargs[
                "save_location_custom_grpc_host"
            ]
            reload_client_manager = True
        if "save_location_custom_grpc_port" in kwargs:
            self.save_location_custom_grpc_port = kwargs[
                "save_location_custom_grpc_port"
            ]
            reload_client_manager = True
        if "save_location_custom_grpc_secure" in kwargs:
            self.save_location_custom_grpc_secure = kwargs[
                "save_location_custom_grpc_secure"
            ]
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
                weaviate_is_local=self.save_location_weaviate_is_local,
                local_weaviate_port=self.save_location_local_weaviate_port,
                local_weaviate_grpc_port=self.save_location_local_weaviate_grpc_port,
                weaviate_is_custom=self.save_location_weaviate_is_custom,
                custom_http_host=self.save_location_custom_http_host,
                custom_http_port=self.save_location_custom_http_port,
                custom_http_secure=self.save_location_custom_http_secure,
                custom_grpc_host=self.save_location_custom_grpc_host,
                custom_grpc_port=self.save_location_custom_grpc_port,
                custom_grpc_secure=self.save_location_custom_grpc_secure,
            )

    @classmethod
    async def from_json(cls, json: dict, logger: Logger):
        fe_config = cls(logger=logger)
        await fe_config.configure(**json)

        return fe_config
