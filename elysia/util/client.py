import asyncio
import datetime
import os
import threading

from contextlib import asynccontextmanager, contextmanager
from typing import AsyncGenerator, Generator, Any
from logging import Logger
from urllib.parse import urlparse

import weaviate
from weaviate.classes.init import Auth, Timeout
from weaviate.config import AdditionalConfig
from weaviate.client import WeaviateClient, WeaviateAsyncClient
from elysia.config import settings as environment_settings, Settings

api_key_map = {
    # Regular API keys
    "ANTHROPIC_APIKEY": "X-Anthropic-Api-Key",
    "ANYSCALE_APIKEY": "X-Anyscale-Api-Key",
    "AWS_ACCESS_KEY": "X-Aws-Access-Key",
    "AWS_SECRET_KEY": "X-Aws-Secret-Key",
    "COHERE_API_KEY": "X-Cohere-Api-Key",
    "DATABRICKS_TOKEN": "X-Databricks-Token",
    "FRIENDLI_TOKEN": "X-Friendli-Api-Key",
    "VERTEX_APIKEY": "X-Goog-Vertex-Api-Key",
    "STUDIO_APIKEY": "X-Goog-Studio-Api-Key",
    "HUGGINGFACE_APIKEY": "X-HuggingFace-Api-Key",
    "JINAAI_APIKEY": "X-JinaAI-Api-Key",
    "MISTRAL_APIKEY": "X-Mistral-Api-Key",
    "NVIDIA_APIKEY": "X-Nvidia-Api-Key",
    "OPENAI_APIKEY": "X-OpenAI-Api-Key",
    "AZURE_APIKEY": "X-Azure-Api-Key",
    "VOYAGE_APIKEY": "X-Voyage-Api-Key",
    "XAI_APIKEY": "X-Xai-Api-Key",
    # And separate out "API_KEY"
    "ANTHROPIC_API_KEY": "X-Anthropic-Api-Key",
    "ANYSCALE_API_KEY": "X-Anyscale-Api-Key",
    "AWS_ACCESS_KEY": "X-Aws-Access-Key",
    "AWS_SECRET_KEY": "X-Aws-Secret-Key",
    "COHERE_API_KEY": "X-Cohere-Api-Key",
    "DATABRICKS_TOKEN": "X-Databricks-Token",
    "FRIENDLI_TOKEN": "X-Friendli-Api-Key",
    "VERTEX_API_KEY": "X-Goog-Vertex-Api-Key",
    "STUDIO_API_KEY": "X-Goog-Studio-Api-Key",
    "HUGGINGFACE_API_KEY": "X-HuggingFace-Api-Key",
    "JINAAI_API_KEY": "X-JinaAI-Api-Key",
    "MISTRAL_API_KEY": "X-Mistral-Api-Key",
    "NVIDIA_API_KEY": "X-Nvidia-Api-Key",
    "OPENAI_API_KEY": "X-OpenAI-Api-Key",
    "AZURE_API_KEY": "X-Azure-Api-Key",
    "VOYAGE_API_KEY": "X-Voyage-Api-Key",
    "XAI_API_KEY": "X-Xai-Api-Key",
}


class ClientManager:
    """
    Handles the creation and management of the Weaviate client.
    Handles cases where the client can be used in more than one thread or async operation at a time,
    via threading and asyncio locks.
    Also can use methods for restarting client if its been inactive.
    """

    def __init__(
        self,
        wcd_url: str | None = None,
        wcd_api_key: str | None = None,
        weaviate_is_local: bool | None = None,
        weaviate_is_custom: bool | None = None,
        local_weaviate_port: int | None = None,
        local_weaviate_grpc_port: int | None = None,
        custom_http_host: str | None = None,
        custom_http_port: int | None = None,
        custom_http_secure: bool | None = None,
        custom_grpc_host: str | None = None,
        custom_grpc_port: int | None = None,
        custom_grpc_secure: bool | None = None,
        client_timeout: datetime.timedelta | int | None = None,
        logger: Logger | None = None,
        settings: Settings | None = None,
        query_timeout: int = 60,
        insert_timeout: int = 120,
        init_timeout: int = 5,
        **kwargs,
    ) -> None:
        """
        Args:
            wcd_url (str): the url of the Weaviate cluster. Defaults to global settings config.
            wcd_api_key (str): the api key for the Weaviate cluster. Defaults to global settings config.
            weaviate_is_local (bool): whether the weaviate cluster is local. Defaults to False.
            weaviate_is_custom (bool): whether to use custom connection parameters. Defaults to False.
            custom_http_host (str): HTTP host for custom connection.
            custom_http_port (int): HTTP port for custom connection. Defaults to 8080.
            custom_http_secure (bool): Use HTTPS for custom connection. Defaults to False.
            custom_grpc_host (str): gRPC host for custom connection.
            custom_grpc_port (int): gRPC port for custom connection. Defaults to 50051.
            custom_grpc_secure (bool): Use secure gRPC for custom connection. Defaults to False.
            client_timeout (datetime.timedelta | int | None): how long (in minutes) means the client should be restarted. Defaults to 3 minutes.
            logger (Logger | None): a logger object for logging messages. Defaults to None.
            settings (Settings | None): a settings object for the client manager. Defaults to environment settings.
            query_timeout (int): the timeout for Weaviate queries. Defaults to 60 seconds (Weaviate default is 30 seconds).
            insert_timeout (int): the timeout for Weaviate inserts. Defaults to 120 seconds (Weaviate default is 90 seconds).
            init_timeout (int): the timeout for Weaviate initialisation. Defaults to 5 seconds (Weaviate default is 2 seconds).
            **kwargs (Any): any other api keys for third party services (formatted as e.g. OPENAI_APIKEY).

        Example:
        ```python
        # Cloud connection
        client_manager = ClientManager(
            wcd_url="https://my-weaviate-cluster...",
            wcd_api_key="my-api-key...",
            OPENAI_APIKEY="my-openai-api-key...",
        )
        
        # Local connection
        client_manager = ClientManager(
            weaviate_is_local=True,
            local_weaviate_port=8080,
            local_weaviate_grpc_port=50051,
        )
        
        # Custom connection
        client_manager = ClientManager(
            weaviate_is_custom=True,
            custom_http_host="custom.weaviate.host",
            custom_http_port=8080,
            custom_http_secure=True,
            custom_grpc_host="custom.weaviate.host",
            custom_grpc_port=50051,
            custom_grpc_secure=True,
            wcd_api_key="optional-api-key",
        )
        ```
        """

        self.logger = logger

        if client_timeout is None:
            self.client_timeout = datetime.timedelta(
                minutes=int(os.getenv("CLIENT_TIMEOUT", 3))
            )
        elif isinstance(client_timeout, int):
            self.client_timeout = datetime.timedelta(minutes=client_timeout)
        else:
            self.client_timeout = client_timeout

        if settings is None:
            self.settings = environment_settings
        else:
            self.settings = settings

        # Set the weaviate url and api key
        if wcd_url is None:
            self.wcd_url = self.settings.WCD_URL
        else:
            self.wcd_url = wcd_url

        if wcd_api_key is None:
            self.wcd_api_key = self.settings.WCD_API_KEY
        else:
            self.wcd_api_key = wcd_api_key

        if weaviate_is_local is None:
            self.weaviate_is_local = self.settings.WEAVIATE_IS_LOCAL
        else:
            self.weaviate_is_local = weaviate_is_local

        if weaviate_is_custom is None:
            self.weaviate_is_custom = self.settings.WEAVIATE_IS_CUSTOM
        else:
            self.weaviate_is_custom = weaviate_is_custom

        if local_weaviate_port is None:
            self.local_weaviate_port = self.settings.LOCAL_WEAVIATE_PORT
        else:
            self.local_weaviate_port = local_weaviate_port

        if local_weaviate_grpc_port is None:
            self.local_weaviate_grpc_port = self.settings.LOCAL_WEAVIATE_GRPC_PORT
        else:
            self.local_weaviate_grpc_port = local_weaviate_grpc_port

        # Custom connection parameters
        if custom_http_host is None:
            self.custom_http_host = self.settings.CUSTOM_HTTP_HOST
        else:
            self.custom_http_host = custom_http_host

        if custom_http_port is None:
            self.custom_http_port = self.settings.CUSTOM_HTTP_PORT
        else:
            self.custom_http_port = custom_http_port

        if custom_http_secure is None:
            self.custom_http_secure = self.settings.CUSTOM_HTTP_SECURE
        else:
            self.custom_http_secure = custom_http_secure

        if custom_grpc_host is None:
            self.custom_grpc_host = self.settings.CUSTOM_GRPC_HOST
        else:
            self.custom_grpc_host = custom_grpc_host

        if custom_grpc_port is None:
            self.custom_grpc_port = self.settings.CUSTOM_GRPC_PORT
        else:
            self.custom_grpc_port = custom_grpc_port

        if custom_grpc_secure is None:
            self.custom_grpc_secure = self.settings.CUSTOM_GRPC_SECURE
        else:
            self.custom_grpc_secure = custom_grpc_secure

        self.query_timeout = query_timeout
        self.insert_timeout = insert_timeout
        self.init_timeout = init_timeout

        if self.weaviate_is_local and (self.wcd_url is None or self.wcd_url == ""):
            self.wcd_url = "localhost"

        # Set the api keys for non weaviate cluster (third parties)
        self.headers = {}
        for api_key in self.settings.API_KEYS:
            if api_key.lower() in [a.lower() for a in api_key_map.keys()]:
                self.headers[api_key_map[api_key.upper()]] = self.settings.API_KEYS[
                    api_key
                ]

        # From kwargs
        for kwarg in kwargs:
            if kwarg.lower() in [a.lower() for a in api_key_map.keys()]:
                self.headers[api_key_map[kwarg.upper()]] = kwargs[kwarg]

        # Create locks for client events
        self.async_lock = asyncio.Lock()
        self.sync_lock = threading.Lock()

        # In use counter tracks when the client is in use and by how many operations. 0 = can restart
        self.async_in_use_counter = 0
        self.sync_in_use_counter = 0
        self.async_restart_event = asyncio.Event()
        self.sync_restart_event = threading.Event()

        self.last_used_sync_client = datetime.datetime.now()
        self.last_used_async_client = datetime.datetime.now()

        self.async_client = None
        self.async_init_completed = False
        
        # Determine if client is properly configured
        if self.weaviate_is_custom:
            # Custom mode requires http_host and grpc_host
            self.is_client = (
                self.custom_http_host is not None and 
                self.custom_grpc_host is not None
            )
        elif self.weaviate_is_local:
            # Local mode requires wcd_url (or defaults to localhost)
            self.is_client = self.wcd_url != ""
        else:
            # Cloud mode requires both url and api_key
            self.is_client = self.wcd_url != "" and self.wcd_api_key != ""

        if self.logger and not self.is_client:
            if self.weaviate_is_custom:
                self.logger.warning(
                    "Custom Weaviate connection parameters not fully configured. "
                    "CUSTOM_HTTP_HOST and CUSTOM_GRPC_HOST must be set. "
                    "All Weaviate functionality will be disabled."
                )
            elif self.wcd_url == "" and self.weaviate_is_local:
                self.logger.warning(
                    "WCD_URL not set for local Weaviate (This should probably be localhost). "
                    "All Weaviate functionality will be disabled."
                )
            elif (
                not self.weaviate_is_local
                and self.wcd_api_key == ""
                and self.wcd_url != ""
            ):
                self.logger.warning(
                    "WCD_API_KEY and WCD_URL are not set. "
                    "All Weaviate functionality will be disabled."
                )
            elif self.wcd_url == "" and not self.weaviate_is_local:
                self.logger.warning(
                    "WCD_URL is not set. "
                    "All Weaviate functionality will be disabled."
                )
            elif self.wcd_api_key == "" and not self.weaviate_is_local:
                self.logger.warning(
                    "WCD_API_KEY is not set. "
                    "All Weaviate functionality will be disabled."
                )
            else:
                self.logger.debug(
                    "Weaviate client initialised. "
                    "All Weaviate functionality will be enabled."
                )

        if not self.is_client:
            return

        # Start sync client
        try:
            self.client = self.get_client()
        except Exception as e:
            self.logger.error(
                "Error initialising Weaviate client. Please check your Weaviate configuration is set correctly (WCD_URL, WCD_API_KEY, WEAVIATE_IS_LOCAL, WEAVIATE_IS_CUSTOM, or custom connection parameters)."
            )
            self.logger.error(f"Full Weaviate connection error message: {e}")
            self.is_client = False
            return
        self.sync_restart_event.set()

    def _get_local_host_and_port(self) -> tuple[str, int]:
        """
        Derive host and port for local connections from wcd_url and configured ports.
        Accepts full URLs like "http://localhost:8080" and extracts hostname/port.
        """
        host = self.wcd_url if self.wcd_url is not None else "localhost"
        port = self.local_weaviate_port
        try:
            parsed = urlparse(host)
            if parsed.scheme in ("http", "https"):
                if parsed.hostname:
                    host = parsed.hostname
                if parsed.port:
                    port = parsed.port
            # If no scheme, assume the value is a bare hostname (optionally with :port)
            elif ":" in host:
                # Split manually to support host:port form without scheme
                parts = host.split(":")
                host = parts[0]
                try:
                    port = int(parts[1])
                except Exception:
                    port = self.local_weaviate_port
        except Exception:
            # Fallback to defaults
            host = "localhost" if not host else host
            port = self.local_weaviate_port
        return host, port

    async def reset_keys(
        self,
        wcd_url: str | None = None,
        wcd_api_key: str | None = None,
        api_keys: dict[str, str] = {},
        weaviate_is_local: bool = False,
        weaviate_is_custom: bool = False,
        local_weaviate_port: int = 8080,
        local_weaviate_grpc_port: int = 50051,
        custom_http_host: str | None = None,
        custom_http_port: int = 8080,
        custom_http_secure: bool = False,
        custom_grpc_host: str | None = None,
        custom_grpc_port: int = 50051,
        custom_grpc_secure: bool = False,
    ) -> None:
        """
        Set the API keys, WCD_URL and WCD_API_KEY from the settings object.

        Args:
            wcd_url (str): the url of the Weaviate cluster.
            wcd_api_key (str): the api key for the Weaviate cluster.
            api_keys (dict): a dictionary of api keys for third party services.
            weaviate_is_local (bool): whether the weaviate cluster is local.
            weaviate_is_custom (bool): whether to use custom connection parameters.
            local_weaviate_port (int): the port for local Weaviate HTTP.
            local_weaviate_grpc_port (int): the port for local Weaviate gRPC.
            custom_http_host (str): HTTP host for custom connection.
            custom_http_port (int): HTTP port for custom connection.
            custom_http_secure (bool): Use HTTPS for custom connection.
            custom_grpc_host (str): gRPC host for custom connection.
            custom_grpc_port (int): gRPC port for custom connection.
            custom_grpc_secure (bool): Use secure gRPC for custom connection.
        """
        self.wcd_url = wcd_url
        self.wcd_api_key = wcd_api_key
        self.weaviate_is_local = weaviate_is_local
        self.weaviate_is_custom = weaviate_is_custom
        self.local_weaviate_port = local_weaviate_port
        self.local_weaviate_grpc_port = local_weaviate_grpc_port
        
        # Update custom connection parameters
        self.custom_http_host = custom_http_host
        self.custom_http_port = custom_http_port
        self.custom_http_secure = custom_http_secure
        self.custom_grpc_host = custom_grpc_host
        self.custom_grpc_port = custom_grpc_port
        self.custom_grpc_secure = custom_grpc_secure

        # If using a local Weaviate instance and no URL was provided, default to localhost
        if self.weaviate_is_local and (self.wcd_url is None or self.wcd_url == ""):
            self.wcd_url = "localhost"

        self.headers = {}

        for api_key in api_keys:
            if api_key.lower() in [a.lower() for a in api_key_map.keys()]:
                self.headers[api_key_map[api_key.upper()]] = api_keys[api_key]

        # Update is_client check to handle custom mode
        if self.weaviate_is_custom:
            self.is_client = (
                self.custom_http_host is not None and 
                self.custom_grpc_host is not None
            )
        elif self.weaviate_is_local:
            self.is_client = self.wcd_url != ""
        else:
            self.is_client = self.wcd_url != "" and self.wcd_api_key != ""
            
        if self.is_client:
            await self.restart_client(force=True)
            await self.restart_async_client(force=True)
            await self.start_clients()

    async def start_clients(self) -> None:
        """
        Start the async and sync clients if they are not already running.
        """

        if not self.is_client:
            raise ValueError(
                "Weaviate is not available. Please set the WCD_URL and WCD_API_KEY in the settings."
            )

        if self.async_client is None:
            self.async_client = await self.get_async_client()
            self.async_restart_event.set()

        if not self.async_client.is_connected():
            await self.async_client.connect()

        self.async_init_completed = True

        if not self.client.is_connected():
            self.client.connect()

    def update_last_user_request(self) -> None:
        self.last_user_request = datetime.datetime.now()

    def update_last_used_sync_client(self) -> None:
        self.last_used_sync_client = datetime.datetime.now()

    def update_last_used_async_client(self) -> None:
        self.last_used_async_client = datetime.datetime.now()

    def get_client(self) -> WeaviateClient:
        # Custom connection mode
        if self.weaviate_is_custom:
            if self.custom_http_host is None or self.custom_grpc_host is None:
                raise ValueError("CUSTOM_HTTP_HOST and CUSTOM_GRPC_HOST must be set for custom connections")
            
            auth_credentials = (
                Auth.api_key(self.wcd_api_key) if self.wcd_api_key != "" else None
            )
            
            if self.logger:
                self.logger.debug(
                    f"Getting custom client with http_host: {self.custom_http_host}, "
                    f"http_port: {self.custom_http_port}, http_secure: {self.custom_http_secure}, "
                    f"grpc_host: {self.custom_grpc_host}, grpc_port: {self.custom_grpc_port}, "
                    f"grpc_secure: {self.custom_grpc_secure}, api_key_set: {self.wcd_api_key != ''}"
                )
            
            return weaviate.connect_to_custom(
                http_host=self.custom_http_host,
                http_port=self.custom_http_port,
                http_secure=self.custom_http_secure,
                grpc_host=self.custom_grpc_host,
                grpc_port=self.custom_grpc_port,
                grpc_secure=self.custom_grpc_secure,
                auth_credentials=auth_credentials,
                headers=self.headers,
                additional_config=AdditionalConfig(
                    timeout=Timeout(
                        query=self.query_timeout,
                        insert=self.insert_timeout,
                        init=self.init_timeout,
                    )
                ),
            )
        
        # Local connection mode
        if self.weaviate_is_local and self.wcd_url != "":
            auth_credentials = (
                Auth.api_key(self.wcd_api_key) if self.wcd_api_key != "" else None
            )
            host, port = self._get_local_host_and_port()
            if self.logger:
                self.logger.debug(
                    f"Getting client with weaviate_is_local: {self.weaviate_is_local}, "
                    f"wcd_url: {self.wcd_url}, parsed_host: {host}, api_key_set: {self.wcd_api_key != ''}, "
                    f"http_port: {port}, grpc_port: {self.local_weaviate_grpc_port}"
                )
            return weaviate.connect_to_local(
                host=host,
                port=port,
                grpc_port=self.local_weaviate_grpc_port,
                auth_credentials=auth_credentials,
                headers=self.headers,
                skip_init_checks=True,
            )

        # Cloud connection mode
        if self.wcd_url == "" or self.wcd_api_key == "":
            raise ValueError("WCD_URL and WCD_API_KEY must be set")

        return weaviate.connect_to_weaviate_cloud(
            cluster_url=self.wcd_url,
            auth_credentials=Auth.api_key(self.wcd_api_key),
            headers=self.headers,
            skip_init_checks=True,
            additional_config=AdditionalConfig(
                timeout=Timeout(
                    query=self.query_timeout,
                    insert=self.insert_timeout,
                    init=self.init_timeout,
                )
            ),
        )

    async def get_async_client(self) -> WeaviateAsyncClient:
        # Custom connection mode
        if self.weaviate_is_custom:
            if self.custom_http_host is None or self.custom_grpc_host is None:
                raise ValueError("CUSTOM_HTTP_HOST and CUSTOM_GRPC_HOST must be set for custom connections")
            
            auth_credentials = (
                Auth.api_key(self.wcd_api_key) if self.wcd_api_key != "" else None
            )
            
            if self.logger:
                self.logger.debug(
                    f"Getting async custom client with http_host: {self.custom_http_host}, "
                    f"http_port: {self.custom_http_port}, http_secure: {self.custom_http_secure}, "
                    f"grpc_host: {self.custom_grpc_host}, grpc_port: {self.custom_grpc_port}, "
                    f"grpc_secure: {self.custom_grpc_secure}, api_key_set: {self.wcd_api_key != ''}"
                )
            
            return weaviate.use_async_with_custom(
                http_host=self.custom_http_host,
                http_port=self.custom_http_port,
                http_secure=self.custom_http_secure,
                grpc_host=self.custom_grpc_host,
                grpc_port=self.custom_grpc_port,
                grpc_secure=self.custom_grpc_secure,
                auth_credentials=auth_credentials,
                headers=self.headers,
                skip_init_checks=True,
                additional_config=AdditionalConfig(
                    timeout=Timeout(
                        query=self.query_timeout,
                        insert=self.insert_timeout,
                        init=self.init_timeout,
                    )
                ),
            )
        
        # Local connection mode
        if self.weaviate_is_local and self.wcd_url != "":
            auth_credentials = (
                Auth.api_key(self.wcd_api_key) if self.wcd_api_key != "" else None
            )
            host, port = self._get_local_host_and_port()
            if self.logger:
                self.logger.debug(
                    f"Getting async client with weaviate_is_local: {self.weaviate_is_local}, "
                    f"wcd_url: {self.wcd_url}, parsed_host: {host}, api_key_set: {self.wcd_api_key != ''}, "
                    f"http_port: {port}, grpc_port: {self.local_weaviate_grpc_port}"
                )
            return weaviate.use_async_with_local(
                host=host,
                port=port,
                grpc_port=self.local_weaviate_grpc_port,
                auth_credentials=auth_credentials,
                headers=self.headers,
                skip_init_checks=True,
            )

        # Cloud connection mode
        if self.wcd_url == "" or self.wcd_api_key == "":
            raise ValueError("WCD_URL and WCD_API_KEY must be set")

        return weaviate.use_async_with_weaviate_cloud(
            cluster_url=self.wcd_url,
            auth_credentials=Auth.api_key(self.wcd_api_key),
            headers=self.headers,
            skip_init_checks=True,
            additional_config=AdditionalConfig(
                timeout=Timeout(
                    query=self.query_timeout,
                    insert=self.insert_timeout,
                    init=self.init_timeout,
                )
            ),
        )

    @contextmanager
    def connect_to_client(self) -> Generator[WeaviateClient, Any, None]:
        """
        A context manager to connect to the _sync_ client.

        E.g.

        ```python
        with client_manager.connect_to_client():
            # do stuff with the weaviate client
            ...
        ```
        """
        if not self.is_client:
            raise ValueError(
                "Weaviate is not available. Please set the WCD_URL and WCD_API_KEY in the settings or connect to a local Weaviate instance."
            )

        self.sync_restart_event.wait()
        with self.sync_lock:
            self.sync_in_use_counter += 1

        if not self.client.is_connected():
            self.client.connect()

        connection = _ClientConnection(self, self.client)
        with connection:
            yield connection.client

    @asynccontextmanager
    async def connect_to_async_client(
        self,
    ) -> AsyncGenerator[WeaviateAsyncClient, Any]:
        """
        A context manager to connect to the _async_ client.

        E.g.
        ```python
        async with client_manager.connect_to_async_client():
            # do stuff with the async weaviate client
            ...
        ```
        """
        if not self.is_client:
            raise ValueError(
                "Weaviate is not available. Please set the WCD_URL and WCD_API_KEY in the settings or connect to a local Weaviate instance."
            )

        if not self.async_init_completed:
            await self.start_clients()

        await self.async_restart_event.wait()
        async with self.async_lock:
            self.async_in_use_counter += 1

        if self.async_client is None:
            raise ValueError("Async client not initialised")

        if not self.async_client.is_connected():
            await self.async_client.connect()

        connection = _AsyncClientConnection(self, self.async_client)
        async with connection:
            yield connection.client

    async def restart_async_client(self, force=False) -> None:
        """
        Restart the async client if it has not been used in the last client_timeout minutes (set in init).
        """
        if self.client_timeout == datetime.timedelta(minutes=0) and not force:
            return

        # First check if the client has been used in the last X minutes
        if (
            datetime.datetime.now() - self.last_used_async_client > self.client_timeout
            or force
        ):
            # Acquire lock before modifying shared state to prevent race conditions
            try:
                async with self.async_lock:
                    # Clear the event WHILE holding the lock to prevent new connections from starting
                    # This ensures no new connections can proceed until we're done
                    self.async_restart_event.clear()

                    # Record current counter value before waiting
                    last_recorded_counter = self.async_in_use_counter

                    # Set reasonable timeout values
                    time_spent = 0
                    max_wait_time = 10  # seconds
                    check_interval = 0.1  # seconds

                    # Only wait if there are active connections
                    if last_recorded_counter > 0:
                        # Wait for existing connections to complete
                        while self.async_in_use_counter > 0:
                            # Release lock during sleep to prevent deadlock
                            self.async_lock.release()

                            # Only timeout after 10 seconds if nothing is happening
                            # if the counter is changing, then things are happening and we should wait
                            if self.async_in_use_counter != last_recorded_counter:
                                last_recorded_counter = self.async_in_use_counter
                                time_spent = 0

                            try:
                                await asyncio.sleep(check_interval)
                                time_spent += check_interval
                                if time_spent > max_wait_time:
                                    if self.logger:
                                        self.logger.error(
                                            f"Async client restart timed out after {max_wait_time} seconds. "
                                        )
                                    break
                            finally:
                                # Re-acquire lock after sleep
                                await self.async_lock.acquire()

                    # Handle timeout case - must reset state regardless of timeout
                    if self.async_in_use_counter > 0:
                        if self.logger:
                            self.logger.error(
                                "Force resetting async client state due to timeout"
                            )
                        self.async_in_use_counter = 0

                    # Whether we timed out or not, we need to restart the client
                    try:
                        # Only close if client exists and is connected
                        if (
                            hasattr(self, "async_client")
                            and self.async_client is not None
                        ):
                            await self.async_client.close()
                        await asyncio.sleep(0.1)
                        # Create a new client instance
                        self.async_client = await self.get_async_client()
                    except Exception as e:
                        if self.logger:
                            self.logger.error(
                                f"Error during async client restart: {str(e)}"
                            )
                        # Create a new client anyway to ensure we have a valid client
                        self.async_client = await self.get_async_client()
                    finally:
                        # CRITICAL: Always set the event to prevent deadlocks
                        # This ensures waiting connections can proceed
                        self.async_restart_event.set()

            except Exception as e:
                if self.logger:
                    self.logger.error(
                        f"Unexpected error in async client restart: {str(e)}"
                    )
                # Ensure the event is set in all error cases
                self.async_restart_event.set()
                # Attempt to create a new client
                self.async_client = await self.get_async_client()

    async def restart_client(self, force=False) -> None:
        """
        Restart the sync client if it has not been used in the last client_timeout minutes (set in init).
        """
        if self.client_timeout == datetime.timedelta(minutes=0) and not force:
            return

        # First check if the client has been used in the last X minutes
        if (
            datetime.datetime.now() - self.last_used_sync_client > self.client_timeout
            or force
        ):
            # Use both locks to prevent any race conditions between sync and async operations
            try:
                # Acquire sync lock first
                with self.sync_lock:
                    # Clear event while holding the lock to prevent race conditions
                    self.sync_restart_event.clear()

                    # Record current counter value
                    last_recorded_counter = self.sync_in_use_counter

                    # Set reasonable timeout values
                    time_spent = 0
                    max_wait_time = 10  # seconds
                    check_interval = 0.1  # seconds

                    # Only wait if there are active connections
                    if last_recorded_counter > 0:
                        # Wait for existing connections to complete
                        while self.sync_in_use_counter > 0:
                            # Must release lock during async sleep to prevent deadlock
                            self.sync_lock.release()

                            # Only timeout after 10 seconds if nothing is happening
                            # if the counter is changing, then things are happening and we should wait
                            if self.sync_in_use_counter != last_recorded_counter:
                                last_recorded_counter = self.sync_in_use_counter
                                time_spent = 0

                            try:
                                await asyncio.sleep(check_interval)
                                time_spent += check_interval
                                if time_spent > max_wait_time:
                                    if self.logger:
                                        self.logger.error(
                                            f"Sync client restart timed out after {max_wait_time}s. "
                                            f"Initial counter: {last_recorded_counter}, Current: {self.sync_in_use_counter}"
                                        )
                                    break
                            finally:
                                # Re-acquire lock
                                self.sync_lock.acquire()

                    # Handle timeout case - must reset state regardless of timeout
                    if self.sync_in_use_counter > 0:
                        if self.logger:
                            self.logger.error(
                                "Force resetting sync client state due to timeout"
                            )
                        self.sync_in_use_counter = 0

                    # Whether we timed out or not, we need to restart the client
                    try:
                        # Only close if client exists and is connected
                        if hasattr(self, "client") and self.client is not None:
                            self.client.close()
                        await asyncio.sleep(0.1)
                        # Create a new client instance
                        self.client = self.get_client()
                    except Exception as e:
                        if self.logger:
                            self.logger.error(
                                f"Error during sync client restart: {str(e)}"
                            )
                        # Create a new client anyway
                        self.client = self.get_client()
                    finally:
                        # CRITICAL: Always set the event to prevent deadlocks
                        self.sync_restart_event.set()

            except Exception as e:
                if self.logger:
                    self.logger.error(
                        f"Unexpected error in sync client restart: {str(e)}"
                    )
                # Ensure the event is set in all error cases
                self.sync_restart_event.set()
                # Attempt to create a new client
                self.client = self.get_client()

    async def close_clients(self) -> None:
        """
        Close both the async and sync clients.
        Should not be called inside a Tool or other function inside the decision tree.
        """
        if hasattr(self, "async_client") and self.async_client is not None:
            await self.async_client.close()
        if hasattr(self, "client") and self.client is not None:
            self.client.close()


# Custom context managers so that clients do not close after use (instead on a timer)
class _ClientConnection:
    def __init__(self, manager: ClientManager, client: WeaviateClient):
        self.manager = manager
        self.client = client

    def __enter__(self) -> WeaviateClient:
        return self.client

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        with self.manager.sync_lock:
            self.manager.sync_in_use_counter -= 1
        self.manager.update_last_used_sync_client()


class _AsyncClientConnection:
    def __init__(self, manager: ClientManager, client: WeaviateAsyncClient):
        self.manager = manager
        self.client = client

    async def __aenter__(self) -> WeaviateAsyncClient:
        return self.client

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        async with self.manager.async_lock:
            self.manager.async_in_use_counter -= 1
        self.manager.update_last_used_async_client()