import os

import pytest

from elysia.util.client import ClientManager
from elysia import Settings


@pytest.mark.asyncio
async def test_client_manager_starts():
    try:
        client_manager = ClientManager()
    finally:
        await client_manager.close_clients()


@pytest.mark.asyncio
async def test_client_manager_cloud_starts_with_env_vars():
    try:
        client_manager = ClientManager()
        assert client_manager.wcd_url == os.getenv("WCD_URL")
        assert client_manager.wcd_api_key == os.getenv("WCD_API_KEY")
    finally:
        await client_manager.close_clients()


@pytest.mark.asyncio
async def test_client_manager_cloud_starts_with_api_keys():
    try:
        client_manager = ClientManager()

        if "OPENAI_API_KEY" in os.environ:
            assert "X-OpenAI-Api-Key" in client_manager.headers
            assert client_manager.headers["X-OpenAI-Api-Key"] == os.getenv(
                "OPENAI_API_KEY"
            )
    finally:
        await client_manager.close_clients()


@pytest.mark.asyncio
async def test_client_manager_cloud_clients_are_connected():
    try:
        client_manager = ClientManager()
        await client_manager.start_clients()
        assert client_manager.client.is_ready()
    finally:
        await client_manager.close_clients()


@pytest.mark.asyncio
async def test_client_manager_cloud_sync_client_connects():
    try:
        client_manager = ClientManager()
        with client_manager.connect_to_client() as client:
            assert client_manager.sync_in_use_counter == 1
        assert client_manager.sync_in_use_counter == 0
    finally:
        await client_manager.close_clients()


@pytest.mark.asyncio
async def test_client_manager_cloud_async_client_connects():
    try:
        client_manager = ClientManager()
        async with client_manager.connect_to_async_client() as client:
            assert client_manager.async_in_use_counter == 1
            pass
        assert client_manager.async_in_use_counter == 0
    finally:
        await client_manager.close_clients()


@pytest.mark.asyncio
async def test_client_manager_custom_starts():
    try:
        if "WEAVIATE_REST_URL" not in os.environ:
            pytest.skip("WEAVIATE_REST_URL is not set")
        if "WEAVIATE_GRPC_URL" not in os.environ:
            pytest.skip("WEAVIATE_GRPC_URL is not set")
        if "WCD_API_KEY" not in os.environ:
            pytest.skip("WCD_API_KEY is not set")
        client_manager = ClientManager(
            weaviate_is_custom=True,
            custom_http_host=os.getenv("WEAVIATE_REST_URL"),
            custom_http_port=443,
            custom_http_secure=True,
            custom_grpc_host=os.getenv("WEAVIATE_GRPC_URL"),
            custom_grpc_port=443,
            custom_grpc_secure=True,
            wcd_api_key=os.getenv("WCD_API_KEY"),
        )
        await client_manager.start_clients()
        assert client_manager.client.is_ready()
    finally:
        await client_manager.close_clients()


@pytest.mark.asyncio
async def test_client_manager_custom_starts_from_configure():
    try:
        if "WEAVIATE_REST_URL" not in os.environ:
            pytest.skip("WEAVIATE_REST_URL is not set")
        if "WEAVIATE_GRPC_URL" not in os.environ:
            pytest.skip("WEAVIATE_GRPC_URL is not set")
        if "WCD_API_KEY" not in os.environ:
            pytest.skip("WCD_API_KEY is not set")
        settings = Settings()
        settings.configure(
            weaviate_is_custom=True,
            custom_http_host=os.getenv("WEAVIATE_REST_URL"),
            custom_http_port=443,
            custom_http_secure=True,
            custom_grpc_host=os.getenv("WEAVIATE_GRPC_URL"),
            custom_grpc_port=443,
            custom_grpc_secure=True,
            wcd_api_key=os.getenv("WCD_API_KEY"),
            logging_level="DEBUG",
        )
        client_manager = ClientManager(settings=settings)
        await client_manager.start_clients()
        assert client_manager.client.is_ready()
    finally:
        await client_manager.close_clients()
