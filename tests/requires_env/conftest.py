import pytest
from elysia.util.client import ClientManager
from elysia.api.dependencies.common import get_user_manager
from weaviate.util import generate_uuid5
from weaviate.classes.query import Filter
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv(override=True)


def get_frontend_config_file_paths() -> Path:
    elysia_package_dir = Path(__file__).parent.parent.parent  # Gets to elysia/
    config_dir = elysia_package_dir / "elysia" / "api" / "user_configs"
    config_files = os.listdir(config_dir)
    return [
        f"{config_dir}/{c}"
        for c in config_files
        if c.startswith("frontend_config_test_")
    ]


def pytest_collection_modifyitems(config, items):
    """Skip tests if required environment variables are missing."""
    if (
        ("WCD_URL" not in os.environ and "WEAVIATE_URL" not in os.environ)
        or ("WCD_API_KEY" not in os.environ and "WEAVIATE_API_KEY" not in os.environ)
        or "OPENAI_API_KEY" not in os.environ
        or "OPENROUTER_API_KEY" not in os.environ
    ):
        skip_marker = pytest.mark.skip(
            reason="Missing required database environment variables"
        )
        for item in items:
            item.add_marker(skip_marker)


@pytest.fixture(scope="session", autouse=True)
def cleanup_configs(request):
    yield

    client_manager = ClientManager()

    if not client_manager.is_client:
        return

    # check for local frontend configs
    config_files = get_frontend_config_file_paths()
    for config_file in config_files:
        if os.path.exists(config_file):
            os.remove(config_file)

    # check for weaviate configs
    with client_manager.connect_to_client() as client:
        if client.collections.exists("ELYSIA_CONFIG__"):
            collection = client.collections.get("ELYSIA_CONFIG__")
            test_configs_response = collection.query.fetch_objects(
                limit=1000,
                filters=Filter.all_of(
                    [
                        Filter.by_property("user_id").like("test_*"),
                        Filter.by_property("config_id").like("test_*"),
                    ]
                ),
            )
            for config in test_configs_response.objects:
                collection.data.delete_by_id(config.uuid)

    client_manager.client.close()


@pytest.fixture(scope="session", autouse=True)
def cleanup_collections(request):
    yield

    client_manager = ClientManager()

    if not client_manager.is_client:
        return

    # check for weaviate configs
    with client_manager.connect_to_client() as client:
        for collection_name in client.collections.list_all():
            if collection_name.startswith("Test_ELYSIA_"):
                client.collections.delete(collection_name)
            elif collection_name.startswith("ELYSIA_Test_"):
                client.collections.delete(collection_name)

    client_manager.client.close()


@pytest.fixture(scope="session", autouse=True)
def cleanup_feedbacks(request):
    yield

    client_manager = ClientManager()

    if not client_manager.is_client:
        return

    # check for weaviate configs
    with client_manager.connect_to_client() as client:
        if client.collections.exists("ELYSIA_FEEDBACK__"):
            collection = client.collections.get("ELYSIA_FEEDBACK__")
            test_feedback_response = collection.query.fetch_objects(
                limit=1000,
                filters=Filter.by_property("user_id").like("test_*"),
            )
            for feedback in test_feedback_response.objects:
                collection.data.delete_by_id(feedback.uuid)

    client_manager.client.close()
