import pytest
from elysia.api.dependencies.common import get_user_manager


@pytest.fixture(scope="session", autouse=True)
async def cleanup_clients(request):
    yield
    user_manager = get_user_manager()
    await user_manager.close_all_clients()
