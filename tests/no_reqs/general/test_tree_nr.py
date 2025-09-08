import pytest
from elysia.config import Settings
from elysia.tree.tree import Tree
from elysia.util.client import ClientManager


def test_tree_init():
    tree = Tree()
    assert tree is not None


def test_tree_init_with_settings():
    settings = Settings.from_smart_setup()
    tree = Tree(settings=settings)
    assert tree is not None


def test_run_tree():
    settings = Settings()
    settings.configure(
        base_model="gpt-4o-mini",
        base_provider="openai",
        complex_model="gpt-4o",
        complex_provider="openai",
    )
    tree = Tree(settings=settings)
    response, objects = tree.run("Hi!")
    assert response is not None and len(response) > 0
    assert len(objects) is not None


@pytest.mark.asyncio
async def test_init_tree_with_no_client():
    settings = Settings()
    settings.configure(
        base_model="gpt-4o-mini",
        base_provider="openai",
        complex_model="gpt-4o",
        complex_provider="openai",
    )
    settings.WCD_URL = ""
    settings.WCD_API_KEY = ""

    tree = Tree(settings=settings)

    assert not (
        await tree.tools["query"].is_tool_available(
            tree_data=tree.tree_data,
            base_lm=tree.base_lm,
            complex_lm=tree.complex_lm,
            client_manager=ClientManager(
                wcd_url=settings.WCD_URL, wcd_api_key=settings.WCD_API_KEY
            ),
        )
    )
    assert not (
        await tree.tools["aggregate"].is_tool_available(
            tree_data=tree.tree_data,
            base_lm=tree.base_lm,
            complex_lm=tree.complex_lm,
            client_manager=ClientManager(
                wcd_url=settings.WCD_URL, wcd_api_key=settings.WCD_API_KEY
            ),
        )
    )
