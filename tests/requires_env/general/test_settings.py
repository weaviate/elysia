import os
import pytest
from elysia.config import Settings, configure

from elysia.config import settings as global_settings
from elysia.config import reset_settings

from elysia.tree import Tree
from elysia.config import APIKeyError


def test_smart_setup():
    """
    Test that the smart setup is correct
    """
    settings = Settings.from_smart_setup()
    assert settings.BASE_MODEL is not None
    assert settings.COMPLEX_MODEL is not None
    assert settings.BASE_PROVIDER is not None
    assert settings.COMPLEX_PROVIDER is not None


def test_model_keys():

    # use settings from smart setup
    settings = Settings()
    tree = Tree(settings=settings)

    # set the keys to wrong
    settings.configure(
        openrouter_api_key="wrong",
        base_model="gemini-2.0-flash-001",
        base_provider="openrouter/google",
        complex_model="gemini-2.0-flash-001",
        complex_provider="openrouter/google",
    )

    # should error
    with pytest.raises(APIKeyError):
        response, objects = tree("hi elly. use text response only")

    with pytest.raises(APIKeyError):
        tree.create_conversation_title()

    with pytest.raises(APIKeyError):
        tree.get_follow_up_suggestions()

    # missing keys
    settings = Settings()
    settings.configure(
        base_model="gemini-2.0-flash-001",
        base_provider="openrouter/google",
        complex_model="gemini-2.0-flash-001",
        complex_provider="openrouter/google",
    )

    tree = Tree(settings=settings)

    with pytest.raises(APIKeyError):
        response, objects = tree("hi elly. use text response only")

    with pytest.raises(APIKeyError):
        tree.create_conversation_title()

    with pytest.raises(APIKeyError):
        tree.get_follow_up_suggestions()

    # now set the keys back to the .env
    settings.configure(
        openrouter_api_key=os.getenv("OPENROUTER_API_KEY"),
    )

    # should not error
    response, objects = tree("hi elly. use text response only")
