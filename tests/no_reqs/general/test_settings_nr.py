import os
import pytest
from elysia.config import Settings, configure

from elysia.config import settings as global_settings
from elysia.config import reset_settings

from elysia.tree import Tree


def reset_global_settings():
    reset_settings()


def test_setup():
    """
    Test that settings works
    """
    settings = Settings()

    # LLM should not be set by default
    assert settings.BASE_MODEL is None
    assert settings.COMPLEX_MODEL is None
    assert settings.BASE_PROVIDER is None
    assert settings.COMPLEX_PROVIDER is None
    assert settings.MODEL_API_BASE is None

    reset_global_settings()


def test_from_env_vars():
    """
    Test that the config can be loaded from env vars
    """
    os.environ["BASE_MODEL"] = os.getenv("BASE_MODEL", "gemini-2.5-flash")
    os.environ["COMPLEX_MODEL"] = os.getenv("COMPLEX_MODEL", "gemini-2.5-flash")
    os.environ["BASE_PROVIDER"] = os.getenv("BASE_PROVIDER", "openrouter/google")
    os.environ["COMPLEX_PROVIDER"] = os.getenv("COMPLEX_PROVIDER", "openrouter/google")
    os.environ["MODEL_API_BASE"] = os.getenv("MODEL_API_BASE", "https://api.test.com")
    os.environ["WCD_URL"] = os.getenv("WCD_URL", "http://test_url.com")
    os.environ["WCD_API_KEY"] = os.getenv("WCD_API_KEY", "test_key")
    os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY", "test_key_openai")
    os.environ["ANTHROPIC_API_KEY"] = os.getenv(
        "ANTHROPIC_API_KEY", "test_key_anthropic"
    )

    settings = Settings.from_env_vars()
    assert settings.BASE_MODEL == os.environ["BASE_MODEL"]
    assert settings.COMPLEX_MODEL == os.environ["COMPLEX_MODEL"]
    assert settings.BASE_PROVIDER == os.environ["BASE_PROVIDER"]
    assert settings.COMPLEX_PROVIDER == os.environ["COMPLEX_PROVIDER"]
    assert settings.MODEL_API_BASE == os.environ["MODEL_API_BASE"]
    assert settings.WCD_URL == os.environ["WCD_URL"]
    assert settings.WCD_API_KEY == os.environ["WCD_API_KEY"]
    assert settings.API_KEYS["openai_api_key"] == os.environ["OPENAI_API_KEY"]
    assert settings.API_KEYS["anthropic_api_key"] == os.environ["ANTHROPIC_API_KEY"]

    reset_global_settings()


def test_basic_configure_global():
    """
    Does using elysia.configure() change the global settings (for backend)
    """

    configure(base_model="gpt-4o-mini", base_provider="openai")
    assert global_settings.BASE_MODEL == "gpt-4o-mini"

    configure(base_model="gpt-4o", base_provider="openai")
    assert global_settings.BASE_MODEL != "gpt-4o-mini"
    assert global_settings.BASE_MODEL == "gpt-4o"

    configure(complex_model="claude-3-5-haiku-latest", complex_provider="anthropic")
    assert global_settings.COMPLEX_MODEL == "claude-3-5-haiku-latest"
    assert global_settings.COMPLEX_PROVIDER == "anthropic"

    reset_global_settings()


def test_basic_configure_local():
    """
    Does using elysia.configure() change the local settings (for backend)
    """
    settings = Settings()

    settings.configure(
        complex_model="claude-3-5-haiku-latest", complex_provider="anthropic"
    )
    assert settings.COMPLEX_MODEL == "claude-3-5-haiku-latest"
    assert settings.COMPLEX_PROVIDER == "anthropic"
    assert settings.COMPLEX_MODEL != global_settings.COMPLEX_MODEL
    assert settings.COMPLEX_PROVIDER != global_settings.COMPLEX_PROVIDER

    settings.configure(base_model="gpt-4o-mini", base_provider="openai")
    assert settings.BASE_MODEL == "gpt-4o-mini"
    assert settings.BASE_PROVIDER == "openai"
    assert settings.BASE_MODEL != global_settings.BASE_MODEL
    assert settings.BASE_PROVIDER != global_settings.BASE_PROVIDER

    settings.configure(base_model="gpt-4o", base_provider="openai")
    assert settings.BASE_MODEL != "gpt-4o-mini"
    assert settings.BASE_MODEL == "gpt-4o"
    assert settings.BASE_PROVIDER == "openai"
    assert settings.BASE_MODEL != global_settings.BASE_MODEL
    assert settings.BASE_PROVIDER != global_settings.BASE_PROVIDER

    reset_global_settings()


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

    # explicitly set the adapter back to ChatAdapter
    from dspy.adapters import ChatAdapter
    import dspy

    with dspy.context(adapter=ChatAdapter()):
        # should error
        with pytest.raises(Exception):
            response, objects = tree("hi elly. use text response only")

        with pytest.raises(Exception):
            tree.create_conversation_title()

        with pytest.raises(Exception):
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

        with pytest.raises(Exception):
            response, objects = tree("hi elly. use text response only")

        with pytest.raises(Exception):
            tree.create_conversation_title()

        with pytest.raises(Exception):
            tree.get_follow_up_suggestions()

    # now set the keys back to the .env
    settings.configure(
        openrouter_api_key=os.getenv("OPENROUTER_API_KEY"),
    )

    # should not error
    response, objects = tree("hi elly. use text response only")
    tree.create_conversation_title()
    tree.get_follow_up_suggestions()
