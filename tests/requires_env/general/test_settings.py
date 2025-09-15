import os
import pytest
from elysia.config import Settings, configure

from elysia.config import settings as global_settings
from elysia.config import reset_settings

from elysia.tree import Tree


def test_smart_setup():
    """
    Test that the smart setup is correct
    """
    settings = Settings.from_smart_setup()
    assert settings.BASE_MODEL is not None
    assert settings.COMPLEX_MODEL is not None
    assert settings.BASE_PROVIDER is not None
    assert settings.COMPLEX_PROVIDER is not None
