import asyncio
import pytest
from elysia.util.dummy_adapter import DummyAdapter
from dspy import configure, ChatAdapter, LM
import dspy


@pytest.fixture(scope="session")
def event_loop():
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
    try:
        yield loop
    finally:
        loop.close()


@pytest.fixture(autouse=True, scope="module")
def use_dummy_adapter():
    dummy_adapter = DummyAdapter()
    configure(adapter=dummy_adapter)

    yield

    configure(adapter=ChatAdapter())
