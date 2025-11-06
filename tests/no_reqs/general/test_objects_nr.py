import asyncio
import os
import pytest
from typing import Any
from weaviate.classes.query import Filter, QueryReference

from elysia.tree.objects import Environment
from elysia.objects import (
    Completed,
    Response,
    Status,
    Text,
    Update,
    Warning,
    Result,
)
from elysia.tools.retrieval.chunk import AsyncCollectionChunker
from elysia.tools.retrieval.objects import (
    ConversationRetrieval,
    DocumentRetrieval,
    MessageRetrieval,
)
from elysia.util.client import ClientManager
from elysia.util.parsing import format_dict_to_serialisable


@pytest.mark.asyncio
async def test_basic_result():
    result = Result(
        objects=[{"a": 1, "b": 2, "c": 3}],
        metadata={"metadata_a": 1, "metadata_b": 2, "metadata_c": 3},
        payload_type="test_type",
    )
    result_json = result.to_json()

    assert result_json == [{"a": 1, "b": 2, "c": 3}], "Objects should be equal"

    frontend_result = await result.to_frontend("user_id", "conversation_id", "query_id")

    # all id vars should be present
    for key in ["user_id", "conversation_id", "query_id"]:
        assert key in frontend_result, f"'{key}' not found in the frontend result"

    # type and payload type should be correct
    assert frontend_result["type"] == "result"
    assert frontend_result["payload"]["type"] == "test_type"

    # objects should be correct (but there can be more objects in frontend result)
    for i, obj in enumerate(result_json):
        for key in obj:
            assert (
                key in frontend_result["payload"]["objects"][i]
            ), f"'{key}' not found in the frontend result"

    # metadata should be the same
    assert frontend_result["payload"]["metadata"] == result.metadata


@pytest.mark.asyncio
async def test_result_with_mapping():
    objects = [{"a": 1, "b": 2, "c": 3, "d": 4, "e": 5, "f": 6}]
    mapping = {"new_a": "a", "new_b": "b", "new_c": "c"}
    unmapped_keys = ["d", "e", "f"]

    result = Result(
        objects=objects, metadata={}, mapping=mapping, unmapped_keys=unmapped_keys
    )
    result_json = result.to_json(True)

    for i, obj in enumerate(objects):
        for mapping_key in mapping:
            assert mapping_key in result_json[i]

        for obj_key in obj:
            if obj_key not in unmapped_keys:
                assert obj_key not in result_json[i]

        for unmapped_key in unmapped_keys:
            assert unmapped_key in result_json[i]

    frontend_result = await result.to_frontend("user_id", "conversation_id", "query_id")

    for i, obj in enumerate(objects):
        for mapping_key in mapping:
            assert mapping_key in frontend_result["payload"]["objects"][i]


def test_environment_nr():
    environment = Environment()
    assert environment.is_empty()
    assert environment.is_empty(tool_name="any_Tool_Name")

    result_a = Result(
        objects=[{"a": 1, "b": 2}], metadata={"metadata_a": "hello", "metadata_b": 10}
    )
    result_b = Result(
        objects=[{"c": 3, "d": 4}], metadata={"metadata_c": "world", "metadata_d": 20}
    )

    environment.add("test_tool", result_a)
    environment.add("test_tool", result_b)

    assert not environment.is_empty()

    get = environment.get("test_tool")
    assert get is not None
    assert len(get) == 2
    assert get[0].objects == [{"a": 1, "b": 2}]
    assert get[0].metadata == {"metadata_a": "hello", "metadata_b": 10}
    assert get[1].objects == [{"c": 3, "d": 4}]
    assert get[1].metadata == {"metadata_c": "world", "metadata_d": 20}

    objects = environment.get_objects(
        "test_tool", metadata_key="metadata_a", metadata_value="hello"
    )
    assert objects is not None
    assert len(objects) == 1
    assert objects[0]["a"] == 1
    assert objects[0]["b"] == 2

    objects = environment.get_objects(
        "test_tool", metadata={"metadata_a": "hello", "metadata_b": 10}
    )
    assert objects is not None
    assert len(objects) == 1
    assert objects[0]["a"] == 1
    assert objects[0]["b"] == 2

    environment.remove("test_tool", metadata={"metadata_a": "hello", "metadata_b": 10})
    assert not environment.is_empty()

    get = environment.get("test_tool")
    assert get is not None
    assert len(get) == 2  # not empty, but the list is empty

    objects = environment.get_objects(
        "test_tool", metadata_key="metadata_a", metadata_value="hello"
    )
    assert objects is not None
    assert len(objects) == 0

    # other objects are unaffected
    objects = environment.get_objects(
        "test_tool", metadata_key="metadata_c", metadata_value="world"
    )
    assert objects is not None
    assert len(objects) == 1
    assert objects[0]["c"] == 3
    assert objects[0]["d"] == 4

    environment.replace(
        "test_tool",
        [{"other": "object", "e": 1234}],
        metadata_key="metadata_c",
        metadata_value="world",
    )

    objects = environment.get_objects(
        "test_tool", metadata_key="metadata_c", metadata_value="world"
    )
    assert objects is not None
    assert len(objects) == 1
    assert objects[0]["other"] == "object"
    assert objects[0]["e"] == 1234
    assert "c" not in objects[0]
    assert "d" not in objects[0]

    environment.replace(
        "test_tool",
        [{"another": "object", "f": 12345}],
        metadata={"metadata_c": "world", "metadata_d": 20},
    )
    objects = environment.get_objects(
        "test_tool", metadata={"metadata_c": "world", "metadata_d": 20}
    )
    assert objects is not None
    assert len(objects) == 1
    assert objects[0]["another"] == "object"
    assert objects[0]["f"] == 12345
    assert "c" not in objects[0]
    assert "d" not in objects[0]

    environment.remove("test_tool", metadata_key="metadata_c", metadata_value="world")
    assert environment.is_empty()

    environment.clear()
    assert environment.is_empty()


@pytest.mark.asyncio
async def test_updates():
    types = [
        Status,
        Warning,
    ]
    names = [
        "status",
        "warning",
        "error",
    ]
    for object_type, object_name in zip(types, names):
        result = object_type(
            "this is a test",
        )
        frontend_result = await result.to_frontend(
            "user_id", "conversation_id", "query_id"
        )
        assert frontend_result["type"] == object_name
        assert isinstance(frontend_result["payload"], dict)
