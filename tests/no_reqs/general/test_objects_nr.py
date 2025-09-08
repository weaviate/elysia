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


def test_environment():
    environment = Environment()

    assert environment.is_empty()

    res = Result(
        objects=[{"a": 1, "b": 2}],
        metadata={"metadata_a": "hello", "metadata_b": 10},
        name="test_result",
    )
    environment.add("test_tool", res)

    assert not environment.is_empty()
    assert (
        environment.environment["test_tool"]["test_result"][0]["objects"][0]["a"] == 1
    )
    assert (
        environment.environment["test_tool"]["test_result"][0]["objects"][0]["b"] == 2
    )

    assert environment.environment["test_tool"]["test_result"][0]["metadata"] == {
        "metadata_a": "hello",
        "metadata_b": 10,
    }

    found_item = environment.find("test_tool", "test_result")
    assert found_item[0]["objects"][0]["a"] == 1
    assert found_item[0]["objects"][0]["b"] == 2
    assert found_item[0]["metadata"] == {
        "metadata_a": "hello",
        "metadata_b": 10,
    }

    found_item_index = environment.find("test_tool", "test_result", 0)
    assert found_item_index["objects"][0]["a"] == 1
    assert found_item_index["objects"][0]["b"] == 2
    assert found_item_index["metadata"] == {
        "metadata_a": "hello",
        "metadata_b": 10,
    }

    environment.replace("test_tool", "test_result", [{"c": 3, "d": 4}])
    assert (
        environment.environment["test_tool"]["test_result"][0]["objects"][0]["c"] == 3
    )
    assert (
        environment.environment["test_tool"]["test_result"][0]["objects"][0]["d"] == 4
    )

    environment.replace("test_tool", "test_result", [{"e": 5, "f": "hello"}], index=0)
    assert (
        environment.environment["test_tool"]["test_result"][0]["objects"][0]["e"] == 5
    )
    assert (
        environment.environment["test_tool"]["test_result"][0]["objects"][0]["f"]
        == "hello"
    )

    json_obj = environment.environment
    assert "test_tool" in json_obj
    assert "test_result" in json_obj["test_tool"]
    assert json_obj["test_tool"]["test_result"][0]["objects"][0]["e"] == 5
    assert json_obj["test_tool"]["test_result"][0]["objects"][0]["f"] == "hello"

    environment.remove("test_tool", "test_result")
    assert environment.is_empty()

    environment.add_objects(
        "test_tool2",
        "test_result2",
        [{"a": 1, "b": 2}],
        metadata={"item_number": "first"},
    )
    assert "test_tool2" in environment.environment
    assert "test_result2" in environment.environment["test_tool2"]
    assert (
        environment.environment["test_tool2"]["test_result2"][0]["objects"][0]["a"] == 1
    )
    assert (
        environment.environment["test_tool2"]["test_result2"][0]["objects"][0]["b"] == 2
    )
    assert (
        environment.environment["test_tool2"]["test_result2"][0]["metadata"][
            "item_number"
        ]
        == "first"
    )

    environment.add_objects(
        "test_tool2",
        "test_result2",
        [{"c": 1, "d": 2}],
        metadata={"item_number": "second"},
    )
    assert (
        environment.environment["test_tool2"]["test_result2"][1]["objects"][0]["c"] == 1
    )
    assert (
        environment.environment["test_tool2"]["test_result2"][1]["objects"][0]["d"] == 2
    )
    assert (
        environment.environment["test_tool2"]["test_result2"][1]["metadata"][
            "item_number"
        ]
        == "second"
    )

    assert (
        environment.environment["test_tool2"]["test_result2"][0]["objects"][0]["a"] == 1
    )
    assert (
        environment.environment["test_tool2"]["test_result2"][0]["objects"][0]["b"] == 2
    )

    environment.remove("test_tool2", "test_result2", 1)
    assert environment.environment["test_tool2"]["test_result2"] != []
    assert (
        environment.environment["test_tool2"]["test_result2"][0]["objects"][0]["a"] == 1
    )
    assert (
        environment.environment["test_tool2"]["test_result2"][0]["objects"][0]["b"] == 2
    )
    assert (
        environment.environment["test_tool2"]["test_result2"][0]["metadata"][
            "item_number"
        ]
        == "first"
    )
    assert len(environment.environment["test_tool2"]["test_result2"]) == 1


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
