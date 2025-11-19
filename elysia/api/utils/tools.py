from elysia.util.client import ClientManager
from elysia.api.api_types import ToolItem, BranchInfo, ToolPreset
import elysia.api.custom_tools as custom_tools
from elysia import Tool

from weaviate.classes.config import Configure, Property, DataType
from weaviate.util import generate_uuid5
from weaviate.classes.query import Filter

from typing import Dict, Type


def find_tool_classes() -> Dict[str, Type[Tool]]:
    """
    Find all Tool subclasses in the custom_tools module.

    Returns:
        (dict): Dictionary mapping class names to Tool subclass types
    """
    tool_classes = {}

    # Get all objects from the custom_tools module
    module_objects = dict[str, type[Tool]](
        [
            (name, cls)
            for name, cls in custom_tools.__dict__.items()
            if isinstance(cls, type)
        ]
    )

    # Filter for Tool subclasses (excluding the base Tool class)
    for cls in module_objects.values():
        if issubclass(cls, Tool) and cls.__name__ != "Tool":
            name = cls.get_metadata()["name"]
            tool_classes[name] = cls

    return tool_classes


def find_tool_metadata() -> Dict[str, Dict[str, str]]:
    """
    Find all tool metadata in the custom_tools module.

    Returns:
        (dict): Dictionary mapping tool names to tool metadata
    """
    tools = find_tool_classes()
    metadata = {}
    for name, cls in tools.items():
        metadata[name] = cls.get_metadata()
    return metadata


async def add_preset_weaviate(
    user_id: str,
    preset_id: str,
    name: str,
    order: list[ToolItem],
    branches: list[BranchInfo],
    default: bool,
    client_manager: ClientManager,
):
    async with client_manager.connect_to_async_client() as client:

        # check if collection exists
        if not await client.collections.exists(f"ELYSIA_TOOL_PRESETS__"):
            await client.collections.create(
                f"ELYSIA_TOOL_PRESETS__",
                vector_config=Configure.Vectors.self_provided(),
                properties=[
                    Property(name="preset_id", data_type=DataType.TEXT),
                    Property(name="name", data_type=DataType.TEXT),
                    Property(
                        name="order",
                        data_type=DataType.OBJECT_ARRAY,
                        nested_properties=[
                            Property(name="name", data_type=DataType.TEXT),
                            Property(name="from_branch", data_type=DataType.TEXT),
                            Property(name="from_tools", data_type=DataType.TEXT_ARRAY),
                            Property(name="is_branch", data_type=DataType.BOOL),
                        ],
                    ),
                    Property(
                        name="branches",
                        data_type=DataType.OBJECT_ARRAY,
                        nested_properties=[
                            Property(name="name", data_type=DataType.TEXT),
                            Property(name="description", data_type=DataType.TEXT),
                            Property(name="instruction", data_type=DataType.TEXT),
                        ],
                    ),
                    Property(name="default", data_type=DataType.BOOL),
                ],
                multi_tenancy_config=Configure.multi_tenancy(
                    enabled=True,
                    auto_tenant_creation=True,
                    auto_tenant_activation=True,
                ),
            )

        # get collection
        base_preset_collection = client.collections.get(f"ELYSIA_TOOL_PRESETS__")

        if not await base_preset_collection.tenants.exists(user_id):
            await base_preset_collection.tenants.create(user_id)

        preset_collection = base_preset_collection.with_tenant(user_id)

        # if setting default, need to update all other default presets to False
        if default:
            default_presets = await preset_collection.query.fetch_objects(
                filters=Filter.by_property("default").equal(True),
                limit=9999,
            )
            for default_preset in default_presets.objects:
                await preset_collection.data.update(
                    uuid=default_preset.uuid,
                    properties={"default": False},
                )

        # update or insert preset
        uuid = generate_uuid5(preset_id)
        if await preset_collection.data.exists(uuid):
            await preset_collection.data.update(
                uuid=uuid,
                properties={
                    "preset_id": preset_id,
                    "name": name,
                    "order": [item.model_dump() for item in order],
                    "branches": [item.model_dump() for item in branches],
                    "default": default,
                },
            )
        else:
            await preset_collection.data.insert(
                uuid=uuid,
                properties={
                    "preset_id": preset_id,
                    "name": name,
                    "order": [item.model_dump() for item in order],
                    "branches": [item.model_dump() for item in branches],
                    "default": default,
                },
            )


async def get_presets_weaviate(
    user_id: str,
    client_manager: ClientManager,
):
    async with client_manager.connect_to_async_client() as client:

        preset_collection = client.collections.get(f"ELYSIA_TOOL_PRESETS__")
        if await preset_collection.tenants.exists(user_id):
            preset_collection = preset_collection.with_tenant(user_id)

            presets = await preset_collection.query.fetch_objects(
                limit=9999,
            )
            presets = [
                ToolPreset.model_validate(preset.properties)
                for preset in presets.objects
            ]
        else:
            presets = []

    return presets


async def delete_preset_weaviate(
    user_id: str,
    preset_id: str,
    client_manager: ClientManager,
):
    async with client_manager.connect_to_async_client() as client:
        preset_collection = client.collections.get(
            f"ELYSIA_TOOL_PRESETS__"
        ).with_tenant(user_id)

        uuid = generate_uuid5(preset_id)
        if await preset_collection.data.exists(uuid):
            await preset_collection.data.delete_by_id(uuid)
