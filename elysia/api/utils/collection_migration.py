import json
from weaviate.client import WeaviateAsyncClient
from weaviate.classes.config import Property, DataType, Configure
from weaviate.classes.query import Filter
from weaviate.client import WeaviateAsyncClient
from weaviate.classes.aggregate import GroupByAggregate


async def check_elysia_version(client: WeaviateAsyncClient) -> float:

    if not await client.collections.exists("ELYSIA_VERSION__"):
        # check if no collections exist
        config_exists = await client.collections.exists("ELYSIA_CONFIG__")
        trees_exists = await client.collections.exists("ELYSIA_TREES__")
        feedback_exists = await client.collections.exists("ELYSIA_FEEDBACK__")

        # if no collections, user hasn't used elysia before, set to latest version
        if not config_exists and not trees_exists and not feedback_exists:
            await set_elysia_version(client, 0.3, _check_exists=False)
            return 0.3

        # if collections exist, user has used elysia before (without ELYSIA_VERSION__), set to old version
        else:
            await set_elysia_version(client, 0.2, _check_exists=False)
            return 0.2

    version_collection = client.collections.get("ELYSIA_VERSION__")
    version = await version_collection.query.fetch_objects(limit=1)
    if len(version.objects) == 0:
        await set_elysia_version(client, 0.2, _check_exists=False)
        return 0.2

    return version.objects[0].properties["version"]  # type: ignore


async def set_elysia_version(
    client: WeaviateAsyncClient,
    version: float,
    _check_exists: bool = False,
    _create_collection: bool = False,
) -> None:

    if _create_collection:
        collection = await client.collections.create(
            "ELYSIA_VERSION__",
            properties=[Property(name="version", data_type=DataType.NUMBER)],
        )
        await collection.data.insert(properties={"version": version})
        return

    if (
        not _create_collection
        and not _check_exists
        and not await client.collections.exists("ELYSIA_VERSION__")
    ):
        collection = await client.collections.create(
            "ELYSIA_VERSION__",
            properties=[Property(name="version", data_type=DataType.NUMBER)],
        )
        await collection.data.insert(properties={"version": version})
        return

    collection = client.collections.use("ELYSIA_VERSION__")

    # get existing version
    response = await collection.query.fetch_objects(limit=1)
    if len(response.objects) == 0:
        await collection.data.insert(properties={"version": version})
        return

    await collection.data.update(
        uuid=response.objects[0].uuid,
        properties={"version": version},
    )


# Need collections for
# ELYSIA_TREES__
# ELYSIA_FEEDBACK__
# ELYSIA_CONFIG__
# ELYSIA_TOOL_PRESETS__
#
async def create_config_collection(client, collection_name: str = "ELYSIA_CONFIG__"):
    await client.collections.create(
        collection_name,
        vector_config=Configure.Vectors.self_provided(),
        inverted_index_config=Configure.inverted_index(index_timestamps=True),
        properties=[
            Property(
                name="name",
                data_type=DataType.TEXT,
            ),
            Property(
                name="style",
                data_type=DataType.TEXT,
            ),
            Property(
                name="agent_description",
                data_type=DataType.TEXT,
            ),
            Property(
                name="end_goal",
                data_type=DataType.TEXT,
            ),
            Property(
                name="branch_initialisation",
                data_type=DataType.TEXT,
            ),
            Property(
                name="config_id",
                data_type=DataType.TEXT,
            ),
            Property(
                name="default",
                data_type=DataType.BOOL,
            ),
        ],
        multi_tenancy_config=Configure.multi_tenancy(
            enabled=True,
            auto_tenant_creation=True,
            auto_tenant_activation=True,
        ),
    )


async def create_tree_collection(client, collection_name: str = "ELYSIA_TREES__"):
    await client.collections.create(
        collection_name,
        vector_config=Configure.Vectors.self_provided(),
        inverted_index_config=Configure.inverted_index(
            index_timestamps=True,
            index_null_state=True,
        ),
        multi_tenancy_config=Configure.multi_tenancy(
            enabled=True,
            auto_tenant_creation=True,
            auto_tenant_activation=True,
        ),
        properties=[
            Property(
                name="conversation_id",
                data_type=DataType.TEXT,
            ),
            Property(
                name="tree",
                data_type=DataType.TEXT,
            ),
            Property(
                name="title",
                data_type=DataType.TEXT,
            ),
        ],
    )


from elysia.util.feedback import create_feedback_collection


async def create_preset_collection(
    client, collection_name: str = "ELYSIA_TOOL_PRESETS__"
):
    await client.collections.create(
        collection_name,
        vector_config=Configure.Vectors.self_provided(),
        properties=[
            Property(name="preset_id", data_type=DataType.TEXT),
            Property(name="name", data_type=DataType.TEXT),
            Property(name="default", data_type=DataType.BOOL),
            Property(
                name="nodes",
                data_type=DataType.OBJECT_ARRAY,
                nested_properties=[
                    Property(name="instance_id", data_type=DataType.TEXT),
                    Property(name="name", data_type=DataType.TEXT),
                    Property(name="is_branch", data_type=DataType.BOOL),
                    Property(name="description", data_type=DataType.TEXT),
                    Property(name="instruction", data_type=DataType.TEXT),
                    Property(name="is_root", data_type=DataType.BOOL),
                ],
            ),
            Property(
                name="edges",
                data_type=DataType.OBJECT_ARRAY,
                nested_properties=[
                    Property(name="from", data_type=DataType.TEXT),
                    Property(name="to", data_type=DataType.TEXT),
                ],
            ),
        ],
        multi_tenancy_config=Configure.multi_tenancy(
            enabled=True,
            auto_tenant_creation=True,
            auto_tenant_activation=True,
        ),
    )


async def migrate_data_to_multi_tenancy(
    client: WeaviateAsyncClient,
    source_collection_name: str,
    target_collection_name: str,
):
    if source_collection_name == "ELYSIA_CONFIG__":
        await create_config_collection(client, target_collection_name)
    elif source_collection_name == "ELYSIA_TREES__":
        await create_tree_collection(client, target_collection_name)
    elif source_collection_name == "ELYSIA_TOOL_PRESETS__":
        await create_preset_collection(client, target_collection_name)
    elif source_collection_name == "ELYSIA_FEEDBACK__":
        await create_feedback_collection(client, target_collection_name)
    else:
        raise ValueError(f"Collection {source_collection_name} not supported")

    source_collection = client.collections.get(source_collection_name)
    target_collection = client.collections.get(target_collection_name)

    # get all user IDs in source collection
    user_id_resp = await source_collection.aggregate.over_all(
        group_by=GroupByAggregate(prop="user_id")
    )

    user_ids: list[str] = [group.grouped_by.value for group in user_id_resp.groups]  # type: ignore
    for user_id in user_ids:
        user_collection = target_collection.with_tenant(user_id)

        user_id_objects = await source_collection.query.fetch_objects(
            filters=Filter.by_property("user_id").equal(user_id)
        )
        for q in user_id_objects.objects:
            props = {k: v for k, v in q.properties.items() if k != "user_id"}

            if source_collection_name == "ELYSIA_FEEDBACK__":
                props["action_information"] = json.dumps(props["action_information"])

            await user_collection.data.insert(properties=props, uuid=q.uuid)


async def migrate_data_both_multi_tenancy(
    client: WeaviateAsyncClient,
    source_collection_name: str,
    target_collection_name: str,
):

    if target_collection_name == "ELYSIA_CONFIG__":
        await create_config_collection(client, target_collection_name)
    elif target_collection_name == "ELYSIA_TREES__":
        await create_tree_collection(client, target_collection_name)
    elif target_collection_name == "ELYSIA_TOOL_PRESETS__":
        await create_preset_collection(client, target_collection_name)
    elif target_collection_name == "ELYSIA_FEEDBACK__":
        await create_feedback_collection(client, target_collection_name)
    else:
        raise ValueError(f"Collection {target_collection_name} not supported")

    source_collection = client.collections.get(source_collection_name)
    target_collection = client.collections.get(target_collection_name)

    tenants = list((await source_collection.tenants.get()).keys())

    for tenant in tenants:
        source_tenant_collection = source_collection.with_tenant(tenant)
        target_tenant_collection = target_collection.with_tenant(tenant)
        user_id_objects = await source_tenant_collection.query.fetch_objects(
            limit=9999,
        )
        for q in user_id_objects.objects:
            props = {k: v for k, v in q.properties.items()}
            await target_tenant_collection.data.insert(properties=props, uuid=q.uuid)


async def reset_collections(client: WeaviateAsyncClient):
    if await client.collections.exists("ELYSIA_CONFIG__"):
        await client.collections.delete("ELYSIA_CONFIG__")
    if await client.collections.exists("ELYSIA_TREES__"):
        await client.collections.delete("ELYSIA_TREES__")
    if await client.collections.exists("ELYSIA_TOOL_PRESETS__"):
        await client.collections.delete("ELYSIA_TOOL_PRESETS__")
    if await client.collections.exists("ELYSIA_FEEDBACK__"):
        await client.collections.delete("ELYSIA_FEEDBACK__")
