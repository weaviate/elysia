from weaviate.classes.config import Configure, Property, DataType
from elysia.util.client import ClientManager


async def create_tree_collection(client):
    await client.collections.create(
        "ELYSIA_TREES__",
        vectorizer_config=Configure.Vectorizer.none(),
        inverted_index_config=Configure.inverted_index(index_timestamps=True),
        properties=[
            Property(
                name="user_id",
                data_type=DataType.TEXT,
            ),
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


async def create_feedback_collection(client):
    await client.collections.create(
        "ELYSIA_FEEDBACK__",
        properties=[
            # session data
            Property(
                name="user_id",
                data_type=DataType.TEXT,
            ),
            Property(
                name="conversation_id",
                data_type=DataType.TEXT,
            ),
            Property(
                name="query_id",
                data_type=DataType.TEXT,
            ),
            # feedback value (between -2, -1, 1, 2)
            Property(
                name="feedback",
                data_type=DataType.NUMBER,
            ),
            # track which models were used
            Property(
                name="modules_used",
                data_type=DataType.TEXT_ARRAY,
            ),
            # Tree data (except available_information)
            Property(
                name="user_prompt",
                data_type=DataType.TEXT,
            ),
            Property(
                name="conversation_history",
                data_type=DataType.OBJECT_ARRAY,
                nested_properties=[
                    Property(
                        name="role",
                        data_type=DataType.TEXT,
                    ),
                    Property(
                        name="content",
                        data_type=DataType.TEXT,
                    ),
                ],
            ),
            Property(
                name="tasks_completed",
                data_type=DataType.OBJECT_ARRAY,
                nested_properties=[
                    Property(
                        name="prompt",
                        data_type=DataType.TEXT,
                    ),
                    Property(
                        name="tasks",
                        data_type=DataType.OBJECT_ARRAY,
                        nested_properties=[
                            Property(
                                name="task",
                                data_type=DataType.TEXT,
                            ),
                            Property(
                                name="reasoning",
                                data_type=DataType.TEXT,
                            ),
                            Property(
                                name="todo",
                                data_type=DataType.TEXT,
                            ),
                            Property(name="action", data_type=DataType.BOOL),
                            Property(
                                name="count",
                                data_type=DataType.NUMBER,
                            ),
                            Property(
                                name="extra_string",
                                data_type=DataType.TEXT,
                            ),
                        ],
                    ),
                ],
            ),
            # Extra specifics
            Property(
                name="route",
                data_type=DataType.TEXT_ARRAY,
            ),
            Property(
                name="action_information",
                data_type=DataType.OBJECT_ARRAY,
                nested_properties=[
                    Property(
                        name="collection_name",
                        data_type=DataType.TEXT,
                    ),
                    Property(
                        name="action_name",
                        data_type=DataType.TEXT,
                    ),
                    Property(
                        name="code",
                        data_type=DataType.OBJECT,
                        nested_properties=[
                            Property(
                                name="title",
                                data_type=DataType.TEXT,
                            ),
                            Property(
                                name="language",
                                data_type=DataType.TEXT,
                            ),
                            Property(
                                name="text",
                                data_type=DataType.TEXT,
                            ),
                        ],
                    ),
                    Property(
                        name="return_type",
                        data_type=DataType.TEXT,
                    ),
                    Property(
                        name="output_type",
                        data_type=DataType.TEXT,
                    ),
                ],
            ),
            # metadata
            Property(
                name="time_taken_seconds",
                data_type=DataType.NUMBER,
            ),
            Property(
                name="base_lm_used",
                data_type=DataType.TEXT,
            ),
            Property(
                name="complex_lm_used",
                data_type=DataType.TEXT,
            ),
            Property(
                name="feedback_date",
                data_type=DataType.DATE,
            ),
            Property(
                name="decision_time",
                data_type=DataType.NUMBER,
            ),
            # dump training_updates as string
            Property(name="training_updates", data_type=DataType.TEXT),
        ],
        vectorizer_config=[
            Configure.NamedVectors.text2vec_openai(
                name="user_prompt",
                model="text-embedding-3-small",
                source_properties=["user_prompt"],
                vector_index_config=Configure.VectorIndex.hnsw(
                    quantizer=Configure.VectorIndex.Quantizer.sq(),
                ),
            ),
        ],
    )


async def create_config_collection(client):
    await client.collections.create(
        "ELYSIA_CONFIG__",
        vectorizer_config=Configure.Vectorizer.none(),
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
                name="user_id",
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
    )


import json
from weaviate.util import generate_uuid5
from elysia.tree.tree import Tree


async def export_to_weaviate(
    tree: Tree, collection_name: str, client_manager: ClientManager | None = None
) -> None:
    """
    Export the tree to a Weaviate collection.

    Args:
        collection_name (str): The name of the collection to export to.
        client_manager (ClientManager): The client manager to use.
            If not provided, a new ClientManager will be created from environment variables.
    """
    if client_manager is None:
        client_manager = ClientManager()
        close_after_use = True
    else:
        close_after_use = False

    async with client_manager.connect_to_async_client() as client:

        collection = client.collections.get(collection_name)

        json_data_str = json.dumps(tree.export_to_json())

        uuid = generate_uuid5(tree.conversation_id)

        if await collection.data.exists(uuid):
            await collection.data.update(
                uuid=uuid,
                properties={
                    "user_id": tree.user_id,
                    "conversation_id": tree.conversation_id,
                    "tree": json_data_str,
                    "title": tree.conversation_title,
                },
            )
            tree.settings.logger.info(
                f"Successfully updated existing tree in collection '{collection_name}' with id '{tree.conversation_id}'"
            )
        else:
            await collection.data.insert(
                uuid=uuid,
                properties={
                    "user_id": tree.user_id,
                    "conversation_id": tree.conversation_id,
                    "tree": json_data_str,
                    "title": tree.conversation_title,
                },
            )
            tree.settings.logger.info(
                f"Successfully inserted new tree in collection '{collection_name}' with id '{tree.conversation_id}'"
            )

    if close_after_use:
        await client_manager.close_clients()


from datetime import datetime
from elysia.util.parsing import format_datetime


async def create_feedback(
    user_id: str, conversation_id: str, query_id: str, feedback: int, tree: Tree, client
):
    feedback_collection = client.collections.get("ELYSIA_FEEDBACK__")

    history = tree.history[query_id]

    # ensure "action" is a bool in tasks_completed
    for task_prompt in history["tree_data"].tasks_completed:
        for task in task_prompt["task"]:
            task["action"] = bool(task["action"])

    date_now = datetime.now()

    properties = {
        "user_id": user_id,
        "conversation_id": conversation_id,
        "query_id": query_id,
        "feedback": int(feedback),
        "modules_used": list(
            set([h["module_name"] for h in history["training_updates"]])
        ),
        "user_prompt": history["tree_data"].user_prompt,
        "conversation_history": history["tree_data"].conversation_history,
        "tasks_completed": history["tree_data"].tasks_completed,
        "route": history["decision_history"],
        "action_information": history["action_information"],
        "time_taken_seconds": history["time_taken_seconds"],
        "decision_time": tree.tracker.get_average_time("decision_node"),
        "base_lm_used": tree.base_lm.model,
        "complex_lm_used": tree.complex_lm.model,
        "feedback_datetime": format_datetime(date_now),
        "feedback_date": format_datetime(
            date_now.replace(hour=0, minute=0, second=0, microsecond=0)
        ),
        "training_updates": json.dumps(history["training_updates"]),
        "initialisation": history["initialisation"],
    }

    # uuid is generated based on the user_id, conversation_id, query_id ONLY
    # so if the user re-selects the same feedback, it will be updated instead of added
    session_uuid = generate_uuid5(
        {
            "user_id": user_id,
            "conversation_id": conversation_id,
            "query_id": query_id,
        }
    )

    if await feedback_collection.data.exists(session_uuid):
        await feedback_collection.data.update(properties=properties, uuid=session_uuid)
    else:
        await feedback_collection.data.insert(properties=properties, uuid=session_uuid)


from elysia.util.parsing import format_dict_to_serialisable
from weaviate.classes.query import Filter


async def create_config(
    user_id: str,
    config_id: str,
    client,
    name: str,
    style: str,
    agent_description: str,
    end_goal: str,
    branch_initialisation: str,
    default: bool,
):

    uuid = generate_uuid5(config_id)
    collection = client.collections.get("ELYSIA_CONFIG__")

    config_item = {
        "name": name,
        "settings": {
            "API_KEYS": {"null": "null"},
            "base_model": "gpt-4o-mini",
            "complex_model": "gpt-4o",
            "base_provider": "openai",
            "complex_provider": "openai",
            "model_api_base": None,
            "wcd_url": None,
            "wcd_api_key": None,
            "weaviate_is_local": False,
            "local_weaviate_port": 8080,
            "local_weaviate_grpc_port": 50051,
        },
        "style": style,
        "agent_description": agent_description,
        "end_goal": end_goal,
        "branch_initialisation": branch_initialisation,
        "frontend_config": {"save_configs_to_weaviate": True},
        "config_id": config_id,
        "user_id": user_id,
        "default": default,
    }

    # if the config is a default config, set all other default configs to False
    if default:
        existing_default_config = await collection.query.fetch_objects(
            filters=Filter.all_of(
                [
                    Filter.by_property("default").equal(True),
                    Filter.by_property("user_id").equal(user_id),
                ]
            )
        )
        for item in existing_default_config.objects:
            await collection.data.update(properties={"default": False}, uuid=item.uuid)

    # save the config to the weaviate database
    if await collection.data.exists(uuid=uuid):
        await collection.data.update(properties=config_item, uuid=uuid)
    else:
        await collection.data.insert(config_item, uuid=uuid)


async def main(wcd_url: str, wcd_api_key: str):
    from uuid import uuid4

    client_manager = ClientManager(wcd_url=wcd_url, wcd_api_key=wcd_api_key)

    # delete existing collections
    async with client_manager.connect_to_async_client() as client:
        if await client.collections.exists("ELYSIA_TREES__"):
            await client.collections.delete("ELYSIA_TREES__")
        if await client.collections.exists("ELYSIA_FEEDBACK__"):
            await client.collections.delete("ELYSIA_FEEDBACK__")
        if await client.collections.exists("ELYSIA_CONFIG__"):
            await client.collections.delete("ELYSIA_CONFIG__")
        if await client.collections.exists("ELYSIA_VERSION__"):
            await client.collections.delete("ELYSIA_VERSION__")

    # create the collections
    async with client_manager.connect_to_async_client() as client:
        await create_tree_collection(client)
        await create_feedback_collection(client)
        await create_config_collection(client)

    # do some stuff to add data to the collections
    from elysia import Tree, Settings, tool

    settings = Settings()
    settings.smart_setup()
    settings.configure(logging_level="ERROR")

    print("\n" * 3)
    print("Running example conversations with tree (1/2)...")
    print("\n" * 3)

    tree = Tree(settings=settings)
    tree("Hi!")

    # add feedback
    print("\n" * 3)
    print("Adding feedback to tree (1/2)...")
    print("\n" * 3)
    async with client_manager.connect_to_async_client() as client:
        await create_feedback(
            tree.user_id,
            tree.conversation_id,
            list(tree.query_id_to_prompt.keys())[0],
            1,
            tree,
            client,
        )

    # save the tree
    print("\n" * 3)
    print("Saving tree (1/2) to Weaviate...")
    print("\n" * 3)
    await export_to_weaviate(tree, "ELYSIA_TREES__", client_manager)

    print("\n" * 3)
    print("Running example conversations with tree (2/2)...")
    print("\n" * 3)
    # add another interaction
    tree = Tree(settings=settings)
    tree("Tell me what collections are available")

    print("\n" * 3)
    print("Adding feedback to tree (2/2)...")
    print("\n" * 3)
    async with client_manager.connect_to_async_client() as client:
        await create_feedback(
            tree.user_id,
            tree.conversation_id,
            list(tree.query_id_to_prompt.keys())[0],
            0,
            tree,
            client,
        )

    # save the tree
    print("\n" * 3)
    print("Saving tree (2/2) to Weaviate...")
    print("\n" * 3)
    await export_to_weaviate(tree, "ELYSIA_TREES__", client_manager)

    print("\n" * 3)
    print("Saving some configs...")
    print("\n" * 3)
    # save some configs
    async with client_manager.connect_to_async_client() as client:
        print("\n" * 3)
        print("Creating config 1...")
        print("\n" * 3)
        await create_config(
            user_id=tree.user_id,
            config_id=str(uuid4()),
            client=client,
            name="Test Config 1",
            style="Informative, polite and friendly. 1",
            agent_description="Test description 1",
            end_goal="Test end goal 1",
            branch_initialisation="one_branch",
            default=True,
        )
        print("\n" * 3)
        print("Creating config 2...")
        print("\n" * 3)
        await create_config(
            user_id=tree.user_id,
            config_id=str(uuid4()),
            client=client,
            name="Test Config 2",
            style="Informative, polite and friendly. 2",
            agent_description="Test description 2",
            end_goal="Test end goal 2",
            branch_initialisation="one_branch",
            default=False,
        )
        print("\n" * 3)
        print("Creating config 3...")
        print("\n" * 3)
        await create_config(
            user_id=tree.user_id,
            config_id=str(uuid4()),
            client=client,
            name="Test Config 3",
            style="Informative, polite and friendly. 3",
            agent_description="Test description 3",
            end_goal="Test end goal 3",
            branch_initialisation="one_branch",
            default=False,
        )

    print("\n" * 3)
    print("Success!!")
    print(
        "Collections and data are now on the v0.2 version. This should hopefully match what users who come from v0.2 will see."
    )


if __name__ == "__main__":
    import asyncio
    import argparse

    parser = argparse.ArgumentParser(description="Migration test script")
    parser.add_argument("--api-key", required=True, help="Weaviate API key")
    parser.add_argument("--wcd-url", required=True, help="Weaviate cluster URL")
    args = parser.parse_args()

    asyncio.run(main(args.wcd_url, args.api_key))
