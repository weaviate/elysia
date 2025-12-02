import pytest

from elysia import Tree, Settings
from elysia.util.client import ClientManager


async def create_green_pants_feedback(user_id):

    client_manager = ClientManager()

    # Check collection to see if the objects exist in Feedback for testing, to avoid duplicating
    async with client_manager.connect_to_async_client() as client:

        base_collection = client.collections.get("ELYSIA_FEEDBACK__")
        if not await base_collection.tenants.exists(user_id):
            need_to_create = True
        else:
            collection = base_collection.with_tenant(user_id)

            green_pants_response = await collection.query.bm25(
                "green pants", query_properties=["user_prompt"]
            )
            green_pants_objects = [o.properties for o in green_pants_response.objects]

            if len(green_pants_objects) < 2:
                need_to_create = True
            else:
                need_to_create = False

    if need_to_create:
        # Create two similar prompts + examples
        tree = Tree(user_id=user_id)
        tree("I am looking for green suede pants")
        # doesn't matter what the response is

        await tree.feedback(2)

        tree = Tree(user_id=user_id)
        tree("I need green pants")
        await tree.feedback(1)


@pytest.mark.asyncio
async def test_adding_retrieving_feedback():

    user_id = "test_use_feedback_879053589310"

    await create_green_pants_feedback(user_id)

    # create new tree with NUM_FEEDBACK_EXAMPLES=1 (required examples=1 so 2 will be enough)
    settings = Settings()
    settings.smart_setup()
    settings.configure(use_feedback=True, num_feedback_examples=1)

    tree = Tree(user_id=user_id, settings=settings)

    results = []
    async for result in tree.async_run("Green suede pants"):
        results.append(result)

    examples_found = False
    for res in results:
        if res["type"] == "fewshot_examples":
            examples_found = True
            assert len(res["payload"]["uuids"]) > 0

    assert examples_found


@pytest.mark.asyncio
async def test_adding_retrieving_feedback_with_view_environment():

    user_id = "test_use_feedback_879053589310"  # use same user ID to avoid making feedback twice

    await create_green_pants_feedback(user_id)

    # create new tree with NUM_FEEDBACK_EXAMPLES=1 (required examples=1 so 2 will be enough)
    settings = Settings()
    settings.smart_setup()
    settings.configure(use_feedback=True, num_feedback_examples=1)

    tree = Tree(user_id=user_id, settings=settings)

    # add random items to the environment
    tree.tree_data.environment.add_objects(
        "yap", [{"message": "Hello, world!"}] * 3000, {"id": "pure yap"}
    )

    # add a fake message to the environment
    tree.tree_data.environment.add_objects(
        "old_query_results",
        [
            {
                "description": "These bold red pants are perfect for chilling in.",
                "price": 29.99,
                "colour": "red",
            }
        ],
        {
            "search_parameters": "limit=1, sort=desc(timestamp), collection=Communications,query=pants"
        },
    )

    results = []
    async for result in tree.async_run("Green suede pants"):
        results.append(result)

    examples_found = False
    for res in results:
        if res["type"] == "fewshot_examples":
            examples_found = True
            assert len(res["payload"]["uuids"]) > 0

    assert examples_found
