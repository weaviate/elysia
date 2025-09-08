from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse

from elysia.api.api_types import (
    ViewPaginatedCollectionData,
    UpdateCollectionMetadataData,
)
from elysia.api.core.log import logger
from elysia.api.dependencies.common import get_user_manager
from elysia.api.services.user import UserManager
from elysia.util.parsing import format_dict_to_serialisable
from elysia.util.collection import (
    async_get_collection_data_types,
    paginated_collection,
)
from elysia.preprocessing.collection import (
    edit_preprocessed_collection_async,
    delete_preprocessed_collection_async,
    preprocessed_collection_exists_async,
)
from elysia.util.return_types import specific_return_types, types_dict, all_return_types
from elysia.util.client import ClientManager

from weaviate.classes.query import Filter


router = APIRouter()


@router.get("/mapping_types")
async def mapping_types():
    """
    Retrieve mapping types from hardcoded values.

    Returns:
        (JSONResponse): A JSON response containing the mapping types.
    """
    logger.debug(f"/mapping_types API request received")

    headers = {"Cache-Control": "no-cache"}
    try:
        return JSONResponse(
            content={
                "mapping_types": [
                    {
                        "name": return_type,
                        "description": all_return_types[return_type],
                        "fields": types_dict[return_type],
                    }
                    for return_type in all_return_types
                ],
                "error": "",
            },
            status_code=200,
            headers=headers,
        )
    except Exception as e:
        logger.exception(f"Error in /mapping_types API")
        return JSONResponse(
            content={"mapping_types": [], "error": str(e)},
            status_code=500,
            headers=headers,
        )


@router.get("/{user_id}/list")
async def collections_list(
    user_id: str, user_manager: UserManager = Depends(get_user_manager)
):
    """
    Retrieve a list of collections from the currently connected Weaviate cluster for the user.

    Args:
        user_id (str): The ID of the user to retrieve collections for.
        user_manager (UserManager): The user manager.

    Returns:
        (JSONResponse): A JSON response containing:
            - collections (list[dict]): A list of collections with their:
                - name (str): The name of the collection.
                - total (int): The total number of objects in the collection.
                - vectorizer (dict): The vectoriser configuration for the collection.
                - processed (bool): Whether the collection has been processed.
                - error (bool): Whether there is an error retrieving _this specific_ collection.
            - error (str): An overall error message if there is an error in the main function, otherwise an empty string.
    """
    logger.debug(f"/collections API request received")

    headers = {"Cache-Control": "no-cache"}

    try:

        user_local = await user_manager.get_user_local(user_id)
        client_manager: ClientManager = user_local["client_manager"]

        if not client_manager.is_client:
            return JSONResponse(
                content={
                    "collections": [],
                    "error": (
                        "Client manager is not connected. "
                        "This is not an error if the user does not want to use Weaviate."
                    ),
                },
                status_code=200,
                headers=headers,
            )

        async with client_manager.connect_to_async_client() as client:

            collections = [
                c
                for c in await client.collections.list_all()
                if not c.startswith("ELYSIA_")
            ]

            # get processed collections
            if await client.collections.exists("ELYSIA_METADATA__"):
                metadata_collection = client.collections.get("ELYSIA_METADATA__")
                processed_collections = await metadata_collection.query.fetch_objects(
                    filters=Filter.any_of(
                        [Filter.by_property("name").equal(c) for c in collections]
                    ),
                    limit=9999,
                )
                processed_collection_names = [
                    processed_collection.properties["name"]
                    for processed_collection in processed_collections.objects
                ]
                processed_collections_prompts = {
                    processed_collection.properties[
                        "name"
                    ]: processed_collection.properties["prompts"]
                    for processed_collection in processed_collections.objects
                }
            else:
                processed_collection_names = []
                processed_collections_prompts = {}

            # get collection metadata
            metadata = []
            for collection_name in collections:
                logger.debug(
                    f"Gathering information for collection_name: {collection_name}"
                )

                try:
                    # for vectoriser
                    collection = client.collections.get(collection_name)
                    config = await collection.config.get()

                    # for total count
                    all_aggregate = await collection.aggregate.over_all(
                        total_count=True
                    )
                    len = all_aggregate.total_count

                    # for processed
                    processed = collection_name in processed_collection_names

                    if processed:
                        prompts = processed_collections_prompts[collection_name]
                    else:
                        prompts = []

                    vector_config = {"fields": {}, "global": {}}
                    if config.vector_config:
                        for (
                            named_vector_name,
                            named_vector_config,
                        ) in config.vector_config.items():

                            # for this vectoriser, what model is being used?
                            model = named_vector_config.vectorizer.model
                            if isinstance(model, dict) and "model" in model:
                                model = model["model"]
                            else:
                                model = "Unknown"

                            # check what fields are being vectorised
                            fields = named_vector_config.vectorizer.source_properties
                            if fields is None:
                                fields = [
                                    c.name
                                    for c in config.properties
                                    if c.data_type[:].startswith("text")
                                ]

                            # for each field, add the vectoriser and model to the vector config
                            for field_name in fields:
                                if field_name not in vector_config["fields"]:
                                    vector_config["fields"][field_name] = [
                                        {
                                            "named_vector": named_vector_name,
                                            "vectorizer": named_vector_config.vectorizer.vectorizer.name,
                                            "model": model,
                                        }
                                    ]
                                else:
                                    vector_config["fields"][field_name].append(
                                        {
                                            "named_vector": named_vector_name,
                                            "vectorizer": named_vector_config.vectorizer.vectorizer.name,
                                            "model": model,
                                        }
                                    )

                    if config.vectorizer_config:
                        model = config.vectorizer_config.model
                        if isinstance(model, dict) and "model" in model:
                            model = model["model"]

                        vector_config["global"] = {
                            "vectorizer": config.vectorizer_config.vectorizer.name,
                            "model": model,
                        }
                    else:
                        vector_config["global"] = {}

                    metadata.append(
                        {
                            "name": collection_name,
                            "total": len,
                            "vectorizer": vector_config,
                            "processed": processed,
                            "error": False,
                            "prompts": prompts,
                        }
                    )
                except Exception as e:
                    metadata.append(
                        {
                            "name": collection_name,
                            "total": 0,
                            "vectorizer": {},
                            "processed": False,
                            "error": True,
                            "prompts": [],
                        }
                    )

            return JSONResponse(
                content={"collections": metadata, "error": ""},
                status_code=200,
                headers=headers,
            )

    except Exception as e:
        logger.exception(f"Error in /collections API")
        return JSONResponse(
            content={"collections": [], "error": str(e)},
            status_code=500,
            headers=headers,
        )


@router.post("/{user_id}/view/{collection_name}")
async def view_paginated_collection(
    user_id: str,
    collection_name: str,
    data: ViewPaginatedCollectionData,
    user_manager: UserManager = Depends(get_user_manager),
):
    """
    Find a list of objects in a collection, with pagination and filtering.

    Args:
        user_id (str): The ID of the user whose collections to view.
        collection_name (str): The name of the collection to view.
        data (ViewPaginatedCollectionData): The data for the request, containing:
            - page_size (int): The number of objects to return per page.
            - page_number (int): The page number to return.
            - query (str): The query to search for. If empty, all objects will be returned.
                If non-empty, BM25 will be used to search for objects.
            - sort_on (str): The property to sort on.
            - ascending (bool): Whether to sort in ascending or descending order.
            - filter_config (dict): The filter configuration, containing:
                - type (str): The type of filter to apply, one of 'all' or 'any'.
                - filters (list[dict]): The filters to apply, containing:
                    - field (str): The field to filter on.
                    - operator (str): The operator to apply,
                        one of 'equal', 'not_equal', 'greater_than', 'less_than', 'greater_than_or_equal',
                        'less_than_or_equal', 'contains', 'not_contains', 'is_empty', 'is_not_empty'.
                    - value (str): The value to filter on.
        user_manager (UserManager): The user manager.

    Returns:
        (JSONResponse): A JSON response containing:
            - properties (list[dict]): The properties of the collection.
            - items (list[dict]): The items in the collection.
    """

    logger.debug(f"/view_paginated_collection API request received")
    logger.debug(f"User ID: {user_id}")
    logger.debug(f"Query: {data.query}")
    logger.debug(f"Collection name: {collection_name}")
    logger.debug(f"Page size: {data.page_size}")
    logger.debug(f"Page number: {data.page_number}")
    logger.debug(f"Sort on: {data.sort_on}")
    logger.debug(f"Ascending: {data.ascending}")
    logger.debug(f"Filter config: {data.filter_config}")

    try:
        user_local = await user_manager.get_user_local(user_id)
        client_manager = user_local["client_manager"]
        async with client_manager.connect_to_async_client() as client:
            # get collection properties
            data_types = await async_get_collection_data_types(client, collection_name)

            # obtain paginated results from collection
            items = await paginated_collection(
                client=client,
                collection_name=collection_name,
                query=data.query,
                page_size=data.page_size,
                page_number=data.page_number,
                sort_on=data.sort_on,
                ascending=data.ascending,
                filter_config=data.filter_config,
            )

            logger.info(f"Returning collection info for {collection_name}")
            return JSONResponse(
                content={"properties": data_types, "items": items, "error": ""},
                status_code=200,
            )
    except Exception as e:
        logger.exception(f"Error in /view_paginated_collection API")
        return JSONResponse(
            content={"properties": [], "items": [], "error": str(e)}, status_code=500
        )


@router.get("/{user_id}/get_object/{collection_name}/{uuid}")
async def get_object(
    user_id: str,
    collection_name: str,
    uuid: str,
    user_manager: UserManager = Depends(get_user_manager),
):
    """
    Get a single object from a collection by its UUID.

    Args:
        user_id (str): The ID of the user whose object to retrieve.
        collection_name (str): The name of the collection to retrieve the object from.
        uuid (str): The UUID of the object to retrieve.
        user_manager (UserManager): The user manager.

    Returns:
        (JSONResponse): A JSON response containing:
            - properties (list[dict]): The data types of the object.
            - items (list[dict]): The object itself (one element list).
            - error (str): An error message if there is an error, otherwise an empty string.
    """
    logger.debug(f"/get_object API request received")
    logger.debug(f"User ID: {user_id}")
    logger.debug(f"Collection name: {collection_name}")
    logger.debug(f"UUID: {uuid}")

    headers = {"Cache-Control": "max-age=300"}  # cache for 5 minutes

    try:
        user_local = await user_manager.get_user_local(user_id)
        client_manager = user_local["client_manager"]
        async with client_manager.connect_to_async_client() as client:
            collection = client.collections.get(collection_name)

            try:
                object = await collection.query.fetch_object_by_id(uuid)
                object = object.properties
            except Exception as e:
                logger.error(
                    f"Error in /get_object API: No object found with UUID {uuid} in collection {collection_name}"
                )
                return JSONResponse(
                    content={
                        "properties": [],
                        "items": [],
                        "error": f"No object found with UUID {uuid} in collection {collection_name}.",
                    },
                    status_code=404,
                    headers=headers,
                )

            data_types = await async_get_collection_data_types(client, collection_name)
            format_dict_to_serialisable(object)

    except Exception as e:
        logger.exception(f"Error in /get_object API")
        return JSONResponse(
            content={"properties": [], "items": [], "error": str(e)},
            status_code=500,
            headers=headers,
        )

    return JSONResponse(
        content={"properties": data_types, "items": [object], "error": ""},
        status_code=200,
        headers=headers,
    )


@router.get("/{user_id}/metadata/{collection_name}")
async def collection_metadata(
    user_id: str,
    collection_name: str,
    user_manager: UserManager = Depends(get_user_manager),
):
    """
    Get the preprocessed metadata for a collection.
    This retrieves the output of the `.preprocess` function in Elysia.

    Args:
        user_id (str): The ID of the user whose metadata to retrieve.
        collection_name (str): The name of the collection to retrieve the metadata for.
        user_manager (UserManager): The user manager.

    Returns:
        (JSONResponse): A JSON response containing:
            - metadata (dict): The metadata for the collection.
            - error (str): An error message if there is an error, otherwise an empty string.

    Example metadata:
    ```json
    {
        "metadata": dict = {

            # summary statistics of each field in the collection
            "fields": list = [
                {
                    "name": str,
                    "description": str,
                    "range": list[float],
                    "type": str,
                    "groups": dict[str, str],
                    "mean": float
                },
                field_name_2: dict,
                ...
            ],

            # mapping_1, mapping_2 etc refer to frontend-specific types that the AI has deemed appropriate for this data
            # then the dict is to map the frontend fields to the data fields
            "mappings": dict = {
                mapping_1: dict,
                mapping_2: dict,
                ...,
            },

            # number of items in collection (float but just for consistency)
            "length": float,

            # AI generated summary of the dataset
            "summary": str,

            # name of collection
            "name": str,

            # what named vectors are available (if any)
            "named_vectors": list = [
                {
                    "name": str,
                    "vectorizer": str,
                    "model": str,
                    "enabled": bool,
                    "source_properties": list,
                    "description": str # defaults to empty
                },
                ...
            ],

            # vectoriser configuration
            "vectoriser": dict = {
                "vectorizer": str,
                "model": str,
            },

            # some config settings relevant for queries
            "index_properties": {
                "isNullIndexed": bool,
                "isLengthIndexed": bool,
                "isTimestampIndexed": bool,
            }
        }
        "error": ""
    }
    ```
    """
    logger.debug(f"/collection_metadata API request received")
    logger.debug(f"User ID: {user_id}")
    logger.debug(f"Collection name: {collection_name}")

    headers = {"Cache-Control": "no-cache"}

    try:
        user_local = await user_manager.get_user_local(user_id)
        client_manager = user_local["client_manager"]

        async with client_manager.connect_to_async_client() as client:
            metadata_name = f"ELYSIA_METADATA__"

            # check if the collection itself exists
            if not await client.collections.exists(collection_name):
                raise Exception(f"Collection {collection_name} does not exist")

            # check if the metadata collection exists
            if not await preprocessed_collection_exists_async(
                collection_name, client_manager
            ):
                raise Exception(
                    f"Metadata collection for {collection_name} does not exist"
                )

            metadata_collection = client.collections.get(metadata_name)
            metadata = await metadata_collection.query.fetch_objects(
                filters=Filter.by_property("name").equal(collection_name),
                limit=1,
            )
            properties = metadata.objects[0].properties
            format_dict_to_serialisable(properties)

            if properties["named_vectors"] is None:
                properties["named_vectors"] = []

    except Exception as e:
        logger.exception(f"Error in /collection_metadata API")
        return JSONResponse(
            content={
                "metadata": {},
                "error": str(e),
            },
            status_code=200,
            headers=headers,
        )

    logger.debug(f"\n\nProperties: {properties}\n\n")

    return JSONResponse(
        content={
            "metadata": properties,
            "error": "",
        },
        status_code=200,
        headers=headers,
    )


@router.patch("/{user_id}/metadata/{collection_name}")
async def update_metadata(
    user_id: str,
    collection_name: str,
    data: UpdateCollectionMetadataData,
    user_manager: UserManager = Depends(get_user_manager),
):
    """
    Update the preprocessed metadata for a collection.
    This updates the output of the `.preprocess` function in Elysia.
    Any fields not provided will not be updated.

    Args:
        user_id (str): The ID of the user whose metadata to update.
        collection_name (str): The name of the collection to update the metadata for.
        data (UpdateCollectionMetadataData): The data for the request, containing:
            - named_vectors (list[dict]): The named vectors to update.
            - summary (str): The summary to update.
            - mappings (dict): The mappings to update.
            - fields (list[dict]): The fields to update.
        user_manager (UserManager): The user manager.

    Returns:
        (JSONResponse): A JSON response containing:
            - metadata (dict): The updated metadata for the collection (same as the "metadata" output of `collection_metadata`).
            - error (str): An error message if there is an error, otherwise an empty string.
    """
    logger.debug(f"/update_metadata API request received")
    logger.debug(f"User ID: {user_id}")
    logger.debug(f"Collection name: {collection_name}")
    logger.debug(f"Data: {data}")

    # retrieve the current metadata
    try:
        user_local = await user_manager.get_user_local(user_id)
        client_manager = user_local["client_manager"]

        properties = await edit_preprocessed_collection_async(
            collection_name=collection_name,
            client_manager=client_manager,
            named_vectors=(
                [
                    named_vector.model_dump()
                    for named_vector in data.named_vectors
                    if named_vector
                ]
                if data.named_vectors
                else None
            ),
            summary=data.summary,
            mappings=data.mappings,
            fields=(
                [field.model_dump() if field else None for field in data.fields]
                if data.fields
                else None
            ),
        )
    except Exception as e:
        logger.exception(f"Error in /update_metadata API:")
        return JSONResponse(
            content={"metadata": {}, "error": str(e)},
            status_code=200,
        )

    return JSONResponse(
        content={"metadata": properties, "error": ""},
        status_code=200,
    )


@router.delete("/{user_id}/metadata/{collection_name}")
async def delete_metadata(
    user_id: str,
    collection_name: str,
    user_manager: UserManager = Depends(get_user_manager),
) -> JSONResponse:
    """
    Delete the metadata for a collection.

    Args:
        user_id (str): The ID of the user.
        collection_name (str): The name of the collection.
        user_manager (UserManager): The user manager.

    Returns:
        (JSONResponse): A JSON response containing an error message if there is an error, otherwise an empty string.
    """
    logger.debug(f"/delete_metadata API request received")
    logger.debug(f"User ID: {user_id}")
    logger.debug(f"Collection name: {collection_name}")

    # retrieve the current metadata
    try:
        user_local = await user_manager.get_user_local(user_id)
        client_manager = user_local["client_manager"]

        await delete_preprocessed_collection_async(collection_name, client_manager)

    except Exception as e:
        logger.exception(f"Error in /delete_metadata API")
        return JSONResponse(content={"error": str(e)}, status_code=200)

    return JSONResponse(content={"error": ""}, status_code=200)


@router.delete("/{user_id}/metadata/delete/all")
async def delete_all_metadata(
    user_id: str,
    user_manager: UserManager = Depends(get_user_manager),
) -> JSONResponse:
    """
    Delete the metadata for a collection.

    Args:
        user_id (str): The ID of the user.
        user_manager (UserManager): The user manager.

    Returns:
        (JSONResponse): A JSON response containing an error message if there is an error, otherwise an empty string.
    """
    logger.debug(f"/delete_all_metadata API request received")
    logger.debug(f"User ID: {user_id}")

    # retrieve the current metadata
    try:
        user_local = await user_manager.get_user_local(user_id)
        client_manager: ClientManager = user_local["client_manager"]
        async with client_manager.connect_to_async_client() as client:
            if await client.collections.exists("ELYSIA_METADATA__"):
                await client.collections.delete("ELYSIA_METADATA__")

    except Exception as e:
        logger.exception(f"Error in /delete_all_metadata API")
        return JSONResponse(content={"error": str(e)}, status_code=200)

    return JSONResponse(content={"error": ""}, status_code=200)
