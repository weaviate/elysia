# FastAPI
from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse

from elysia.api.api_types import (
    DebugData,
    FollowUpSuggestionsData,
    NERData,
    TitleData,
)

from elysia.tree.tree import Tree

# Logging
from elysia.api.core.log import logger

# Dependencies
from elysia.api.dependencies.common import get_user_manager

# Services
from elysia.api.services.user import UserManager

# Settings
from elysia.config import nlp

# util
from elysia.api.core.log import logger

router = APIRouter()


@router.post("/ner")
async def named_entity_recognition(data: NERData):
    """
    Performs Named Entity Recognition using spaCy.
    Returns a list of entities with their labels, start and end positions.
    """
    logger.debug(f"/ner API request received")
    logger.debug(f"Text: {data.text}")

    try:

        doc = nlp(data.text)
        out = {"text": data.text, "entity_spans": [], "noun_spans": [], "error": ""}

        for ent in doc.ents:
            out["entity_spans"].append((ent.start_char, ent.end_char))

        # Get noun spans
        for token in doc:
            if token.pos_ == "NOUN":
                span = doc[token.i : token.i + 1]
                out["noun_spans"].append((span.start_char, span.end_char))

        return JSONResponse(content=out, status_code=200)

    except Exception as e:
        logger.exception(f"Error in /ner API")
        return JSONResponse(
            content={
                "text": data.text,
                "entity_spans": [],
                "noun_spans": [],
                "error": str(e),
            },
            status_code=200,
        )


@router.post("/title")
async def title(data: TitleData, user_manager: UserManager = Depends(get_user_manager)):
    logger.debug(f"/title API request received")
    logger.debug(f"User ID: {data.user_id}")
    logger.debug(f"Conversation ID: {data.conversation_id}")
    logger.debug(f"Text: {data.text}")

    if user_manager.check_tree_timeout(data.user_id, data.conversation_id):
        logger.warning(
            f"(/title) Conversation {data.conversation_id} has timed out for user {data.user_id}"
        )
        return JSONResponse(
            content={"title": "", "error": "Conversation has timed out"},
            status_code=401,
        )

    try:
        tree: Tree = await user_manager.get_tree(data.user_id, data.conversation_id)
        title = await tree.create_conversation_title_async()
        return JSONResponse(
            content={"title": title, "error": ""},
            status_code=200,
        )

    except Exception as e:
        logger.exception(f"Error in /title API")
        return JSONResponse(
            content={"title": "", "error": str(e)},
            status_code=200,
        )


# @router.post("/object_relevance")
# async def object_relevance(
#     data: ObjectRelevanceData, user_manager: UserManager = Depends(get_user_manager)
# ):
#     logger.debug(f"/object_relevance API request received")
#     logger.debug(f"User ID: {data.user_id}")
#     logger.debug(f"Conversation ID: {data.conversation_id}")
#     logger.debug(f"Query ID: {data.query_id}")
#     logger.debug(f"Number of objects: {len(data.objects)}")

#     if user_manager.check_tree_timeout(data.user_id, data.conversation_id):
#         logger.warning(
#             f"(/object_relevance) Conversation {data.conversation_id} has timed out for user {data.user_id}"
#         )
#         return JSONResponse(
#             content={"title": "", "error": "Conversation has timed out"},
#             status_code=401,
#         )


#     try:

#         config = user_manager.get_current_user_config(data.user_id)

#         error = ""
#         object_relevance = ObjectRelevanceExecutor()

#         tree = await user_manager.get_tree(data.user_id, data.conversation_id)
#         user_prompt = tree.query_id_to_prompt[data.query_id]
#         prediction = object_relevance(
#             user_prompt=user_prompt,
#             objects=data.objects,
#             lm=load_lm(config.BASE_PROVIDER, config.BASE_MODEL, config.MODEL_API_BASE),
#         )
#         any_relevant = prediction.any_relevant

#     except Exception as e:
#         any_relevant = True
#         error = str(e)

#     return JSONResponse(
#         content={
#             "conversation_id": data.conversation_id,
#             "any_relevant": any_relevant,
#             "error": error,
#         },
#         status_code=200,
#     )


@router.post("/follow_up_suggestions")
async def follow_up_suggestions(
    data: FollowUpSuggestionsData, user_manager: UserManager = Depends(get_user_manager)
):
    logger.debug(f"/follow_up_suggestions API request received")
    logger.debug(f"User ID: {data.user_id}")
    logger.debug(f"Conversation ID: {data.conversation_id}")

    if user_manager.check_tree_timeout(data.user_id, data.conversation_id):
        logger.warning(
            f"(/follow_up_suggestions) Conversation {data.conversation_id} has timed out for user {data.user_id}"
        )
        return JSONResponse(
            content={"suggestions": [], "error": "Conversation has timed out"},
            status_code=408,
        )

    try:
        # wait for tree event to be completed
        local_user = await user_manager.get_user_local(data.user_id)
        event = local_user["tree_manager"].get_event(data.conversation_id)
        # await event.wait()

        # get tree from user_id, conversation_id
        tree: Tree = await user_manager.get_tree(data.user_id, data.conversation_id)

        suggestions = await tree.get_follow_up_suggestions_async()

        event.set()

        return JSONResponse(
            content={"suggestions": suggestions, "error": ""},
            status_code=200,
        )

    except Exception as e:
        logger.exception(f"Error in /follow_up_suggestions API")
        return JSONResponse(
            content={"suggestions": [], "error": str(e)}, status_code=200
        )


@router.post("/debug")
async def debug(data: DebugData, user_manager: UserManager = Depends(get_user_manager)):
    logger.debug(f"/debug API request received")
    logger.debug(f"User ID: {data.user_id}")
    logger.debug(f"Conversation ID: {data.conversation_id}")

    try:
        tree = await user_manager.get_tree(data.user_id, data.conversation_id)
        if tree.debug:
            base_lm = tree.base_lm
            complex_lm = tree.complex_lm

            histories = [None] * 2
            for i, lm in enumerate([base_lm, complex_lm]):
                histories[i] = []
                for lm_history in lm.history:
                    message_thread = []
                    for message in lm_history["messages"]:
                        if isinstance(message["content"], list):
                            # assume its a system message list with one dict element
                            message["content"] = message["content"][0]["text"]
                        message_thread.append(message)

                    message_thread.append(
                        {
                            "role": "assistant",
                            "content": lm_history["response"]
                            .choices[0]
                            .message.content,
                        }
                    )
                    histories[i].append(message_thread)

            out = {
                "base_lm": {"model": base_lm.model, "chat": histories[0]},
                "complex_lm": {"model": complex_lm.model, "chat": histories[1]},
            }
            return JSONResponse(content=out, status_code=200)

        else:
            return JSONResponse(content={}, status_code=200)
    except Exception as e:
        logger.exception(f"Error in /debug API")
        return JSONResponse(
            content={"base_lm": {}, "complex_lm": {}, "error": str(e)}, status_code=200
        )


# @router.post("/get_user_requests")
# async def get_user_requests(
#     data: GetUserRequestsData, user_manager: UserManager = Depends(get_user_manager)
# ):
#     num_requests, max_requests = await user_manager.get_user_requests(data.user_id)
#     return JSONResponse(
#         content={"num_requests": num_requests, "max_requests": max_requests},
#         status_code=200,
#     )
