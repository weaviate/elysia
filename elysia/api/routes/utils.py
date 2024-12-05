
from elysia.api.api_types import (
    NERData, 
    TitleData, 
    ObjectRelevanceData,
    DebugData
)

# Logging
from elysia.api.core.logging import logger

# Settings
from elysia.api.core.config import nlp

# Dependencies
from elysia.api.dependencies.common import get_tree_manager

# Services
from elysia.api.services.tree import TreeManager

# FastAPI
from fastapi import APIRouter, Depends    
from fastapi.responses import JSONResponse

# LLM
from elysia.api.services.prompt_executors import TitleCreatorExecutor, ObjectRelevanceExecutor

router = APIRouter()

@router.post("/ner")
async def named_entity_recognition(data: NERData):
    """
    Performs Named Entity Recognition using spaCy.
    Returns a list of entities with their labels, start and end positions.
    """
    doc = nlp(data.text)
    
    out = {
        "text": data.text,  
        "entity_spans": [],
        "noun_spans": [],
        "error": ""
    }
    
    try:
        # Get entity spans
        for ent in doc.ents:
            out["entity_spans"].append((ent.start_char, ent.end_char))
        
        # Get noun spans
        for token in doc:
            if token.pos_ == "NOUN":
                span = doc[token.i:token.i + 1]  
                out["noun_spans"].append((span.start_char, span.end_char))
        
        logger.info(f"Returning NER results: {out}")
    except Exception as e:
        logger.error(f"Error in NER: {str(e)}")
        out["error"] = str(e)
    
    return JSONResponse(content=out, status_code=200)

@router.post("/title")
async def title(data: TitleData):
    try:
        title_creator = TitleCreatorExecutor()
        title = title_creator(data.text)
        logger.info(f"Returning title: {title.title}")
    except Exception as e:
        logger.error(f"Error in title: {str(e)}")
        out = {
            "title": "",
            "error": str(e)
        }
        return JSONResponse(content=out, status_code=200)

    out = {
        "title": title.title,
        "error": ""
    }
    return JSONResponse(content=out, status_code=200)

@router.post("/object_relevance")
async def object_relevance(
    data: ObjectRelevanceData,
    tree_manager: TreeManager = Depends(get_tree_manager)
):
    error = ""
    object_relevance = ObjectRelevanceExecutor()
    
    try:
        tree = tree_manager.get_tree(data.user_id, data.conversation_id)
        user_prompt = tree.query_id_to_prompt[data.query_id]
        prediction = object_relevance(user_prompt, data.objects)
        any_relevant = eval(prediction.any_relevant)
        assert isinstance(any_relevant, bool)
    
    except Exception as e:
        any_relevant = False
        error = str(e)

    return JSONResponse(
        content={
            "conversation_id": data.conversation_id,
            "any_relevant": any_relevant,
            "error": error
        }, 
        status_code=200
    )


@router.post("/debug")
async def debug(
    data: DebugData,
    tree_manager: TreeManager = Depends(get_tree_manager)
):
    tree = tree_manager.get_tree(data.user_id, data.conversation_id)
    base_lm = tree.base_lm
    complex_lm = tree.complex_lm 

    histories = [None]*2
    for i, lm in enumerate([base_lm, complex_lm]):
        histories[i] = [
            lm_history["messages"] + [
                {
                    "role": "assistant",
                    "content": lm_history["response"].choices[0].message.content
                }
            ]
            for lm_history in lm.history
        ]
        
    out = {
        "base_lm": {
            "model": base_lm.model,
            "chat": histories[0]
        },
        "complex_lm": {
            "model": complex_lm.model,
            "chat": histories[1]
        }
    }
    return JSONResponse(content=out, status_code=200)
