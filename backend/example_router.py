from backend.querying.prompt_executors import QueryRewriterExecutor
from backend.summarizing.prompt_executors import SummarizingExecutor
from backend.routing.prompt_executors import RoutingExecutor

from backend.querying.query import QueryOptions
from backend.util.logging import backend_print

import weaviate
import os
import dspy
from weaviate.classes.init import Auth

from rich import print

client = weaviate.connect_to_weaviate_cloud(
    cluster_url = os.environ.get("WCD_URL"),
    auth_credentials = Auth.api_key(os.environ.get("WCD_API_KEY")),
    headers = {"X-OpenAI-API-Key": os.environ.get("OPENAI_API_KEY")}
)

dspy.settings.configure(lm = dspy.LM(model = "claude-3-5-sonnet-20240620"))

if __name__ == "__main__":

    routing_options = {
        "query": QueryOptions,
        "summarize": SummarizingExecutor()
    }

    routing_properties = [{
        "name": "query",
        "description": "Use this property to query the knowledge base. This should be used when the user is lacking information about a specific issue.",
    },
    {
        "name": "summarize",
        "description": "Use this property to summarize some information. This should be used when the user wants a high-level overview of some retrieved information.",
    }]

    user_prompt = "Write a summary of the issue related to pdf limit exceeded error from github issues"

    completed_tasks = []

    main_router = RoutingExecutor()
    properties = [{
        "name": "query",
        "description": "Use this property to query the knowledge base. This should be used when the user is lacking information about a specific issue.",
    },
    {
        "name": "summarize",
        "description": "Use this property to summarize some information. This should be used when the user wants a high-level overview of some retrieved information.",
    }]

    result = main_router(user_prompt=user_prompt, available_properties=properties, completed_tasks=completed_tasks)
    print(result)