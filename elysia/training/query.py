import datetime
import random
import os
import dspy
import sys

sys.path.append(os.getcwd())
os.chdir("../..")

from dspy.teleprompt import LabeledFewShot

from weaviate.classes.query import Filter, MetadataQuery, Sort

from elysia.globals.weaviate_client import client
from elysia.querying.prompt_executors import QueryInitialiserExecutor

available_collections = [
    "example_verba_github_issues", 
    "example_verba_slack_conversations", 
    "example_verba_email_chains",
    "ecommerce"
]
available_return_types = {
    "conversation": "retrieve a full conversation, including all messages and message authors, with timestamps and context of other messages in the conversation.",
    "message": "retrieve only a single message, only including the author of each individual message and timestamp, without surrounding context of other messages by different people.",
    "ticket": "retrieve a single ticket, including all fields of the ticket.",
    "generic": "retrieve any other type of information that does not fit into the other categories.",
    "ecommerce": "retrieve a single item from the ecommerce collection, including all fields of the item."
}