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
from elysia.querying.prompt_executors import QueryExecutor
from elysia.tree.tree import Tree
from elysia.tree.objects import Returns
from elysia.util.parsing import remove_whitespace, format_datetime

available_collections = [
    "example_verba_github_issues", 
    "example_verba_slack_conversations", 
    "example_verba_email_chains",
    "ecommerce"
]
available_return_types={
    "conversation": "retrieve full conversations, including all messages and message authors, with timestamps and context of other messages in the conversation.",
    "message": "retrieve individual messages, only including the author of each individual message and timestamp, without surrounding context of other messages by different people.",
    "ticket": "retrieve individual tickets, including all fields of the ticket.",
    "ecommerce": "retrieve individual products, including all fields of the product.",
    "generic": "retrieve any other type of information that does not fit into the other categories."
}

def create_example(
    # inputs
    user_prompt: str, # input

    # outputs
    query_code: str,
    collection_name: str,
    return_type: str,
    output_type: str,
    input_text_return: bool = False, # optional: text return from the query
    reasoning: str = None, # optional: model reasoning for the choices

    # extra
    route: str = "", # what path should the tree take to get to this point? e.g. specify multiple trees with "search/query/search/query"
):
  
    # generate by running a tree up to a certain point
    tree = Tree(
        collection_names=available_collections,
        verbosity=2,
        training_route=route,
        training_decision_output=True
    )

    # arguments output from tree
    available_information = tree.returns
    previous_reasoning = tree.previous_reasoning
    collection_information = tree.collection_information
    current_message = tree.current_message

    # text return can be input
    if input_text_return:
        text_return = input(f"Current message so far: {current_message}\n\nPlease enter the next sentence: ")
      
if __name__ == "__main__":
    query_executor = QueryExecutor(available_collections, available_return_types)