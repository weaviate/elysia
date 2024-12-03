import datetime
import random
import os
import dspy
import sys
import json

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

def remove_whitespace_indented(code: str) -> str:
    lines = code.splitlines()
    # Remove empty lines at start and end
    lines = [line for line in lines if line.strip()]
    if not lines:
        return ""
    
    # Find the minimum indentation level (excluding empty lines)
    min_indent = min(len(line) - len(line.lstrip()) for line in lines)
    
    # Remove the common indentation from all lines and clean up internal whitespace
    cleaned_lines = []
    for line in lines:
        # Remove the common indentation prefix
        dedented_line = line[min_indent:]
        # Preserve the remaining indentation but clean up internal whitespace
        indent = len(dedented_line) - len(dedented_line.lstrip())
        cleaned_line = ' ' * indent + dedented_line.strip()
        cleaned_lines.append(cleaned_line)
    
    return '\n'.join(cleaned_lines)


def load_example_from_dict(d: dict):
    return dspy.Example({k: v for k, v in d.items()}).with_inputs(
        "user_prompt", 
        "previous_queries", 
        "conversation_history",
        "data_queried", 
        "previous_reasoning", 
        "collection_information", 
        "current_message"
    )

def find_previous_queries(available_information: Returns):

    previous_queries = []
    for collection_name in available_collections:
        if collection_name in available_information.retrieved:
            metadata = available_information.retrieved[collection_name].metadata
            if "previous_queries" in metadata:
                previous_queries.append({"collection_name": collection_name, "previous_queries": metadata["previous_queries"]})  

    return previous_queries

def create_example(
    # inputs
    user_prompt: str, # input

    # outputs
    code: str,
    collection_name: str,
    return_type: str,
    output_type: str,
    reasoning: str = None, # optional: model reasoning for the choices

    # extra
    route: str = "search/query", # what path should the tree take to get to this point? e.g. specify multiple trees with "search/query/search/query"
):
  
    # clean the strings
    user_prompt = remove_whitespace(user_prompt)
    code = remove_whitespace_indented(code)
    reasoning = remove_whitespace(reasoning)

    # generate by running a tree up to a certain point
    tree = Tree(
        collection_names=available_collections,
        verbosity=0,
        training_route=route,
        training_decision_output=True
    )


    reasoning_update_message = f"I'll search {collection_name} using"

    # get random message features
    if "fetch_objects" in code:
        query_type = "fetch_objects"
    elif "hybrid" in code:
        query_type = "hybrid"
    elif "near_text" in code:
        query_type = "semantic"

    reasoning_update_message += f" {query_type} search"
    
    if "sort" in code:
        sort_start = code.index('Sort.by_property("')
        sort_end = code.index(')', sort_start)
        sort_property = code[sort_start+len("Sort.by_property("):sort_end]
        sort_property = sort_property[sort_property.find('"')+1:sort_property.rfind('"')]

        reasoning_update_message += f", sorting by {sort_property}"

    if "filters" in code:
        filter_start = code.index('Filter.by_property("')
        filter_end = code.index(')', filter_start)
        filter_property = code[filter_start+len("Filter.by_property("):filter_end]
        filter_property = filter_property[filter_property.find('"')+1:filter_property.rfind('"')]

        reasoning_update_message += f", filtering by {filter_property}"

    reasoning_update_message += "."

    # run the tree
    tree.process_sync(user_prompt)

    # arguments output from tree
    available_information = tree.decision_data.available_information
    previous_reasoning = tree.tree_data.previous_reasoning
    collection_information = tree.action_data.collection_information
    current_message = tree.tree_data.current_message
    data_queried = tree.tree_data.data_queried
    conversation_history = tree.tree_data.conversation_history

    reasoning_update_message = current_message + " " + reasoning_update_message

    # find previous queries
    previous_queries = find_previous_queries(available_information)
 
    # return dspy example
    return dspy.Example(
        # inputs
        user_prompt=user_prompt, 
        previous_queries=previous_queries, 
        conversation_history=conversation_history,
        data_queried=data_queried,
        previous_reasoning=previous_reasoning,
        collection_information=collection_information,
        current_message=current_message,
        is_query_possible=True,
        
        # outputs
        code=code,
        collection_name=collection_name,
        return_type=return_type,
        output_type=output_type,
        reasoning_update_message=reasoning_update_message,
        reasoning=reasoning
    ).with_inputs(
        "user_prompt", 
        "previous_queries", 
        "conversation_history",
        "data_queried", 
        "previous_reasoning", 
        "collection_information", 
        "current_message"
    )
    

def make_examples(force=False):

    #  save the examples
    filepath = os.path.join(
        "elysia",
        "training",
        "data"
    )

    filename = "query.json"

    if not os.path.exists(os.path.join(filepath, filename)) or force:
        data = [
            create_example(
                user_prompt="List and sort the most common issues from the verba github issues collection from 2024.",
                collection_name="example_verba_github_issues",
                code="""
                collection.query.fetch_objects(
                    filters=(
                        Filter.by_property("issue_created_at").greater_than(format_datetime(datetime.datetime(2024, 1, 1, tzinfo=datetime.timezone.utc))) &
                        Filter.by_property("issue_created_at").less_than(format_datetime(datetime.datetime(2025, 1, 1, tzinfo=datetime.timezone.utc)))
                    ),
                    sort = Sort.by_property("issue_created_at", ascending=False), # sort so most recent is first
                    limit=30 
                )
                """,
                return_type="ticket",
                output_type="original",
                reasoning="""
                We need to use fetch_objects, as no semantic or keyword search is required, as the user is specifying a generic retrieval.
                We also need to specify a time range between 2024 and 2025, as the user is specifying a time-based retrieval.
                Since the user is asking for the most common issues, we should sort the results by the issue_created_at property in descending order.
                """,
            ),
            create_example(
                user_prompt="Summarise the issue of 'PDF being too large' in verba.",
                collection_name="example_verba_github_issues",
                code="""
                collection.query.hybrid(
                    query="PDF too large",
                    limit=10
                )
                """,
                return_type="ticket",
                output_type="original",
                reasoning="""
                The user is asking for a specific issue, so hybrid search, which captures the exact phrase as well as its meaning, is appropriate.
                The phrase "PDF too large" should capture the issue of the PDF being too large, hoping to pick up this description in the content of the issues.
                We apply a generic limit of 10, to get a broad picture of the issue.
                Later, we will summarise the results, but the user is not asking for an itemised summary or does not seem to be, so we will output "original" return type.
                """,
            ),
            create_example(
                user_prompt="Tell me what people are saying about the issue of the bug where you can't open the settings page in verba.",
                collection_name="example_verba_slack_conversations",
                code="""
                collection.query.hybrid(
                    query="can't open settings page",
                    limit=10
                )
                """,
                return_type="conversation",
                output_type="original",
                reasoning="""
                The user is asking for a specific issue, so hybrid search, which captures the exact phrase as well as its meaning, is appropriate.
                The phrase "can't open settings page" should capture the issue of the settings page not being accessible.
                We apply a generic limit of 10, to get a broad picture of the issue.
                Since the user is asking generically what people (plural) are saying, we should retrieve full conversations, including all messages and message authors, with timestamps and context of other messages in the conversation.
                Also, we will output "original" return type, as the user does not seem to be asking for an itemised summary.
                """,
            ),
            create_example(
                user_prompt="What are the most recent emails people are writing about verba?",
                collection_name="example_verba_email_chains",
                code="""
                collection.query.fetch_objects(
                    filters=(
                        Filter.by_property("message_timestamp").greater_than(format_datetime(datetime.datetime(2024, 9, 1, tzinfo=datetime.timezone.utc))) 
                    ),
                    sort = Sort.by_property("message_timestamp", ascending=False),
                    limit=30 
                )
                """,
                return_type="message",
                output_type="original",
                reasoning="""
                Since we only need to sort and not perform any searches, we can use fetch_objects.
                The user is asking for the most recent emails, so we should sort the results by the message_timestamp property in descending order.
                The user is asking for the most recent emails, without a specified time range. We will use a single filter for this on a generic span of the last 2 months, since the current month is November 2024.
                Additionally, we apply a generic limit of 30, to get a broad picture of the issue.
                Also, we will output "original" return type, as the user does not seem to be asking for an itemised summary.
                We will additionally sort the results by the message_timestamp property in descending order, to get the most recent emails first.
                """,
            ),
            create_example(
                user_prompt="Has Kaladin proposed any new features for verba recently?",
                collection_name="example_verba_slack_conversations",
                code="""
                collection.query.near_text(
                    query="feature proposal",
                    filters=(
                        Filter.by_property("message_timestamp").greater_than(format_datetime(datetime.datetime(2024, 9, 1, tzinfo=datetime.timezone.utc))) &
                        Filter.by_property("message_author").equal("Kaladin")
                    ),
                    limit=5
                )
                """,
                return_type="conversation",
                output_type="original",
                reasoning="""
                The user is asking for a specific type of message, a "feature proposal", so we should search for this phrase. Since this can be quite broad, we will use a semantic search (near_text).
                The phrase "feature proposal" should capture the issue of the user proposing a new feature.
                We apply a generic limit of 5, to get a broad picture of the issue.
                We can see in the collection information that there is an author called "Kaladin", so we can filter for this exactly using the Filter.by_property("message_author").equal("Kaladin") filter.
                Also, we will output "original" return type, as the user does not seem to be asking for an itemised summary.
                """,
            ),  
            create_example(
                user_prompt="Give me a range of the different tops there are in my ecommernce collection",
                collection_name="ecommerce",
                code="""
                collection.query.fetch_objects(
                    filters=(
                        Filter.by_property("category").equal("Tops")
                    ),
                    limit=10
                )
                """,
                return_type="ecommerce",
                output_type="summary",
                reasoning="""
                The user is asking for a range of the different tops there are in the ecommerce collection, so we should use a generic fetch_objects query.
                We can see in the collection information that there is a category called "Tops", so we can filter for this exactly using the Filter.by_property("category").equal("Tops") filter.
                We apply an arbitrary limit of 10, to get a broad picture of the issue.
                Since the user seems to be looking for an overview, and not the properties of the individual items, we should output "summary" return type to give a concise overview of the different tops.
                """,
            ),
            create_example(
                user_prompt="I want to find some 90s aesthetic clothing",
                collection_name="ecommerce",
                code="""
                collection.query.near_text(
                    query="90s aesthetic",
                    limit=10
                )
                """,
                return_type="ecommerce",
                output_type="original",
                reasoning="""
                The user is asking for some 90s aesthetic clothing. None of the categories in the collection match this request, so we should use a semantic search (near_text), to pick up the 90s aesthetic in the description of the clothing.
                The phrase "90s aesthetic" should capture the issue of the user looking for 90s aesthetic clothing.
                We apply a generic limit of 10, to get a broad picture of the issue.
                The user is requesting individual items, so we should show them the original return type of the clothing as they are.
                """,
            ),
            create_example(
                user_prompt="Summarise Sofia's slack messages from the last 2 months.",
                collection_name="example_verba_slack_conversations",
                code="""
                collection.query.fetch_objects(
                    filters=(
                        Filter.by_property("message_timestamp").greater_than(format_datetime(datetime.datetime(2024, 9, 1, tzinfo=datetime.timezone.utc))) &
                        Filter.by_property("message_author").equal("Sofia")
                    ),
                    limit=30
                )
                """,
                return_type="message",
                output_type="original",
                reasoning="""
                The user is asking for a summary of Sofia's slack messages from the last 2 months. Note that they are not asking for an summary of each message, but a summary of all messages from Sofia. So we should output "original" return type.
                Since the user is not looking for any specific information within the messages, we should use the generic fetch_objects query, with filters to pick out Sofia's messages.
                We can see in the collection information that there is an author called "Sofia", so we can filter for this exactly using the Filter.by_property("message_author").equal("Sofia") filter.
                We apply a generic limit of 30, to get a broad picture of the issue.
                """,
            )
        ]

        os.makedirs(filepath, exist_ok=True)
        with open(os.path.join(filepath, filename), "w") as f:
            out = [example.toDict() for example in data]
            json.dump(out, f)
        
        print(f"Examples saved to {os.path.join(filepath, filename)}")

    else:
        with open(os.path.join(filepath, filename), "r") as f:
            data = [load_example_from_dict(example) for example in json.load(f)]

    return data


def train_query_fewshot(force=True):
    
    train = make_examples(force=force)

    query_executor = QueryExecutor(available_collections, available_return_types)
    
    num_fewshot_demos = len(train)
    optimizer = LabeledFewShot(k=num_fewshot_demos)
    trained_fewshot = optimizer.compile(
        query_executor.activate_assertions(), 
        trainset=train
    )

    # Create the full directory path
    filepath = os.path.join(
        "elysia",
        "training",
        "dspy_models",
        "query"
    )
    os.makedirs(filepath, exist_ok=True)  # This creates the directory if it doesn't exist

    # Save the file in the created directory
    full_filepath = os.path.join(filepath, f"fewshot_k{num_fewshot_demos}.json")
    trained_fewshot.save(full_filepath)

    print("Training completed!")
    print(f"Model saved to {full_filepath}")


if __name__ == "__main__":
    train_query_fewshot()