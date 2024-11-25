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

available_collections = ["example_verba_github_issues", "example_verba_slack_conversations", "example_verba_email_chains"]
available_return_types = {
    "conversation": "retrieve a full conversation, including all messages and message authors, with timestamps and context of other messages in the conversation.",
    "message": "retrieve only a single message, only including the author of each individual message and timestamp, without surrounding context of other messages by different people.",
    "ticket": "retrieve a single ticket, including all fields of the ticket.",
    "generic": "retrieve any other type of information that does not fit into the other categories."
}

def remove_whitespace(code: str) -> str:
    return " ".join(code.split())

def format_datetime(dt: datetime.datetime) -> str:
    dt = dt.isoformat("T")
    return dt[:dt.find("+")] + "Z"

def create_example(user_prompt, collection_name, return_type, output_type, data_queried):

    # random date for reference field
    year = random.randint(2020, 2024)
    month = random.randint(1, 12)
    date = datetime.datetime(year, month, random.randint(1, 28))

    reference = {
        "datetime": format_datetime(date),
        "day_of_week": date.strftime("%A"),
        "time_of_day": date.strftime("%I:%M %p")
    }
    
    # return the example as a dspy.Example object
    return dspy.Example(
        user_prompt=user_prompt,
        reference=reference,
        previous_reasoning={},
        data_queried=data_queried,
        available_collections=available_collections,
        available_return_types=available_return_types,
        collection_name=collection_name,
        return_type=return_type,
        output_type=output_type
    ).with_inputs("user_prompt", "reference", "previous_reasoning", "data_queried", "collection_name", "return_type", "output_type")

data = [
    create_example(
        user_prompt = "List the common issues in the app",
        collection_name = "example_verba_github_issues",
        return_type = "ticket",
        output_type = "original",
        data_queried = {}
    ),
    create_example(
        user_prompt = "List the most common issues from the verba github issues collection from 2024, sort by the most recent.",
        collection_name = "example_verba_github_issues",
        return_type = "ticket",
        output_type = "original",
        data_queried = {}
    ),
    create_example(
        user_prompt = "What did Bob say in his last slack message?",
        collection_name = "example_verba_slack_conversations",
        return_type = "message",
        output_type = "original",
        data_queried = {}
    ),
    create_example(
        user_prompt = "Summarise Kaladins emails and messages about verba from 2024",
        collection_name = "example_verba_slack_conversations",
        return_type = "message",
        output_type = "original",
        data_queried = {"example_verba_email_chains": 20}
    ),
    create_example(
        user_prompt = "Give an itemised summary of the most recent messages in the verba slack conversations collection",
        collection_name = "example_verba_slack_conversations",
        return_type = "message",
        output_type = "summary",
        data_queried = {}
    ),
    create_example(
        user_prompt = "Give me individual summaries of Laura's messages and conversations about the PDF issue in verba",
        collection_name = "example_verba_email_chains",
        return_type = "conversation",
        output_type = "summary",
        data_queried = {"example_verba_slack_conversations": 14}
    ),
    create_example(
        user_prompt = "Summarise issues with the OpenAI vectoriser in verba",
        collection_name = "example_verba_github_issues",
        return_type = "ticket",
        output_type = "original",
        data_queried = {}
    ),
    create_example(
        user_prompt = "Tell me all you can about the PDF issue in verba",
        collection_name = "example_verba_slack_conversations",
        return_type = "conversation",
        output_type = "original",
        data_queried = {}
    ),
    create_example(
        user_prompt = "Tell me all you can about the PDF issue in verba",
        collection_name = "example_verba_github_issues",
        return_type = "ticket",
        output_type = "original",
        data_queried = {"example_verba_slack_conversations": 20}
    ),
    create_example(
        user_prompt = "Tell me all you can about the PDF issue in verba",
        collection_name = "example_verba_email_chains",
        return_type = "conversation",
        output_type = "original",
        data_queried = {"example_verba_slack_conversations": 20, "example_verba_github_issues": 20}
    ),
    create_example(
        user_prompt = "Give a breakdown of the issues in verba",
        collection_name = "example_verba_github_issues",
        return_type = "ticket",
        output_type = "summary",
        data_queried = {}
    )
]

def train_query_initialiser_fewshot():
    
    train = data

    num_fewshot_demos = 12
    optimizer = LabeledFewShot(k=num_fewshot_demos)
    trained_fewshot = optimizer.compile(
        QueryInitialiserExecutor().activate_assertions(), 
        trainset=train
    )

    # Create the full directory path
    filepath = os.path.join(
        "elysia",
        "training",
        "dspy_models",
        "query_initialiser"
    )
    os.makedirs(filepath, exist_ok=True)  # This creates the directory if it doesn't exist

    # Save the file in the created directory
    full_filepath = os.path.join(filepath, f"fewshot_k{num_fewshot_demos}.json")
    print(f"CWD: {os.getcwd()}")
    print(f"Saving fewshot model to {full_filepath}")
    trained_fewshot.save(full_filepath)

if __name__ == "__main__":
    train_query_initialiser_fewshot()