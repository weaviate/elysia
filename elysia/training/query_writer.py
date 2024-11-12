import datetime
import random
import os
import dspy
import sys

sys.path.append(os.getcwd())
os.chdir("../..")

from dspy.teleprompt import LabeledFewShot

from weaviate.classes.query import Filter, MetadataQuery, Sort

from elysia.querying.prompt_executors import QueryCreatorExecutor
from elysia.globals.weaviate_client import client

def remove_whitespace(code: str) -> str:
    return " ".join(code.split())

def format_datetime(dt: datetime.datetime) -> str:
    dt = dt.isoformat("T")
    return dt[:dt.find("+")] + "Z"

def create_example(user_prompt, year, month, collection_name, previous_queries, code):

    date = datetime.datetime(year, month, random.randint(1, 28))
    reference = {
        "datetime": format_datetime(date),
        "day_of_week": date.strftime("%A"),
        "time_of_day": date.strftime("%I:%M %p")
    }

    # Get the collection
    collection = client.collections.get(collection_name)
    
    # Get the example field - an example of the data in the collection
    example_field = collection.query.fetch_objects(limit=10).objects[1].properties
    for key in example_field:
        if isinstance(example_field[key], datetime.datetime):
            example_field[key] = example_field[key].isoformat().replace("+","Z")

    # list of all the items in the example field / any field in the collection
    data_fields = list(example_field.keys())

    # return the example as a dspy.Example object
    return dspy.Example(
        user_prompt=(user_prompt),
        reference=reference,
        data_fields=data_fields,
        example_field=example_field,
        previous_queries=[(q) for q in previous_queries],
        code=(code)
    ).with_inputs("user_prompt", "reference", "data_fields", "example_field", "previous_queries")

data = [
    create_example(
        user_prompt="List and sort the most common issues from the verba github issues collection from 2024.",
        year=2024,
        month=6,
        collection_name="example_verba_github_issues",
        previous_queries=[],
        code="""
        collection.query.fetch_objects(
            filters=(
                Filter.by_property("issue_created_at").greater_than(format_datetime(datetime.datetime(2024, 1, 1, tzinfo=datetime.timezone.utc))) &
                Filter.by_property("issue_created_at").less_than(format_datetime(datetime.datetime(2025, 1, 1, tzinfo=datetime.timezone.utc)))
            ),
            sort = Sort.by_property("issue_created_at", ascending=False), # sort so most recent is first
            limit=30 # an arbitrary large number
        )
        """.strip()
    ),
    create_example(
        user_prompt="Summarise the issue of 'PDF being too large' in verba.",
        year=2023,
        month=2,
        collection_name="example_verba_github_issues",
        previous_queries=[],
        code="""
        collection.query.hybrid(
            query="large pdf upload",
            limit=3
        )
        """.strip()
    ),
    create_example(
        user_prompt="Tell me what people are saying about the issue of the bug where you can't open the settings page in verba.",
        year=2024,
        month=4,
        collection_name="example_verba_slack_conversations",
        previous_queries=[],
        code="""
        collection.query.hybrid(
            query="settings page won't open",
            limit=3
        )
        """.strip()
    ),
    create_example(
        user_prompt="What are the most recent emails people are writing about verba?",
        year=2024,
        month=11,
        collection_name="example_verba_email_chains",
        previous_queries=[],
        code="""
        collection.query.fetch_objects(
            filters=(
                Filter.by_property("message_timestamp").greater_than(format_datetime(datetime.datetime(2024, 9, 1, tzinfo=datetime.timezone.utc))) # last 2 months
            ),
            sort = Sort.by_property("message_timestamp", ascending=False), # sort so most recent is first (only for fetch_objects)
            limit=30 # an arbitrary large number
        )
        """.strip()
    ),
    create_example(
        user_prompt="Write me a summary of the issue with the openAI vectorizer not working in the last year",
        year=2024,
        month=10,
        collection_name="example_verba_github_issues",
        previous_queries=[],
        code="""
        collection.query.hybrid(
            query="openai vectorizer",
            filters=(
                Filter.by_property("issue_created_at").greater_than(format_datetime(datetime.datetime(2023, 10, 1, tzinfo=datetime.timezone.utc))) &
                Filter.by_property("issue_created_at").less_than(format_datetime(datetime.datetime(2024, 10, 1, tzinfo=datetime.timezone.utc)))
            )
            limit = 10
        )
        """.strip()
    ),
    create_example(
        user_prompt="What are people saying lately about the issue of the bug where you can't open the settings page in verba?",
        year=2023,
        month=7,
        collection_name="example_verba_slack_conversations",
        previous_queries=["""
        collection.query.near_text(
            query="can't open the settings page",
            filters=(
                Filter.by_property("message_timestamp").greater_than(format_datetime(datetime.datetime(2023, 6, 1, tzinfo=datetime.timezone.utc)))
            )
            limit = 10
        )""".strip()
        ],
        code="""
        collection.query.near_text(
            query="menu page won't open",
            filters=(
                Filter.by_property("message_timestamp").greater_than(format_datetime(datetime.datetime(2023, 4, 1, tzinfo=datetime.timezone.utc)))
            ),
            limit=3
        )
        """.strip()
    ),
    create_example(
        user_prompt="Summarise Sofia's slack messages from the last 2 months.",
        year=2024,
        month=11,
        collection_name="example_verba_slack_conversations",
        previous_queries=[],
        code="""
        collection.query.fetch_objects(
            filters=(
                Filter.by_property("message_timestamp").greater_than(format_datetime(datetime.datetime(2024, 9, 1, tzinfo=datetime.timezone.utc))) &
                Filter.by_property("message_author").equal("Sofia")
            ),
            limit=30
        )
        """.strip()
    ),
    create_example(
        user_prompt="Has Kaladin proposed any new features for verba recently?",
        year=2024,
        month=10,
        collection_name="example_verba_slack_conversations",
        previous_queries=[],
        code="""
        collection.query.hybrid(
            query="feature proposal",
            filters=(
                Filter.by_property("message_timestamp").greater_than(format_datetime(datetime.datetime(2024, 9, 1, tzinfo=datetime.timezone.utc))) &
                Filter.by_property("message_author").equal("Kaladin")
            ),
            limit=3
        )
        """.strip()
    ),
    create_example(
        user_prompt="Has Kaladin proposed any new features for verba recently?",
        year=2024,
        month=10,
        collection_name="example_verba_slack_conversations",
        previous_queries=[],
        code="""
        collection.query.hybrid(
            query="feature proposal for verba",
            filters=(
                Filter.by_property("message_timestamp").greater_than(format_datetime(datetime.datetime(2024, 9, 1, tzinfo=datetime.timezone.utc))) &
                Filter.by_property("message_author").equal("Kaladin")
            ),
            limit=3
        )
        """.strip()
    ),
    create_example(
        user_prompt="What kind of marketing emails have been sent to verba users?",
        year=2022,
        month=2,
        collection_name="example_verba_email_chains",
        previous_queries=[],
        code="""
        collection.query.near_text(
            query="marketing",
            limit=3
        )
        """.strip()
    ),
    create_example(
        user_prompt="Alphabetically list the emails from Vin sent in 2024.",
        year=2025,
        month=3,
        collection_name="example_verba_email_chains",
        previous_queries=[],
        code="""
        collection.query.fetch_objects(
            filters=(
                Filter.by_property("message_author").equal("Vin") &
                Filter.by_property("message_timestamp").greater_than(format_datetime(datetime.datetime(2024, 1, 1, tzinfo=datetime.timezone.utc))) &
                Filter.by_property("message_timestamp").less_than(format_datetime(datetime.datetime(2025, 1, 1, tzinfo=datetime.timezone.utc)))
            ),
            sort = Sort.by_property("message_content", ascending=True), # sort so most recent is first
        )
        """.strip()
    )
]

def train_query_creator_fewshot():
    
    train = data

    num_fewshot_demos = 12
    optimizer = LabeledFewShot(k=num_fewshot_demos)
    trained_fewshot = optimizer.compile(
        QueryCreatorExecutor().activate_assertions(), 
        trainset=train
    )

    # Create the full directory path
    filepath = os.path.join(
        "elysia",
        "training",
        "dspy_models",
        "agentic_query"
    )
    os.makedirs(filepath, exist_ok=True)  # This creates the directory if it doesn't exist

    # Save the file in the created directory
    full_filepath = os.path.join(filepath, f"fewshot_k{num_fewshot_demos}.json")
    trained_fewshot.save(full_filepath)

if __name__ == "__main__":
    
    lm = dspy.LM(model="gpt-4o-mini")
    dspy.settings.configure(lm=lm)

    train_query_creator_fewshot()