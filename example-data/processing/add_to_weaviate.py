import weaviate
import pandas as pd
import json
import os
import sys
import ast

from tqdm.auto import tqdm

from weaviate.classes.query import Filter
from weaviate.classes.init import Auth
from weaviate.classes.config import Configure, Property, DataType
from weaviate.util import generate_uuid5  

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def force_create_collection(client, collection_name: str, main_vector_name: str):

    # Create collection or delete existing one
    if client.collections.exists(collection_name):
        print(f"Collection {collection_name} already exists, deleting...")
        collection = client.collections.delete(collection_name)

    collection = client.collections.create(
        name = collection_name,
        vectorizer_config = [
            Configure.NamedVectors.text2vec_openai(
                name = main_vector_name
            )
        ],
        properties = [
            Property(
                name = main_vector_name,
                data_type = DataType.TEXT
            )
        ]
    )

    return collection

def soft_create_collection(client, collection_name: str, main_vector_name: str):

    # Create collection or delete existing one
    if client.collections.exists(collection_name):
        print(f"Collection {collection_name} already exists, loading...")
        return client.collections.get(collection_name)

    else:
        collection = client.collections.create(
            name = collection_name,
            vectorizer_config = [
                Configure.NamedVectors.text2vec_openai(
                    name = main_vector_name
                )
            ],
            properties = [
                Property(
                    name = main_vector_name,
                    data_type = DataType.TEXT
                )
            ]
        )

        return collection


def add_conversations_to_weaviate(client, df: pd.DataFrame, collection_name: str, force: bool = False):

    if force:
        collection = force_create_collection(client, collection_name, "message_content")
    else:
        collection = soft_create_collection(client, collection_name, "message_content")

    unq_id = 0
    # add data to collection
    with collection.batch.dynamic() as batch:

        for i, row in tqdm(df.iterrows(), total=len(df), desc=f"Adding {collection_name} to Weaviate"):

            conversation = ast.literal_eval(row["conversation"])

            for j, message in enumerate(conversation):
                try:
                    # create data object
                    data_object = {
                        "conversation_id": unq_id,
                        "message_index": j,
                        "message_author": message["name"],
                        "message_content": message["content"],
                        "message_timestamp": message["timestamp"]
                    }
                    
                    # weaviate metadata
                    uuid = generate_uuid5(data_object)
                    
                    if not collection.data.exists(uuid):
                        batch.add_object(
                            properties = data_object,
                            uuid = uuid
                        )
                except Exception as e:
                    print(f"Error adding message {j} of conversation {unq_id}: {e}, continuing...")
            
            unq_id += 1

def add_issues_to_weaviate(client, df: pd.DataFrame, collection_name: str, force: bool = False): 

    if force:
        collection = force_create_collection(client, collection_name, "issue_content")
    else:
        collection = soft_create_collection(client, collection_name, "issue_content")

    # add data to collection
    for i, row in tqdm(df.iterrows(), total=len(df), desc=f"Adding {collection_name} to Weaviate"):

        try:
            # create data object
            data_object = {
                "issue_id": row["id"],
                "issue_title": row["title"],
                "issue_content": row["body"],
                "issue_created_at": row["created_at"],
                "issue_updated_at": row["updated_at"],
            }

            # weaviate metadata
            uuid = generate_uuid5(data_object)

            if not collection.data.exists(uuid):
                collection.data.insert(
                    properties = data_object,
                    uuid = uuid
                )
        except Exception as e:
            print(f"Error adding issue {i}: {e}, continuing...")

if __name__ == "__main__":

    # connect to weaviate cloud
    client = weaviate.connect_to_weaviate_cloud(
        cluster_url = os.environ.get("WCD_URL"),
        auth_credentials = Auth.api_key(os.environ.get("WCD_API_KEY")),
        headers = {"X-OpenAI-API-Key": os.environ.get("OPENAI_API_KEY")}
    )

    # read data frames
    github_issues_df  = pd.read_csv("verba_github_issues.csv")
    slack_messages_df = pd.read_csv("verba_slack_conversations.csv")
    email_chains_df   = pd.read_csv("verba_email_chains.csv")

    # add github issues data to weaviate collection
    add_issues_to_weaviate(client, github_issues_df, "example_verba_github_issues", force=True)
    
    # # add email and slack data to weaviate collections
    add_conversations_to_weaviate(client, email_chains_df, "example_verba_email_chains", force=True)
    add_conversations_to_weaviate(client, slack_messages_df, "example_verba_slack_conversations", force=True)
