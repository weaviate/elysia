import weaviate
import pandas as pd
import json
import datetime
import os
import sys
import ast

from tqdm.auto import tqdm

import weaviate.classes.config as wvc
from weaviate.classes.query import Filter
from weaviate.classes.init import Auth
from weaviate.classes.config import Configure, Property, DataType
from weaviate.util import generate_uuid5  

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def create_collection(client, collection_name: str, main_vector_names: list[str], force: bool = False):

    # Create collection or delete existing one
    if client.collections.exists(collection_name):

        if force: 
            print(f"Collection {collection_name} already exists, deleting...")
            collection = client.collections.delete(collection_name)
        else:
            print(f"Collection {collection_name} already exists, loading...")
            return client.collections.get(collection_name)


    if len(main_vector_names) > 0:
        collection = client.collections.create(
            name = collection_name,
            vectorizer_config = [
                Configure.NamedVectors.text2vec_openai(
                    name = main_vector_name
                ) for main_vector_name in main_vector_names
            ],
            properties = [
                Property(
                    name = main_vector_name,
                    data_type = DataType.TEXT
                ) for main_vector_name in main_vector_names
            ]
        )
    # no main vector names, use default vectorizer
    else:
        collection = client.collections.create(
            name = collection_name,
            vectorizer_config = wvc.Configure.Vectorizer.text2vec_openai()
        )

    return collection

def add_conversations_to_weaviate(client, df: pd.DataFrame, collection_name: str, force: bool = False):

    collection = create_collection(client, collection_name, ["message_content"], force=force)

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

    collection = create_collection(client, collection_name, ["issue_content"], force=force)

    usernames = [d[len("'login': '")+1:d[len("'login': '")+1:].find("'")+len("'login': '")+1] for d in df.user]
    labels = [[r["name"] for r in ast.literal_eval(row["labels"])] for i, row in df.iterrows()]

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
                "issue_url": row["html_url"],
                "issue_author": usernames[i],
                "issue_labels": labels[i],
                "issue_state": row["state"],
                "issue_comments": row["comments"]
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

def add_ecommerce_to_weaviate(client, df: pd.DataFrame, collection_name: str, force: bool = False):

    collection = create_collection(client, collection_name, ["description"], force=force)

    # add data to collection
    for i, row in tqdm(df.iterrows(), total=len(df), desc=f"Adding {collection_name} to Weaviate"):

        try:

            data_object = row.to_dict()

            # weaviate metadata
            uuid = generate_uuid5(data_object)

            # evaluate the colours as a list
            if data_object["colors"].startswith("["):
                data_object["colors"] = eval(data_object["colors"])
            elif "," in data_object["colors"]:
                data_object["colors"] = data_object["colors"].split(",")
            elif isinstance(data_object["colors"], str):
                data_object["colors"] = [data_object["colors"]]

            if not collection.data.exists(uuid):
                collection.data.insert(
                    properties = data_object,
                    uuid = uuid
                )

        except Exception as e:
            print(f"Error adding ecommerce item {i}: {e}, continuing...")

def add_financial_contracts_to_weaviate(client, df: pd.DataFrame, collection_name: str, force: bool = False):

    collection = create_collection(client, collection_name, ["contract_text"], force=force)

    # add data to collection
    doc_id = 0
    for i, row in tqdm(df.iterrows(), total=len(df), desc=f"Adding {collection_name} to Weaviate"):

        try:
            data_object = row.to_dict()

            doc_id += 1
            data_object["doc_id"] = doc_id

            # Parse the date string and make it timezone-aware
            date = datetime.datetime.strptime(data_object["date"], "%Y-%m-%d %H:%M:%S")
            date = date.replace(tzinfo=datetime.timezone.utc)  # Add UTC timezone
            data_object["date"] = date

            # weaviate metadata
            uuid = generate_uuid5(data_object)

            if not collection.data.exists(uuid):
                collection.data.insert(
                    properties = data_object,
                    uuid = uuid
                )

        except Exception as e:
            print(f"Error adding financial contract {i}: {e}, continuing...")

def add_weather_to_weaviate(client, df: pd.DataFrame, collection_name: str, force: bool = False):

    collection = create_collection(client, collection_name, [], force=force)

    
    for i, row in tqdm(df.iterrows(), total=len(df), desc=f"Adding {collection_name} to Weaviate"):

        data_object = row.to_dict()

        uuid = generate_uuid5(data_object)

        if not collection.data.exists(uuid):
            collection.data.insert(
                properties = data_object,
                uuid = uuid
            )
        


if __name__ == "__main__":

    # connect to weaviate cloud
    client = weaviate.connect_to_weaviate_cloud(
        cluster_url = os.environ.get("WCD_URL"),
        auth_credentials = Auth.api_key(os.environ.get("WCD_API_KEY")),
        headers = {"X-OpenAI-API-Key": os.environ.get("OPENAI_API_KEY")}
    )

    # read data frames
    # github_issues_df  = pd.read_csv("../verba_github_issues.csv")
    # slack_messages_df = pd.read_csv("../verba_slack_conversations.csv")
    # email_chains_df   = pd.read_csv("../verba_email_chains.csv")
    # ecommerce_df      = pd.read_csv("../ecommerce.csv")
    # contracts_df      = pd.read_csv("../financial_contracts.csv")
    weather_df        = pd.read_csv("../weather_data.csv")


    # # add github issues data to weaviate collection
    # add_issues_to_weaviate(client, github_issues_df, "example_verba_github_issues", force=True)

    # # add ecommerce data to weaviate collection
    # add_ecommerce_to_weaviate(client, ecommerce_df, "ecommerce", force=True)
    
    # # add email and slack data to weaviate collections
    # add_conversations_to_weaviate(client, email_chains_df, "example_verba_email_chains", force=True)
    # add_conversations_to_weaviate(client, slack_messages_df, "example_verba_slack_conversations", force=True)

    # add financial contracts data to weaviate collection
    # add_financial_contracts_to_weaviate(client, contracts_df, "financial_contracts", force=True)

    # add weather data to weaviate collection
    add_weather_to_weaviate(client, weather_df, "weather", force=True)
