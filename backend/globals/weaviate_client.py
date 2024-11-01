import weaviate
from weaviate.classes.init import Auth
import os

client = weaviate.connect_to_weaviate_cloud(
    cluster_url = os.environ.get("WCD_URL"),
    auth_credentials = Auth.api_key(os.environ.get("WCD_API_KEY")),
    headers = {"X-OpenAI-API-Key": os.environ.get("OPENAI_API_KEY")}
)