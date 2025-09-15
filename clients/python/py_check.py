import os
import sys
import traceback

import weaviate
from weaviate.classes.init import Timeout, Auth
from weaviate.config import AdditionalConfig


def main():
    url = os.getenv("WEAVIATE_URL", "http://weaviate:8080")
    api_key = os.getenv("WEAVIATE_API_KEY", "elysia-local-admin")

    print(f"Connecting to: {url}")
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}

    try:
        if url.startswith("https://"):
            client = weaviate.connect_to_weaviate_cloud(
                cluster_url=url,
                auth_credentials=Auth.api_key(api_key) if api_key else None,
                headers=headers,
                additional_config=AdditionalConfig(timeout=Timeout(query=60, insert=120, init=5)),
            )
        else:
            # Connect directly to local service within Docker network
            # Extract host and port from URL like http://weaviate:8080
            without_scheme = url.split("://", 1)[1]
            host_port = without_scheme.split("/", 1)[0]
            if ":" in host_port:
                host, port = host_port.split(":", 1)
                http_port = int(port)
            else:
                host = host_port
                http_port = 8080

            client = weaviate.connect_to_local(
                http_host=host,
                http_port=http_port,
                grpc_host=host,
                grpc_port=50051,
                headers=headers,
                additional_config=AdditionalConfig(timeout=Timeout(query=60, insert=120, init=5)),
            )

        print("Connected:", client.is_connected())
        try:
            meta = client.get_meta()
            print("Meta:", meta)
        except Exception:
            print("get_meta() failed, trying simple list of collections...")
            print("Collections:", [c for c in client.collections.list_all()])
        finally:
            client.close()
    except Exception as e:
        print("Error connecting to Weaviate:")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

