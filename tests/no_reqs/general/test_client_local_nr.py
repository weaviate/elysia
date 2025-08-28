import pytest

from elysia.util.client import ClientManager


class _DummySyncClient:
    def __init__(self):
        self._connected = True

    def is_connected(self):
        return self._connected

    def connect(self):
        self._connected = True

    def close(self):
        self._connected = False


class _DummyAsyncClient:
    def __init__(self):
        self._connected = False

    def is_connected(self):
        return self._connected

    async def connect(self):
        self._connected = True

    async def close(self):
        self._connected = False


@pytest.mark.asyncio
async def test_local_sync_client_connects(monkeypatch):
    calls = {}

    def fake_connect_to_local(host, port, grpc_port, auth_credentials=None, headers=None, skip_init_checks=True):
        print(f"fake_connect_to_local called with host: {host}, port: {port}, grpc_port: {grpc_port}, auth_credentials: {auth_credentials}, headers: {headers}, skip_init_checks: {skip_init_checks}")
        calls["host"] = host
        calls["port"] = port
        calls["grpc_port"] = grpc_port
        calls["auth_credentials"] = auth_credentials
        calls["headers"] = headers or {}
        print(f"calls: {calls}")
        return _DummySyncClient()

    # Patch weaviate.connect_to_local
    import weaviate

    monkeypatch.setattr(weaviate, "connect_to_local", fake_connect_to_local)

    # Provide class-level fallbacks for the uppercase attributes used internally
    ClientManager.LOCAL_WEAVIATE_PORT = 8080
    ClientManager.LOCAL_WEAVIATE_GRPC_PORT = 50051

    client_manager = ClientManager(
        wcd_url="localhost",
        wcd_api_key="",
        weaviate_is_local=True,
        local_weaviate_port=8080,
        local_weaviate_grpc_port=50051,
    )

    try:
        assert client_manager.is_client is True
        assert isinstance(client_manager.client, _DummySyncClient)

        # Ensure correct local connection args were used
        assert calls["host"] == "localhost"
        assert calls["port"] == 8080
        assert calls["grpc_port"] == 50051

        # Context manager increments/decrements usage counter
        with client_manager.connect_to_client() as client:
            assert client_manager.sync_in_use_counter == 1
            assert isinstance(client, _DummySyncClient)
        assert client_manager.sync_in_use_counter == 0
    finally:
        await client_manager.close_clients()


@pytest.mark.asyncio
async def test_local_async_client_connects(monkeypatch):
    def fake_connect_to_local(host, port, grpc_port, auth_credentials=None, headers=None, skip_init_checks=True):
        return _DummySyncClient()

    def fake_use_async_with_local(host, port, grpc_port, auth_credentials=None, headers=None, skip_init_checks=True):
        # Real weaviate.use_async_with_local returns an async client object synchronously
        return _DummyAsyncClient()

    import weaviate

    monkeypatch.setattr(weaviate, "connect_to_local", fake_connect_to_local)
    monkeypatch.setattr(weaviate, "use_async_with_local", fake_use_async_with_local)

    ClientManager.LOCAL_WEAVIATE_PORT = 8080
    ClientManager.LOCAL_WEAVIATE_GRPC_PORT = 50051

    client_manager = ClientManager(
        wcd_url="localhost",
        wcd_api_key="",
        weaviate_is_local=True,
        local_weaviate_port=8080,
        local_weaviate_grpc_port=50051,
    )

    try:
        async with client_manager.connect_to_async_client() as aclient:
            assert client_manager.async_in_use_counter == 1
            assert isinstance(aclient, _DummyAsyncClient)
        assert client_manager.async_in_use_counter == 0
    finally:
        await client_manager.close_clients()



@pytest.mark.asyncio
async def test_local_weaviate_seed_and_query(monkeypatch):
    """
    Creates a simple collection and inserts dummy data using the official weaviate client.
    Skips if local Weaviate is not running.
    """
    try:
        import weaviate
        from weaviate.classes.config import Property, DataType, Configure
    except Exception:
        pytest.skip("weaviate client not installed")

    HOST = "localhost"
    HTTP_PORT = 8080
    GRPC_PORT = 50051
    COLLECTION = "DEMO_ITEMS_TEST"

    try:
        client = weaviate.connect_to_local(
            host=HOST,
            port=HTTP_PORT,
            grpc_port=GRPC_PORT,
            skip_init_checks=True,
        )
    except Exception:
        pytest.skip("Local Weaviate not reachable on localhost:8080")

    try:
        if client.collections.exists(COLLECTION):
            coll = client.collections.get(COLLECTION)
        else:
            coll = client.collections.create(
                COLLECTION,
                vectorizer_config=Configure.Vectorizer.none(),
                inverted_index_config=Configure.inverted_index(index_timestamps=True),
                properties=[
                    Property(name="name", data_type=DataType.TEXT),
                    Property(name="age", data_type=DataType.INT),
                ],
            )

        items = [
            {"name": "Alice", "age": 30},
            {"name": "Bob", "age": 25},
            {"name": "Charlie", "age": 35},
        ]
        for obj in items:
            coll.data.insert(obj)

        total = coll.aggregate.over_all(total_count=True).total_count
        assert total >= len(items)

        res = coll.query.fetch_objects(limit=3)
        assert len(res.objects) > 0
        assert all("name" in o.properties for o in res.objects)
    finally:
        client.close()
