import asyncio
import websockets
import httpx
import json
from typing import Dict, Any

class APITester:
    def __init__(self):
        self.base_url = "http://localhost:8000"
        self.ws_base_url = "ws://localhost:8000"
        self.user_id = "test_user"
        self.conversation_id = "test_conversation"

    async def test_http_routes(self):
        """Test all HTTP routes"""
        async with httpx.AsyncClient() as client:
            print("\n=== Testing HTTP Routes ===")
            
            # Test health check
            print("\nTesting /api/health...")
            response = await client.get(f"{self.base_url}/api/health")
            print(f"Status: {response.status_code}")
            print(f"Response: {response.json()}")

            # Test collections endpoint
            print("\nTesting /api/collections...")
            response = await client.get(f"{self.base_url}/api/collections")
            print(f"Status: {response.status_code}")
            print(f"Response: {response.json()}")

            # Test initialize tree
            print("\nTesting /api/initialise_tree...")
            data = {
                "user_id": self.user_id,
                "conversation_id": self.conversation_id
            }
            response = await client.post(
                f"{self.base_url}/api/initialise_tree",
                json=data
            )
            print(f"Status: {response.status_code}")
            print(f"Response: {response.json()}")

            # Test NER endpoint
            print("\nTesting /api/ner...")
            ner_data = {
                "text": "Apple Inc. is located in Cupertino, California."
            }
            response = await client.post(
                f"{self.base_url}/api/ner",
                json=ner_data
            )
            print(f"Status: {response.status_code}")
            print(f"Response: {response.json()}")

    async def test_websocket_routes(self):
        """Test all WebSocket routes"""
        print("\n=== Testing WebSocket Routes ===")

        # Test query websocket
        print("\nTesting /ws/query...")
        async with websockets.connect(f"{self.ws_base_url}/ws/query") as websocket:
            # Send a test query
            query_data = {
                "user_id": self.user_id,
                "conversation_id": self.conversation_id,
                "query_id": "test_query_1",
                "query": "What is the weather like today?"
            }
            await websocket.send(json.dumps(query_data))
            
            # Wait for response with timeout
            try:
                async with asyncio.timeout(10):
                    response = await websocket.recv()
                    print(f"Received: {response}")
            except asyncio.TimeoutError:
                print("Timeout waiting for response")

    async def run_all_tests(self):
        """Run all tests"""
        try:
            await self.test_http_routes()
            await self.test_websocket_routes()
        except Exception as e:
            print(f"Error during tests: {str(e)}")

def main():
    """Main entry point"""
    tester = APITester()
    asyncio.run(tester.run_all_tests())

if __name__ == "__main__":
    main()