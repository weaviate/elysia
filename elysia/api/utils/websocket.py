from fastapi import WebSocket, Request
from starlette.websockets import WebSocketDisconnect
from fastapi.dependencies.utils import solve_dependencies
from fastapi import Request

import psutil
import asyncio
import time

# Logging
from elysia.api.core.logging import logger

# Objects
from elysia.api.objects import Error

async def help_websocket(websocket: WebSocket, ws_route: callable):

    memory_process = psutil.Process()
    initial_memory = memory_process.memory_info().rss
    try:
        await websocket.accept()
        while True:
            try:

                # Wait for a message from the client
                logger.info(f"Memory usage before receiving: {psutil.Process().memory_info().rss / 1024 / 1024}MB")
                data = await websocket.receive_json()
                logger.info(f"Memory usage after receiving: {psutil.Process().memory_info().rss / 1024 / 1024}MB")
                
                # Check if the message is a disconnect request
                if data.get("type") == "disconnect":
                    logger.info("Received disconnect request")
                    break  # Exit the loop instead of closing the websocket here

                logger.info(f"Received message: {data}")

                # == main code ==
                # Add timing information
                start_time = time.time()
                try:
                    async with asyncio.timeout(60):  # adjust timeout as needed
                        await ws_route(data, websocket)
                except asyncio.TimeoutError:
                    logger.warning("Processing timeout - sending heartbeat")
                    await websocket.send_json({"type": "heartbeat"})
                    
                logger.info(f"Processing time: {time.time() - start_time}s")

                if time.time() % 60 < 1:  # Log every minute
                    current_memory = memory_process.memory_info().rss
                    logger.info(f"Memory usage: {(current_memory - initial_memory) / 1024 / 1024}MB")
                

            except WebSocketDisconnect:
                logger.info("WebSocket disconnected", exc_info=True)
                break  # Exit the loop on disconnect

            except RuntimeError as e:
                if (
                    "Cannot call 'receive' once a disconnect message has been received"
                    in str(e)
                ):
                    logger.info("WebSocket already disconnected")
                    break  # Exit the loop if the connection is already closed
                else:
                    raise  # Re-raise other RuntimeErrors

            except Exception as e:
                logger.error(f"Error in WebSocket: {str(e)}")
                try:
                    if data and "conversation_id" in data:
                        error = Error(text=str(e))   
                        await websocket.send_json(error.to_frontend(data["conversation_id"]))
                    else:
                        error = Error(text=str(e))
                        await websocket.send_json(error.to_frontend(""))

                except RuntimeError:
                    logger.warning(
                        "Failed to send error message, WebSocket might be closed"
                    )
                break  # Exit the loop after sending the error message

    except Exception as e:
        logger.warning(f"Closing WebSocket: {str(e)}")
    finally:
        try:
            await websocket.close()
        except RuntimeError:
            logger.info("WebSocket already closed")
