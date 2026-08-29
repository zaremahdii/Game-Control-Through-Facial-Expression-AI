# simple_listener_client.py

import asyncio
import websockets
import json

async def listen_to_server():
    """
    Connects to the server and prints the messages it receives.
    """
    uri = "ws://localhost:8000/ws"
    
    try:
        async with websockets.connect(uri) as websocket:
            print(f"Successfully connected to {uri}")
            print("Listening for messages from the server... (Press Ctrl+C to stop)")
            
            async for message in websocket:
                data = json.loads(message)
                # Print the received data in a clean format
                print(f"Received: Direction={data.get('direction', 'N/A')}, Emotion={data.get('emotion', 'N/A')}")

    except (websockets.exceptions.ConnectionClosedError, ConnectionRefusedError) as e:
        print(f"Connection failed: {e}. Is the server running?")
    except KeyboardInterrupt:
        print("\nClient stopped by user.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    asyncio.run(listen_to_server())