# Facial Expression AI Server

This local service opens the camera, processes head movement and facial expressions, then sends compact control data to Unity through WebSocket. Unity does not upload camera frames to the server.

## Requirements

- Docker Desktop running
- A connected webcam
- Port `8000` available

## Build and run with Docker

Open PowerShell in this folder:

```powershell
cd "E:\facial expression\Game-Control-Through-Facial-Expression-AI"
docker build -t facial-expression-ai:latest .
docker run --rm --name facial-expression-ai -p 8000:8000 facial-expression-ai:latest
```

Check that the server is available:

```powershell
Invoke-RestMethod http://127.0.0.1:8000/health
```

## Unity connection

Unity connects to this WebSocket endpoint:

```text
ws://127.0.0.1:8000/ws
```

The server sends messages in this format:

```json
{
  "direction": "left",
  "emotion": "neutral"
}
```

`direction` can be `left`, `right`, or `neutral`. `emotion` can be `anger`, `happiness`, `neutral`, `sadness`, or `surprise`.

## Required models

Keep these model files in the `Models` directory:

```text
Models/hopenet_robust_alpha1.pkl
Models/vit_best_model.pth
```

## Webcam access in Docker Desktop on Windows

Docker Desktop on Windows may not expose the host webcam directly to a Linux container. If the server starts but cannot open the camera, connect the camera to the Docker VM through USB/IP or run the container in a Linux environment that has direct camera-device access.
