# Facial Expression AI Server

This local service opens the camera, processes head movement and facial expressions, then sends compact control data to Unity through WebSocket. Unity does not upload camera frames to the server.

## Unity client repository

Use [Game-Control-Through-Facial-Expression](https://github.com/zaremahdii/Game-Control-Through-Facial-Expression) as the Unity client. Run this AI service locally, then configure the Unity client to connect to `ws://127.0.0.1:8000/ws`.

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

## Unity camera preview stream

The same WebSocket also sends a lightweight JPEG camera preview to Unity. Python remains the only application that opens the physical webcam.

```text
Webcam
  ↓
Python camera capture
  ├── AI control JSON
  └── JPEG preview frames
          ↓
        Unity
```

Preview frames are resized to `320x180`, JPEG encoded with quality `70`, and limited to `10 FPS`. This keeps the preview lightweight while preserving the control messages on the same local WebSocket.

Restart the Python server after changing `server_with_local_webcam.py`:

```powershell
& $venvPython -m uvicorn server_with_local_webcam:app --host 127.0.0.1 --port 8000
```

## Required models

Keep these model files in the `Models` directory:

```text
Models/hopenet_robust_alpha1.pkl
Models/vit_best_model.pth
```

## Webcam access in Docker Desktop on Windows

Docker Desktop on Windows may not expose the host webcam directly to a Linux container. If the server starts but cannot open the camera, connect the camera to the Docker VM through USB/IP or run the container in a Linux environment that has direct camera-device access.

For an integrated laptop webcam, direct Python execution on Windows is the recommended local development option.

## Run directly with Python on Windows

Stop the Docker container before starting the local server so port `8000` is available.

This project uses Python `3.11`. If the `py` launcher is not available, use the Python executable directly:

```powershell
cd "E:\facial expression\Game-Control-Through-Facial-Expression-AI"

& "C:\Users\Surena\AppData\Local\Programs\Python\Python311\python.exe" -m venv .venv

$venvPython = "$PWD\.venv\Scripts\python.exe"

& $venvPython -m pip install --upgrade pip

& $venvPython -m pip install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cpu

& $venvPython -m pip install -r requirements.txt

& $venvPython -m uvicorn server_with_local_webcam:app --host 127.0.0.1 --port 8000
```

When the server is ready, it displays `Uvicorn running on http://127.0.0.1:8000`. Start the Unity Level1 scene after that message appears.
