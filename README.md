# Facial Expression AI Server

This local service opens the camera, processes head movement and facial expressions, then sends compact control data to Unity through WebSocket. Unity does not upload camera frames to the server.

## Unity client repository

Use [Game-Control-Through-Facial-Expression](https://github.com/zaremahdii/Game-Control-Through-Facial-Expression) as the Unity client. Run this AI service locally, then configure the Unity client to connect to `ws://127.0.0.1:8000/ws`.

## Requirements

- Python 3.11
- A connected webcam
- Port `8000` available

## Run with Python

Open PowerShell in the project folder and run:

```powershell
python -m venv .venv

$venvPython = "$PWD\.venv\Scripts\python.exe"

& $venvPython -m pip install --upgrade pip
& $venvPython -m pip install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cpu
& $venvPython -m pip install -r requirements.txt
& $venvPython -m uvicorn server_with_local_webcam:app --host 127.0.0.1 --port 8000
```

If `python` is not recognized, install Python 3.11 and enable the option to add Python to `PATH` during installation.

When the server is ready, it displays `Uvicorn running on http://127.0.0.1:8000`.

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
