# server_with_local_webcam.py

import cv2
import torch
import torch.nn as nn
import numpy as np
import requests
import os
import time
from torchvision import models, transforms
import torch.nn.functional as F
import json
from collections import deque
import mediapipe as mp
import asyncio

# --- FastAPI Imports ---
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
import uvicorn

# --- HopeNet and GameAIService Class Definitions ---
# (These are identical to your original code. They are omitted here for brevity,
# but you should copy the full HopeNet, Bottleneck, and GameAIService classes
# into this file just as they were before.)

class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
    def forward(self, x):
        residual = x
        out = self.conv1(x); out = self.bn1(out); out = self.relu(out)
        out = self.conv2(out); out = self.bn2(out); out = self.relu(out)
        out = self.conv3(out); out = self.bn3(out)
        if self.downsample is not None: residual = self.downsample(x)
        out += residual; out = self.relu(out)
        return out

class HopeNet(nn.Module):
    def __init__(self, block, layers, num_classes):
        self.inplanes = 64
        super(HopeNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7)
        self.fc_yaw = nn.Linear(512 * block.expansion, num_classes)
        self.fc_pitch = nn.Linear(512 * block.expansion, num_classes)
        self.fc_roll = nn.Linear(512 * block.expansion, num_classes)
    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False), nn.BatchNorm2d(planes * block.expansion))
        layers = []; layers.append(block(self.inplanes, planes, stride, downsample)); self.inplanes = planes * block.expansion
        for i in range(1, blocks): layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)
    def forward(self, x):
        x = self.conv1(x); x = self.bn1(x); x = self.relu(x); x = self.maxpool(x)
        x = self.layer1(x); x = self.layer2(x); x = self.layer3(x); x = self.layer4(x)
        x = self.avgpool(x); x = x.view(x.size(0), -1)
        yaw = self.fc_yaw(x); pitch = self.fc_pitch(x); roll = self.fc_roll(x)
        return yaw, pitch, roll

class GameAIService:
    def __init__(self, fer_model_path, device='cuda'):
        self.device = device
        print(f"--- AI Service using device: {self.device} ---")
        self.face_detector = mp.solutions.face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5)
        self.fer_class_names = ['Anger', 'Happiness', 'Neutral', 'Sadness', 'Surprise']
        self._load_hpe_model()
        self._load_fer_model(fer_model_path)
        self.transform = transforms.Compose([
            transforms.ToPILImage(), transforms.Resize(224), transforms.CenterCrop(224), transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.hpe_idx_tensor = torch.FloatTensor([i for i in range(66)]).to(self.device)
        self.roll_history = deque(maxlen=5)
        self.RIGHT_THRESHOLD = -5
        self.LEFT_THRESHOLD = 5
        print("\nAI Service Initialization Complete. Ready for WebSocket connections.")

    def _load_hpe_model(self):
        self.hpe_model = HopeNet(Bottleneck, [3, 4, 6, 3], 66)
        weights_path = 'Models/hopenet_robust_alpha1.pkl'
        url = 'https://huggingface.co/hysts/Hopenet/resolve/main/orig/hopenet_robust_alpha1.pkl'
        if not os.path.exists(weights_path):
            os.makedirs('Models', exist_ok=True)
            r = requests.get(url, allow_redirects=True, timeout=30)
            r.raise_for_status()
            with open(weights_path, 'wb') as f: f.write(r.content)
        saved_state_dict = torch.load(weights_path, map_location=self.device)
        if list(saved_state_dict.keys())[0].startswith('module.'):
            saved_state_dict = {k[7:]: v for k, v in saved_state_dict.items()}
        self.hpe_model.load_state_dict(saved_state_dict, strict=False)
        self.hpe_model.to(self.device)
        self.hpe_model.eval()

    def _load_fer_model(self, model_path):
        self.fer_model = models.vit_b_16(weights=None)
        num_ftrs = self.fer_model.heads.head.in_features
        self.fer_model.heads.head = nn.Linear(num_ftrs, len(self.fer_class_names))
        self.fer_model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.fer_model.to(self.device)
        self.fer_model.eval()

    def process_image(self, image: np.ndarray):
        output = {"direction": "neutral", "emotion": "neutral"}
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.face_detector.process(image_rgb)
        if results.detections:
            detection = sorted(results.detections, key=lambda d: d.location_data.relative_bounding_box.width * d.location_data.relative_bounding_box.height, reverse=True)[0]
            bboxC = detection.location_data.relative_bounding_box
            ih, iw, _ = image.shape
            x, y, w, h = int(bboxC.xmin * iw), int(bboxC.ymin * ih), int(bboxC.width * iw), int(bboxC.height * ih)
            x1, y1, x2, y2 = x, y, x + w, y + h
            center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
            crop_size = int(max(w, h) * 1.5)
            half_crop = crop_size // 2
            x1_hpe, y1_hpe = max(0, center_x - half_crop), max(0, center_y - half_crop)
            x2_hpe, y2_hpe = min(iw, center_x + half_crop), min(ih, center_y + half_crop)
            face_crop_hpe = image[y1_hpe:y2_hpe, x1_hpe:x2_hpe]
            x1_fer, y1_fer = max(0, x1 - 10), max(0, y1 - 10)
            x2_fer, y2_fer = min(iw, x2 + 10), min(ih, y2 + 10)
            face_crop_fer = image[y1_fer:y2_fer, x1_fer:x2_fer]
            if face_crop_hpe.size > 0:
                img_tensor_hpe = self.transform(face_crop_hpe).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    _, _, roll = self.hpe_model(img_tensor_hpe)
                    roll_binned = F.softmax(roll, dim=1)
                    roll_deg = torch.sum(roll_binned * self.hpe_idx_tensor, dim=1) * 3 - 99
                    self.roll_history.append(roll_deg.item())
                    smoothed_roll = np.mean(self.roll_history)
                if smoothed_roll < self.RIGHT_THRESHOLD: output["direction"] = "right"
                elif smoothed_roll > self.LEFT_THRESHOLD: output["direction"] = "left"
            if output["direction"] == "neutral" and face_crop_fer.size > 0:
                img_tensor_fer = self.transform(face_crop_fer).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    fer_logits = self.fer_model(img_tensor_fer)
                    _, pred_idx = torch.max(fer_logits, 1)
                    output["emotion"] = self.fer_class_names[pred_idx.item()].lower()
            else:
                output["emotion"] = "neutral"
        else:
            self.roll_history.clear()
        return output

# --- FastAPI Setup ---
app = FastAPI()

@app.get("/health")
async def health():
    device = "not-loaded"

    if "ai_service" in globals():
        device = str(ai_service.device)

    return {
        "status": "ok",
        "device": device
    }

# Load the AI service on startup
@app.on_event("startup")
def load_services():
    global ai_service
    fer_model_path = 'Models/vit_best_model.pth'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    ai_service = GameAIService(fer_model_path, device=device)

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("Client connected. Starting webcam capture on server.")
    
    cap = cv2.VideoCapture(0) # Use the server's default webcam
    if not cap.isOpened():
        print("Error: Could not open webcam on server.")
        await websocket.close(code=1011, reason="Server could not access webcam.")
        return

    try:
        while True:
            # Capture frame-by-frame from the server's webcam
            ret, frame = cap.read()
            if not ret:
                print("Error: Can't receive frame from server's webcam. Closing stream.")
                break

            # Process the image using the AI service
            output_json = ai_service.process_image(frame)
            
            # Send the JSON result back to the client
            await websocket.send_text(json.dumps(output_json))
            
            # Yield control to the event loop briefly to allow for other tasks
            # and to prevent blocking. This helps keep the server responsive.
            await asyncio.sleep(0.01)

    except WebSocketDisconnect:
        print("Client disconnected.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
    finally:
        # Release the webcam when the client disconnects
        print("Releasing server webcam.")
        cap.release()