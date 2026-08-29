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
import mediapipe as mp
import asyncio

# --- FastAPI Imports ---
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
import uvicorn


# --- HopeNet and GameAIService Class Definitions ---

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()

        self.conv1 = nn.Conv2d(
            inplanes,
            planes,
            kernel_size=1,
            bias=False
        )

        self.bn1 = nn.BatchNorm2d(planes)

        self.conv2 = nn.Conv2d(
            planes,
            planes,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False
        )

        self.bn2 = nn.BatchNorm2d(planes)

        self.conv3 = nn.Conv2d(
            planes,
            planes * 4,
            kernel_size=1,
            bias=False
        )

        self.bn3 = nn.BatchNorm2d(
            planes * 4
        )

        self.relu = nn.ReLU(
            inplace=True
        )

        self.downsample = downsample
        self.stride = stride


    def forward(self, x):

        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out



class HopeNet(nn.Module):

    def __init__(
        self,
        block,
        layers,
        num_classes
    ):

        self.inplanes = 64

        super(HopeNet, self).__init__()

        self.conv1 = nn.Conv2d(
            3,
            64,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False
        )

        self.bn1 = nn.BatchNorm2d(64)

        self.relu = nn.ReLU(
            inplace=True
        )

        self.maxpool = nn.MaxPool2d(
            kernel_size=3,
            stride=2,
            padding=1
        )

        self.layer1 = self._make_layer(
            block,
            64,
            layers[0]
        )

        self.layer2 = self._make_layer(
            block,
            128,
            layers[1],
            stride=2
        )

        self.layer3 = self._make_layer(
            block,
            256,
            layers[2],
            stride=2
        )

        self.layer4 = self._make_layer(
            block,
            512,
            layers[3],
            stride=2
        )

        self.avgpool = nn.AvgPool2d(7)

        self.fc_yaw = nn.Linear(
            512 * block.expansion,
            num_classes
        )

        self.fc_pitch = nn.Linear(
            512 * block.expansion,
            num_classes
        )

        self.fc_roll = nn.Linear(
            512 * block.expansion,
            num_classes
        )


    def _make_layer(
        self,
        block,
        planes,
        blocks,
        stride=1
    ):

        downsample = None

        if (
            stride != 1
            or self.inplanes
            != planes * block.expansion
        ):
            downsample = nn.Sequential(

                nn.Conv2d(
                    self.inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False
                ),

                nn.BatchNorm2d(
                    planes * block.expansion
                )
            )

        layers = []

        layers.append(
            block(
                self.inplanes,
                planes,
                stride,
                downsample
            )
        )

        self.inplanes = (
            planes * block.expansion
        )

        for i in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes
                )
            )

        return nn.Sequential(*layers)


    def forward(self, x):

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)

        x = x.view(
            x.size(0),
            -1
        )

        yaw = self.fc_yaw(x)
        pitch = self.fc_pitch(x)
        roll = self.fc_roll(x)

        return yaw, pitch, roll



class GameAIService:

    def __init__(
        self,
        fer_model_path,
        device='cuda'
    ):

        self.device = device

        print(
            f"--- AI Service using device: "
            f"{self.device} ---"
        )

        self.face_detector = (
            mp.solutions.face_detection
            .FaceDetection(
                model_selection=0,
                min_detection_confidence=0.5
            )
        )

        self.fer_class_names = [
            'Anger',
            'Happiness',
            'Neutral',
            'Sadness',
            'Surprise'
        ]

        self._load_hpe_model()

        self._load_fer_model(
            fer_model_path
        )

        self.transform = transforms.Compose([
            transforms.ToPILImage(),

            transforms.Resize(224),

            transforms.CenterCrop(224),

            transforms.ToTensor(),

            transforms.Normalize(
                mean=[
                    0.485,
                    0.456,
                    0.406
                ],
                std=[
                    0.229,
                    0.224,
                    0.225
                ]
            )
        ])

        self.hpe_idx_tensor = (
            torch.FloatTensor(
                [
                    i
                    for i in range(66)
                ]
            )
            .to(self.device)
        )

        # =====================================================
        # MODIFIED:
        # Fast EMA filtering instead of 5-frame moving average.
        # =====================================================

        self.filtered_roll = None

        # New measurements receive 80% of the weight.
        # This gives fast response while still removing jitter.
        self.ROLL_EMA_ALPHA = 0.80

        # Hysteresis thresholds:
        #
        # Enter steering when roll exceeds 5 degrees.
        # Exit steering when roll returns inside 3 degrees.
        #
        # Using two thresholds prevents rapid left/neutral/right
        # switching around the boundary.
        self.TILT_ENTER_THRESHOLD = 5.0
        self.TILT_EXIT_THRESHOLD = 3.0

        self.direction_state = "neutral"

        print(
            "\nAI Service Initialization Complete. "
            "Ready for WebSocket connections."
        )



    def _load_hpe_model(self):

        self.hpe_model = HopeNet(
            Bottleneck,
            [3, 4, 6, 3],
            66
        )

        weights_path = (
            'Models/'
            'hopenet_robust_alpha1.pkl'
        )

        url = (
            'https://huggingface.co/'
            'hysts/Hopenet/resolve/main/orig/'
            'hopenet_robust_alpha1.pkl'
        )

        if not os.path.exists(
            weights_path
        ):

            os.makedirs(
                'Models',
                exist_ok=True
            )

            r = requests.get(
                url,
                allow_redirects=True,
                timeout=30
            )

            r.raise_for_status()

            with open(
                weights_path,
                'wb'
            ) as f:

                f.write(
                    r.content
                )

        saved_state_dict = torch.load(
            weights_path,
            map_location=self.device
        )

        if (
            list(
                saved_state_dict.keys()
            )[0]
            .startswith('module.')
        ):

            saved_state_dict = {
                k[7:]: v
                for k, v
                in saved_state_dict.items()
            }

        self.hpe_model.load_state_dict(
            saved_state_dict,
            strict=False
        )

        self.hpe_model.to(
            self.device
        )

        self.hpe_model.eval()



    def _load_fer_model(
        self,
        model_path
    ):

        self.fer_model = (
            models.vit_b_16(
                weights=None
            )
        )

        num_ftrs = (
            self.fer_model
            .heads
            .head
            .in_features
        )

        self.fer_model.heads.head = (
            nn.Linear(
                num_ftrs,
                len(
                    self.fer_class_names
                )
            )
        )

        self.fer_model.load_state_dict(
            torch.load(
                model_path,
                map_location=self.device
            )
        )

        self.fer_model.to(
            self.device
        )

        self.fer_model.eval()



    def process_image(
        self,
        image: np.ndarray
    ):

        output = {
            "direction": "neutral",
            "emotion": "neutral"
        }

        image_rgb = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB
        )

        results = (
            self.face_detector
            .process(image_rgb)
        )

        if results.detections:

            detection = sorted(
                results.detections,

                key=lambda d:
                (
                    d.location_data
                    .relative_bounding_box
                    .width
                    *
                    d.location_data
                    .relative_bounding_box
                    .height
                ),

                reverse=True
            )[0]

            bboxC = (
                detection
                .location_data
                .relative_bounding_box
            )

            ih, iw, _ = image.shape

            x = int(
                bboxC.xmin * iw
            )

            y = int(
                bboxC.ymin * ih
            )

            w = int(
                bboxC.width * iw
            )

            h = int(
                bboxC.height * ih
            )

            x1 = x
            y1 = y

            x2 = x + w
            y2 = y + h

            center_x = (
                x1 + x2
            ) // 2

            center_y = (
                y1 + y2
            ) // 2

            crop_size = int(
                max(w, h) * 1.5
            )

            half_crop = (
                crop_size // 2
            )

            x1_hpe = max(
                0,
                center_x
                - half_crop
            )

            y1_hpe = max(
                0,
                center_y
                - half_crop
            )

            x2_hpe = min(
                iw,
                center_x
                + half_crop
            )

            y2_hpe = min(
                ih,
                center_y
                + half_crop
            )

            face_crop_hpe = image[
                y1_hpe:y2_hpe,
                x1_hpe:x2_hpe
            ]

            x1_fer = max(
                0,
                x1 - 10
            )

            y1_fer = max(
                0,
                y1 - 10
            )

            x2_fer = min(
                iw,
                x2 + 10
            )

            y2_fer = min(
                ih,
                y2 + 10
            )

            face_crop_fer = image[
                y1_fer:y2_fer,
                x1_fer:x2_fer
            ]


            # =================================================
            # HEAD POSE ESTIMATION
            # =================================================

            if face_crop_hpe.size > 0:

                img_tensor_hpe = (
                    self.transform(
                        face_crop_hpe
                    )
                    .unsqueeze(0)
                    .to(self.device)
                )

                # =============================================
                # MODIFIED:
                # inference_mode is optimized for inference.
                # =============================================

                with torch.inference_mode():

                    _, _, roll = (
                        self.hpe_model(
                            img_tensor_hpe
                        )
                    )

                    roll_binned = F.softmax(
                        roll,
                        dim=1
                    )

                    roll_deg = (
                        torch.sum(
                            roll_binned
                            * self.hpe_idx_tensor,
                            dim=1
                        )
                        * 3
                        - 99
                    )

                    current_roll = (
                        roll_deg.item()
                    )


                # =============================================
                # MODIFIED:
                # Fast Exponential Moving Average.
                # =============================================

                if self.filtered_roll is None:

                    self.filtered_roll = (
                        current_roll
                    )

                else:

                    self.filtered_roll = (
                        self.ROLL_EMA_ALPHA
                        * current_roll

                        +

                        (
                            1.0
                            - self.ROLL_EMA_ALPHA
                        )
                        * self.filtered_roll
                    )

                filtered_roll = (
                    self.filtered_roll
                )


                # =============================================
                # MODIFIED:
                # Hysteresis-based directional state machine.
                # =============================================

                if (
                    self.direction_state
                    == "neutral"
                ):

                    if (
                        filtered_roll
                        <
                        -self.TILT_ENTER_THRESHOLD
                    ):

                        self.direction_state = (
                            "right"
                        )

                    elif (
                        filtered_roll
                        >
                        self.TILT_ENTER_THRESHOLD
                    ):

                        self.direction_state = (
                            "left"
                        )


                elif (
                    self.direction_state
                    == "right"
                ):

                    # Allows immediate transition from
                    # right to left if the user moves
                    # strongly across the opposite threshold.
                    if (
                        filtered_roll
                        >
                        self.TILT_ENTER_THRESHOLD
                    ):

                        self.direction_state = (
                            "left"
                        )

                    elif (
                        filtered_roll
                        >
                        -self.TILT_EXIT_THRESHOLD
                    ):

                        self.direction_state = (
                            "neutral"
                        )


                elif (
                    self.direction_state
                    == "left"
                ):

                    # Allows immediate transition from
                    # left to right if the user moves
                    # strongly across the opposite threshold.
                    if (
                        filtered_roll
                        <
                        -self.TILT_ENTER_THRESHOLD
                    ):

                        self.direction_state = (
                            "right"
                        )

                    elif (
                        filtered_roll
                        <
                        self.TILT_EXIT_THRESHOLD
                    ):

                        self.direction_state = (
                            "neutral"
                        )


                output["direction"] = (
                    self.direction_state
                )


            # =================================================
            # FACIAL EXPRESSION RECOGNITION
            # =================================================
            #
            # FER remains exactly hierarchical:
            #
            # Moving head:
            #     HPE active
            #     FER suppressed
            #
            # Neutral head:
            #     FER active
            #
            # The difference is that returning to neutral no
            # longer waits for five previous roll measurements.
            # =================================================

            if (
                output["direction"]
                == "neutral"
                and
                face_crop_fer.size > 0
            ):

                img_tensor_fer = (
                    self.transform(
                        face_crop_fer
                    )
                    .unsqueeze(0)
                    .to(self.device)
                )

                # =============================================
                # MODIFIED:
                # optimized inference context
                # =============================================

                with torch.inference_mode():

                    fer_logits = (
                        self.fer_model(
                            img_tensor_fer
                        )
                    )

                    pred_idx = torch.argmax(
                        fer_logits,
                        dim=1
                    )

                output["emotion"] = (
                    self.fer_class_names[
                        pred_idx.item()
                    ]
                    .lower()
                )

            else:

                output["emotion"] = (
                    "neutral"
                )


        else:

            # =================================================
            # MODIFIED:
            # Reset EMA/state instead of clearing deque.
            # =================================================

            self.filtered_roll = None
            self.direction_state = (
                "neutral"
            )


        return output



# --- FastAPI Setup ---

app = FastAPI()


# Load the AI service on startup
@app.on_event("startup")
def load_services():

    global ai_service

    fer_model_path = (
        'Models/'
        'vit_best_model.pth'
    )

    device = (
        'cuda'
        if torch.cuda.is_available()
        else 'cpu'
    )

    ai_service = GameAIService(
        fer_model_path,
        device=device
    )



@app.websocket("/ws")
async def websocket_endpoint(
    websocket: WebSocket
):

    await websocket.accept()

    print(
        "Client connected. "
        "Starting webcam capture on server."
    )

    cap = cv2.VideoCapture(0)

    preview_last_sent = 0.0

    if not cap.isOpened():

        print(
            "Error: Could not open "
            "webcam on server."
        )

        await websocket.close(
            code=1011,
            reason=(
                "Server could not "
                "access webcam."
            )
        )

        return


    try:

        while True:

            # Capture frame-by-frame
            # from the server's webcam

            ret, frame = cap.read()

            if not ret:

                print(
                    "Error: Can't receive frame "
                    "from server's webcam. "
                    "Closing stream."
                )

                break

            preview_now = time.monotonic()

            if preview_now - preview_last_sent >= 0.1:

                preview_frame = cv2.resize(
                    frame,
                    (320, 180)
                )

                preview_encoded, preview_jpeg = cv2.imencode(
                    '.jpg',
                    preview_frame,
                    [
                        cv2.IMWRITE_JPEG_QUALITY,
                        70
                    ]
                )

                if preview_encoded:

                    await websocket.send_bytes(
                        preview_jpeg.tobytes()
                    )

                preview_last_sent = preview_now


            # Process the image using
            # the AI service

            output_json = (
                ai_service
                .process_image(frame)
            )


            # Send the JSON result
            # back to the client

            await websocket.send_text(
                json.dumps(
                    output_json
                )
            )


            # Yield control to the
            # event loop briefly

            await asyncio.sleep(
                0.01
            )


    except WebSocketDisconnect:

        print(
            "Client disconnected."
        )


    except Exception as e:

        print(
            f"An unexpected error "
            f"occurred: {e}"
        )


    finally:

        print(
            "Releasing server webcam."
        )

        cap.release()
