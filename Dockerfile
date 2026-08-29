FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --upgrade pip

RUN pip install \
    torch==2.1.2 \
    torchvision==0.16.2 \
    --index-url https://download.pytorch.org/whl/cpu

RUN pip install -r requirements.txt

COPY server_with_local_webcam.py .
COPY Models ./Models

EXPOSE 8000

CMD ["uvicorn", "server_with_local_webcam:app", "--host", "0.0.0.0", "--port", "8000"]