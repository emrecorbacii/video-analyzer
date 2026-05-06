FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

WORKDIR /app

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    ffmpeg \
    git \
    && rm -rf /var/lib/apt/lists/*

COPY . .

RUN pip3 install --upgrade pip

RUN pip3 install \
    torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

RUN pip3 install \
    transformers \
    fastapi \
    uvicorn \
    openai-whisper \
    opencv-python-headless \
    av \
    numpy

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]