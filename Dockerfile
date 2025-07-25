# Use the latest CUDA 12 runtime as base image
FROM nvidia/cuda:12.3.1-devel-ubuntu22.04

# Dockerfile
ARG HF_TOKEN
ENV HUGGINGFACE_HUB_TOKEN=$HF_TOKEN

# Set the working directory in the container
WORKDIR /app

# 필수 시스템 패키지 설치
RUN apt-get update && apt-get install -y \
    ffmpeg \
    build-essential \
    python3 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# pip 최신화
RUN python3 -m pip install --upgrade pip

# 코드 복사
COPY . /app

# RUN python3 model_downloader.py

# requirements 설치
RUN pip install -r repo_requirements.txt

# PyTorch 설치 (CUDA 12.1용)
RUN pip install torch==2.5.1+cu121 torchvision==0.20.1+cu121 torchaudio==2.5.1+cu121 \
    --index-url https://download.pytorch.org/whl/cu121

# 서버리스 entrypoint 지정
ENTRYPOINT ["python3", "handler.py"]
