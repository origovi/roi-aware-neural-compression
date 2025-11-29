FROM nvidia/cuda:12.3.0-devel-ubuntu22.04

WORKDIR /workspace

RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    curl \
    gnupg \
    git \
    ca-certificates \
    python3-pip \
    python3-dev \
    libopenblas-dev \
    libomp-dev \
    libglib2.0-0 \
    libgl1-mesa-glx