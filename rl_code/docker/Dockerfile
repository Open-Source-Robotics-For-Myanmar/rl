FROM pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime

# System dependencies for Rendering and Gymnasium
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y \
    ffmpeg \
    freeglut3-dev \
    swig \
    git \
    python3-opengl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# Install RL libraries (Gymnasium + SB3)
RUN pip install --no-cache-dir \
    gymnasium[classic-control,atari]==0.29.1 \
    stable-baselines3==2.3.2 \
    tensorboard \
    moviepy