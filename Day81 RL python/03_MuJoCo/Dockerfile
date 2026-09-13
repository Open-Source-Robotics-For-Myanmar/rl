FROM pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime

# System dependencies for MuJoCo rendering (Hopper-v4/HalfCheetah-v4/Ant-v4) + Gymnasium
ENV DEBIAN_FRONTEND=noninteractive
ARG PIP_DEFAULT_TIMEOUT=300
ARG PIP_RETRIES=10
ENV PIP_DEFAULT_TIMEOUT=${PIP_DEFAULT_TIMEOUT} \
    PIP_RETRIES=${PIP_RETRIES}
# libosmesa6-dev/libgl1-mesa-glx/libglfw3/patchelf တွေက MuJoCo rendering အတွက်လိုအပ်ပါတယ်
RUN apt-get update && apt-get install -y \
    build-essential \
    ffmpeg \
    git \
    python3-opengl \
    libosmesa6-dev \
    libgl1-mesa-glx \
    libglfw3 \
    patchelf \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# Install RL libraries (Gymnasium[mujoco] + SB3) for Hopper-v4/HalfCheetah-v4/Ant-v4
# atari/box2d/pybullet မလိုအပ်တာကို ဖြုတ်ပြီး mujoco အတွက်တာကိုပဲတင်သွင်ပါ
RUN pip install --no-cache-dir \
    gymnasium[mujoco]==0.29.1 \
    mujoco==2.3.7 \
    stable-baselines3==2.3.2 \
    tensorboard \
    moviepy
