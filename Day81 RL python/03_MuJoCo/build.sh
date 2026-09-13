#!/usr/bin/env bash
set -euo pipefail

# Build the MuJoCo (Hopper-v4/HalfCheetah-v4/Ant-v4) docker image from this directory.
IMAGE_NAME="romrobotics/rl-env:cuda12_torch2_3_0_mujoco_v4"

# script ခဲတည်ရှိတဲ့ directory (Dockerfile ရှိသဲ့နပ်း) ကိုပဲ build ခံ run ပါ
cd "$(dirname "${BASH_SOURCE[0]}")"
docker build -t "$IMAGE_NAME" .

echo "Built $IMAGE_NAME"
