import argparse
from pathlib import Path

import numpy as np
from stable_baselines3 import TD3
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.noise import NormalActionNoise

ENV_ID = "Hopper-v4"
TOTAL_TIMESTEPS = 1_000_000
LEARNING_RATE = 3e-4
BUFFER_SIZE = 1_000_000
GAMMA = 0.99
TAU = 0.005
BATCH_SIZE = 256
EXPLORATION_NOISE = 0.1
POLICY_NOISE = 0.2
NOISE_CLIP = 0.5
LEARNING_STARTS = 25_000
POLICY_DELAY = 2
SEED = 1
MODEL_PATH = Path("rom_sb3_td3_hopper")
TENSORBOARD_LOG_DIR = Path("sb3_td3_hopper_tensorboard")


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Train SB3 TD3 on Hopper.")
	parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE)
	parser.add_argument("--gamma", type=float, default=GAMMA)
	parser.add_argument("--tau", type=float, default=TAU)
	parser.add_argument("--seed", type=int, default=SEED)
	return parser.parse_args()


def train(args: argparse.Namespace) -> None:
	env = make_vec_env(ENV_ID, n_envs=1, seed=args.seed)
	action_size = env.action_space.shape[-1]
	model = TD3(
		"MlpPolicy", env, learning_rate=args.learning_rate, buffer_size=BUFFER_SIZE,
		learning_starts=LEARNING_STARTS, batch_size=BATCH_SIZE, tau=args.tau,
		gamma=args.gamma, action_noise=NormalActionNoise(np.zeros(action_size), EXPLORATION_NOISE * np.ones(action_size)),
		policy_delay=POLICY_DELAY, target_policy_noise=POLICY_NOISE, target_noise_clip=NOISE_CLIP,
		seed=args.seed, verbose=1, tensorboard_log=str(TENSORBOARD_LOG_DIR),
	)
	model.learn(total_timesteps=TOTAL_TIMESTEPS)
	model.save(MODEL_PATH)
	print(f"Training complete. Model saved to {MODEL_PATH}")
	env.close()


if __name__ == "__main__":
	train(parse_args())