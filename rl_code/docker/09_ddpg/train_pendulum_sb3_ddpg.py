import argparse
from pathlib import Path

import numpy as np
from stable_baselines3 import DDPG
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.noise import NormalActionNoise

TOTAL_TIMESTEPS = 50_000  # training loop ကို total ဘယ်နှစ် environment step run မလဲ
LEARNING_RATE = 1e-3
GAMMA = 0.99
TAU = 0.005
EXPLORATION_NOISE = 0.1
SEED = 1


# command-line arguments (learning rate, gamma, tau, exploration noise, seed) တွေကို parse လုပ်တယ်
def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Train SB3 DDPG on Pendulum.")
	parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE)
	parser.add_argument("--gamma", type=float, default=GAMMA)
	parser.add_argument("--tau", type=float, default=TAU)
	parser.add_argument("--exploration-noise", type=float, default=EXPLORATION_NOISE)
	parser.add_argument("--seed", type=int, default=SEED)
	return parser.parse_args()


# Stable-Baselines3 ရဲ့ built-in DDPG (deterministic actor-critic + Gaussian action noise) ကို train ပေးတယ်
def train(args: argparse.Namespace) -> None:
	env = make_vec_env("Pendulum-v1", n_envs=1, seed=args.seed)
	action_size = env.action_space.shape[-1]
	action_noise = NormalActionNoise(
		mean=np.zeros(action_size),
		sigma=args.exploration_noise * np.ones(action_size),
	)
	model = DDPG(
		"MlpPolicy",
		env,
		learning_rate=args.learning_rate,
		gamma=args.gamma,
		tau=args.tau,
		action_noise=action_noise,
		seed=args.seed,
		verbose=1,
		tensorboard_log=str(Path("sb3_ddpg_pendulum_tensorboard")),
	)
	model.learn(total_timesteps=TOTAL_TIMESTEPS)

	model_path = Path("rom_sb3_ddpg_pendulum")
	model.save(model_path)
	print(f"Training complete. Model saved to {model_path}")
	env.close()


if __name__ == "__main__":
	train(parse_args())
