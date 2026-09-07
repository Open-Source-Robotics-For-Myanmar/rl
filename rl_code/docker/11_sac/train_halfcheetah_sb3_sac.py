import argparse
from pathlib import Path

import pybullet_envs_gymnasium  # noqa: F401  (registers HalfCheetahBulletEnv-v0)
from stable_baselines3 import SAC
from stable_baselines3.common.env_util import make_vec_env

ENV_ID = "HalfCheetahBulletEnv-v0"
TOTAL_TIMESTEPS = 1_000_000
LEARNING_RATE = 3e-4
BUFFER_SIZE = 1_000_000
GAMMA = 0.99
TAU = 0.005
BATCH_SIZE = 256
LEARNING_STARTS = 5_000
SEED = 1
MODEL_PATH = Path("rom_sb3_sac_halfcheetah")
TENSORBOARD_LOG_DIR = Path("sb3_sac_halfcheetah_tensorboard")


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Train SB3 SAC on HalfCheetahBulletEnv-v0.")
	parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE)
	parser.add_argument("--gamma", type=float, default=GAMMA)
	parser.add_argument("--tau", type=float, default=TAU)
	parser.add_argument("--seed", type=int, default=SEED)
	return parser.parse_args()


def train(args: argparse.Namespace) -> None:
	env = make_vec_env(ENV_ID, n_envs=1, seed=args.seed)
	model = SAC(
		"MlpPolicy", env, learning_rate=args.learning_rate, buffer_size=BUFFER_SIZE,
		learning_starts=LEARNING_STARTS, batch_size=BATCH_SIZE, tau=args.tau,
		gamma=args.gamma, ent_coef="auto",
		seed=args.seed, verbose=1, tensorboard_log=str(TENSORBOARD_LOG_DIR),
	)
	model.learn(total_timesteps=TOTAL_TIMESTEPS)
	model.save(MODEL_PATH)
	print(f"Training complete. Model saved to {MODEL_PATH}")
	env.close()


if __name__ == "__main__":
	train(parse_args())
