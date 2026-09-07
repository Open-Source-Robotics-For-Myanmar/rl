import argparse
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

ENV_ID = "LunarLander-v2"
TOTAL_TIMESTEPS = 1_000_000
NUM_ENVS = 8
NUM_STEPS = 128
BATCH_SIZE = 64
LEARNING_RATE = 2.5e-4
GAMMA = 0.99
GAE_LAMBDA = 0.95
UPDATE_EPOCHS = 4
CLIP_RANGE = 0.2
ENTROPY_COEF = 0.01
VALUE_LOSS_COEF = 0.5
MAX_GRAD_NORM = 0.5
SEED = 1
MODEL_PATH = Path("rom_sb3_ppo_lunarlander")
TENSORBOARD_LOG_DIR = Path("sb3_ppo_lunarlander_tensorboard")


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Train SB3 PPO on LunarLander.")
	parser.add_argument("--num-steps", type=int, default=NUM_STEPS)
	parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE)
	parser.add_argument("--gamma", type=float, default=GAMMA)
	parser.add_argument("--gae-lambda", type=float, default=GAE_LAMBDA)
	parser.add_argument("--clip-range", type=float, default=CLIP_RANGE)
	parser.add_argument("--n-envs", type=int, default=NUM_ENVS)
	parser.add_argument("--seed", type=int, default=SEED)
	return parser.parse_args()


def train(args: argparse.Namespace) -> None:
	env = make_vec_env(ENV_ID, n_envs=args.n_envs, seed=args.seed)
	model = PPO(
		"MlpPolicy", env, n_steps=args.num_steps, batch_size=BATCH_SIZE, n_epochs=UPDATE_EPOCHS,
		learning_rate=args.learning_rate, gamma=args.gamma, gae_lambda=args.gae_lambda,
		clip_range=args.clip_range, ent_coef=ENTROPY_COEF, vf_coef=VALUE_LOSS_COEF, max_grad_norm=MAX_GRAD_NORM,
		seed=args.seed, verbose=1, tensorboard_log=str(TENSORBOARD_LOG_DIR),
	)
	model.learn(total_timesteps=TOTAL_TIMESTEPS)
	model.save(MODEL_PATH)
	print(f"Training complete. Model saved to {MODEL_PATH}")
	env.close()


if __name__ == "__main__":
	train(parse_args())
