import argparse
from pathlib import Path

from stable_baselines3 import A2C
from stable_baselines3.common.env_util import make_vec_env

TOTAL_TIMESTEPS = 200_000  # training loop ကို total ဘယ်နှစ် environment step run မလဲ
NUM_STEPS = 20
LEARNING_RATE = 7e-4
GAMMA = 0.99
GAE_LAMBDA = 0.95


# command-line arguments (num-steps, learning rate, gamma, gae-lambda, n-envs, seed) တွေကို parse လုပ်တယ်
def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Train SB3 A2C on CartPole.")
	parser.add_argument("--num-steps", type=int, default=NUM_STEPS)
	parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE)
	parser.add_argument("--gamma", type=float, default=GAMMA)
	parser.add_argument("--gae-lambda", type=float, default=GAE_LAMBDA)
	parser.add_argument("--n-envs", type=int, default=8)
	parser.add_argument("--seed", type=int, default=1)
	return parser.parse_args()


# Stable-Baselines3 ရဲ့ built-in synchronous A2C (parallel envs + GAE) ကို train ပေးတယ်
def train(args: argparse.Namespace) -> None:
	env = make_vec_env("CartPole-v1", n_envs=args.n_envs, seed=args.seed)
	model = A2C(
		"MlpPolicy",
		env,
		n_steps=args.num_steps,
		learning_rate=args.learning_rate,
		gamma=args.gamma,
		gae_lambda=args.gae_lambda,
		seed=args.seed,
		verbose=1,
		tensorboard_log=str(Path("sb3_a2c_cartpole_tensorboard")),
	)
	model.learn(total_timesteps=TOTAL_TIMESTEPS)

	model_path = Path("rom_sb3_a2c_cartpole")
	model.save(model_path)
	print(f"Training complete. Model saved to {model_path}")
	env.close()


if __name__ == "__main__":
	train(parse_args())
