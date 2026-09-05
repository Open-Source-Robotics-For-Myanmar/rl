import argparse
from pathlib import Path

from stable_baselines3 import A2C
from stable_baselines3.common.env_util import make_vec_env

TOTAL_TIMESTEPS = 200_000  # training loop ကို total ဘယ်နှစ် environment step run မလဲ


# command-line arguments (learning rate, gamma, seed) တွေကို parse လုပ်တယ်
def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Train SB3 A2C on CartPole (VPG comparison baseline).")
	parser.add_argument("--learning-rate", type=float, default=7e-4)
	parser.add_argument("--gamma", type=float, default=0.99)
	parser.add_argument("--seed", type=int, default=1)
	return parser.parse_args()


# Stable-Baselines3 ရဲ့ built-in A2C ကို VPG (CleanRL) ရလဒ်နှင့် နှိုင်းယှဉ်ဖို့ train ပေးတယ်
def train(args: argparse.Namespace) -> None:
	env = make_vec_env("CartPole-v1", n_envs=1, seed=args.seed)
	model = A2C(
		"MlpPolicy",
		env,
		learning_rate=args.learning_rate,
		gamma=args.gamma,
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
