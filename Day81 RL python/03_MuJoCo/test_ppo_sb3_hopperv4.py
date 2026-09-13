import argparse
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize

ENV_ID = "Hopper-v4"
MODEL_PATH = Path("rom_sb3_ppo_hopperv4")
VECNORMALIZE_PATH = Path("rom_sb3_ppo_hopperv4_vecnormalize.pkl")
NUM_EPISODES = 5
SEED = 1


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Evaluate SB3 PPO on Hopper-v4.")
	parser.add_argument("--model-path", type=Path, default=MODEL_PATH)
	parser.add_argument("--vecnormalize-path", type=Path, default=VECNORMALIZE_PATH)
	parser.add_argument("--episodes", type=int, default=NUM_EPISODES)
	parser.add_argument("--seed", type=int, default=SEED)
	return parser.parse_args()


def evaluate(args: argparse.Namespace) -> None:
	if args.episodes < 1:
		raise ValueError("--episodes must be at least 1")
	if not args.model_path.with_suffix(".zip").exists():
		raise FileNotFoundError(f"Model not found: {args.model_path}.zip")
	if not args.vecnormalize_path.exists():
		raise FileNotFoundError(f"VecNormalize statistics not found: {args.vecnormalize_path}")

	env = make_vec_env(ENV_ID, n_envs=1, seed=args.seed, env_kwargs={"render_mode": "human"})
	env = VecNormalize.load(str(args.vecnormalize_path), env)
	env.training = False
	env.norm_reward = False
	model = PPO.load(str(args.model_path), env=env)

	total_rewards = []
	try:
		for episode in range(1, args.episodes + 1):
			observation = env.reset()
			done = False
			episode_reward = 0.0
			step_count = 0

			while not done:
				action, _ = model.predict(observation, deterministic=True)
				observation, reward, done, _ = env.step(action)
				episode_reward += float(reward[0])
				step_count += 1

			total_rewards.append(episode_reward)
			print(f"Episode {episode}/{args.episodes} - Steps: {step_count}, Reward: {episode_reward:.2f}")
	finally:
		env.close()

	average_reward = sum(total_rewards) / len(total_rewards)
	print(f"Testing complete. Mean reward over {args.episodes} episodes: {average_reward:.2f}")


if __name__ == "__main__":
	evaluate(parse_args())
