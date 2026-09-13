import argparse
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize

ENV_ID = "HalfCheetah-v4"  # MuJoCo continuous-control env (observation & action နှစ်ခုစလုံး continuous)
TOTAL_TIMESTEPS = 1_000_000
NUM_ENVS = 4
NUM_STEPS = 2048  # update တစ်ကြိမ်မလုပ်ခင် env တစ်ခုစီရဲ့ rollout length
BATCH_SIZE = 64
LEARNING_RATE = 3e-4
GAMMA = 0.99
GAE_LAMBDA = 0.95
UPDATE_EPOCHS = 10
CLIP_RANGE = 0.2
ENTROPY_COEF = 0.0
VALUE_LOSS_COEF = 0.5
MAX_GRAD_NORM = 0.5
SEED = 1
MODEL_PATH = Path("rom_sb3_ppo_halfcheetahv4")
VECNORMALIZE_PATH = Path("rom_sb3_ppo_halfcheetahv4_vecnormalize.pkl")  # running obs/reward mean-std ကို ပြန်သုံးဖို့ သိမ်း
TENSORBOARD_LOG_DIR = Path("sb3_ppo_halfcheetahv4_tensorboard")


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Train SB3 PPO on HalfCheetah-v4.")
	parser.add_argument("--num-steps", type=int, default=NUM_STEPS)
	parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE)
	parser.add_argument("--gamma", type=float, default=GAMMA)
	parser.add_argument("--gae-lambda", type=float, default=GAE_LAMBDA)
	parser.add_argument("--clip-range", type=float, default=CLIP_RANGE)
	parser.add_argument("--n-envs", type=int, default=NUM_ENVS)
	parser.add_argument("--seed", type=int, default=SEED)
	return parser.parse_args()


def train(args: argparse.Namespace) -> None:
	# n_envs ခုကို parallel run လုပ်ပြီး rollout collect (SB3 default: DummyVecEnv)
	env = make_vec_env(ENV_ID, n_envs=args.n_envs, seed=args.seed)
	# HalfCheetah-v4 လို continuous-control envs အတွက် running mean/std normalization က training stability ကို ကူညီတယ်
	env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0)
	model = PPO(
		"MlpPolicy", env, n_steps=args.num_steps, batch_size=BATCH_SIZE, n_epochs=UPDATE_EPOCHS,
		learning_rate=args.learning_rate, gamma=args.gamma, gae_lambda=args.gae_lambda,
		clip_range=args.clip_range, ent_coef=ENTROPY_COEF, vf_coef=VALUE_LOSS_COEF, max_grad_norm=MAX_GRAD_NORM,
		seed=args.seed, verbose=1, tensorboard_log=str(TENSORBOARD_LOG_DIR),
	)
	model.learn(total_timesteps=TOTAL_TIMESTEPS)
	model.save(MODEL_PATH)
	# testing/evaluation အချိန် observation ကို ပြန် normalize လုပ်ဖို့ VecNormalize stats ကိုပါ သိမ်း
	env.save(str(VECNORMALIZE_PATH))
	print(f"Training complete. Model saved to {MODEL_PATH}")
	env.close()


if __name__ == "__main__":
	train(parse_args())
