import argparse
import random
import time
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from torch.utils.tensorboard import SummaryWriter

ENV_ID = "LunarLander-v2"
TOTAL_TIMESTEPS = 1_000_000
NUM_ENVS = 8
NUM_STEPS = 128
LEARNING_RATE = 2.5e-4
GAMMA = 0.99
GAE_LAMBDA = 0.95
UPDATE_EPOCHS = 4
NUM_MINIBATCHES = 4
CLIP_COEF = 0.2
ENTROPY_COEF = 0.01
VALUE_LOSS_COEF = 0.5
MAX_GRAD_NORM = 0.5
HIDDEN_SIZE = 64
SEED = 1
MODEL_PATH = Path("rom_cleanrl_ppo_lunarlander.cleanrl_model")
TENSORBOARD_LOG_DIR = Path("cleanrl_ppo_lunarlander_tensorboard")


def layer_init(layer: nn.Linear, std: float = np.sqrt(2), bias_const: float = 0.0) -> nn.Linear:
	nn.init.orthogonal_(layer.weight, std)
	nn.init.constant_(layer.bias, bias_const)
	return layer


# actor(policy) နှင့် critic(value) ကို သီးခြား network နှစ်ခုအဖြစ်ထားတဲ့ PPO agent
class Agent(nn.Module):
	def __init__(self, envs: gym.vector.SyncVectorEnv) -> None:
		super().__init__()
		observation_size = int(np.prod(envs.single_observation_space.shape))
		action_count = int(envs.single_action_space.n)
		self.critic = nn.Sequential(
			layer_init(nn.Linear(observation_size, HIDDEN_SIZE)), nn.Tanh(),
			layer_init(nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE)), nn.Tanh(),
			layer_init(nn.Linear(HIDDEN_SIZE, 1), std=1.0),
		)
		self.actor = nn.Sequential(
			layer_init(nn.Linear(observation_size, HIDDEN_SIZE)), nn.Tanh(),
			layer_init(nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE)), nn.Tanh(),
			layer_init(nn.Linear(HIDDEN_SIZE, action_count), std=0.01),
		)

	def get_value(self, observation: torch.Tensor) -> torch.Tensor:
		return self.critic(observation).squeeze(-1)

	# action မပေးလျှင် policy ကနေ sample ယူပြီး log_prob, entropy, value တွေအတူထုတ်ပေးတယ်
	def get_action_and_value(
		self, observation: torch.Tensor, action: torch.Tensor | None = None
	) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
		logits = self.actor(observation)
		distribution = Categorical(logits=logits)
		if action is None:
			action = distribution.sample()
		return action, distribution.log_prob(action), distribution.entropy(), self.critic(observation).squeeze(-1)


def make_env() -> gym.Env:
	return gym.wrappers.RecordEpisodeStatistics(gym.make(ENV_ID))


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Train CleanRL-style PPO on LunarLander.")
	parser.add_argument("--num-envs", type=int, default=NUM_ENVS)
	parser.add_argument("--num-steps", type=int, default=NUM_STEPS)
	parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE)
	parser.add_argument("--gamma", type=float, default=GAMMA)
	parser.add_argument("--gae-lambda", type=float, default=GAE_LAMBDA)
	parser.add_argument("--update-epochs", type=int, default=UPDATE_EPOCHS)
	parser.add_argument("--num-minibatches", type=int, default=NUM_MINIBATCHES)
	parser.add_argument("--clip-coef", type=float, default=CLIP_COEF)
	parser.add_argument("--entropy-coef", type=float, default=ENTROPY_COEF)
	parser.add_argument("--value-loss-coef", type=float, default=VALUE_LOSS_COEF)
	parser.add_argument("--max-grad-norm", type=float, default=MAX_GRAD_NORM)
	parser.add_argument("--seed", type=int, default=SEED)
	parser.add_argument("--cuda", action=argparse.BooleanOptionalAction, default=True)
	return parser.parse_args()


def train(args: argparse.Namespace) -> None:
	random.seed(args.seed)
	np.random.seed(args.seed)
	torch.manual_seed(args.seed)
	device = torch.device("cuda" if args.cuda and torch.cuda.is_available() else "cpu")
	run_name = f"{ENV_ID}__cleanrl_ppo__{args.seed}__{int(time.time())}"
	writer = SummaryWriter(TENSORBOARD_LOG_DIR / run_name)

	envs = gym.vector.SyncVectorEnv([make_env for _ in range(args.num_envs)])
	envs.single_action_space.seed(args.seed)
	agent = Agent(envs).to(device)
	optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

	batch_size = args.num_envs * args.num_steps
	minibatch_size = batch_size // args.num_minibatches
	num_updates = TOTAL_TIMESTEPS // batch_size

	observations = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape, device=device)
	actions = torch.zeros((args.num_steps, args.num_envs), dtype=torch.long, device=device)
	log_probs = torch.zeros((args.num_steps, args.num_envs), device=device)
	rewards = torch.zeros((args.num_steps, args.num_envs), device=device)
	dones = torch.zeros((args.num_steps, args.num_envs), device=device)
	values = torch.zeros((args.num_steps, args.num_envs), device=device)

	next_observation, _ = envs.reset(seed=args.seed)
	next_observation = torch.as_tensor(next_observation, dtype=torch.float32, device=device)
	next_done = torch.zeros(args.num_envs, device=device)
	global_step = 0
	start_time = time.time()

	for update in range(num_updates):
		# num_steps step အထိ parallel environment အားလုံးကို run ပြီး trajectory ကောက်ယူ
		for step in range(args.num_steps):
			global_step += args.num_envs
			observations[step] = next_observation
			dones[step] = next_done

			with torch.no_grad():
				action, log_prob, _, value = agent.get_action_and_value(next_observation)
				values[step] = value
			actions[step] = action
			log_probs[step] = log_prob

			next_observation_np, reward, terminations, truncations, infos = envs.step(action.cpu().numpy())
			done = np.logical_or(terminations, truncations)
			rewards[step] = torch.as_tensor(reward, dtype=torch.float32, device=device)
			next_observation = torch.as_tensor(next_observation_np, dtype=torch.float32, device=device)
			next_done = torch.as_tensor(done, dtype=torch.float32, device=device)

			if "final_info" in infos:
				for info in infos["final_info"]:
					if info and "episode" in info:
						episode_return = float(np.asarray(info["episode"]["r"]).item())
						episode_length = int(np.asarray(info["episode"]["l"]).item())
						writer.add_scalar("charts/episodic_return", episode_return, global_step)
						writer.add_scalar("charts/episodic_length", episode_length, global_step)
						print(f"step={global_step:,} return={episode_return:.1f} length={episode_length}")

		# rollout ပြီးလျှင် GAE (Generalized Advantage Estimation) ကို reverse recursion ဖြင့်တွက်
		with torch.no_grad():
			next_value = agent.get_value(next_observation)
			advantages = torch.zeros_like(rewards, device=device)
			last_gae_lambda = torch.zeros(args.num_envs, device=device)
			for t in reversed(range(args.num_steps)):
				if t == args.num_steps - 1:
					next_non_terminal = 1.0 - next_done
					next_values = next_value
				else:
					next_non_terminal = 1.0 - dones[t + 1]
					next_values = values[t + 1]
				delta = rewards[t] + args.gamma * next_values * next_non_terminal - values[t]
				last_gae_lambda = delta + args.gamma * args.gae_lambda * next_non_terminal * last_gae_lambda
				advantages[t] = last_gae_lambda
			returns = advantages + values

		batch_observations = observations.reshape((-1,) + envs.single_observation_space.shape)
		batch_log_probs = log_probs.reshape(-1)
		batch_actions = actions.reshape(-1)
		batch_advantages = advantages.reshape(-1)
		batch_returns = returns.reshape(-1)

		# clipped surrogate objective ဖြင့် update_epochs ကြိမ်၊ minibatch အလိုက် policy/value update
		batch_indices = np.arange(batch_size)
		for _ in range(args.update_epochs):
			np.random.shuffle(batch_indices)
			for start in range(0, batch_size, minibatch_size):
				minibatch_indices = batch_indices[start : start + minibatch_size]

				_, new_log_prob, entropy, new_value = agent.get_action_and_value(
					batch_observations[minibatch_indices], batch_actions[minibatch_indices]
				)
				ratio = (new_log_prob - batch_log_probs[minibatch_indices]).exp()

				minibatch_advantages = batch_advantages[minibatch_indices]
				minibatch_advantages = (minibatch_advantages - minibatch_advantages.mean()) / (minibatch_advantages.std() + 1e-8)

				policy_loss_unclipped = -minibatch_advantages * ratio
				policy_loss_clipped = -minibatch_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
				policy_loss = torch.max(policy_loss_unclipped, policy_loss_clipped).mean()
				value_loss = 0.5 * ((new_value - batch_returns[minibatch_indices]) ** 2).mean()
				entropy_loss = entropy.mean()
				loss = policy_loss - args.entropy_coef * entropy_loss + args.value_loss_coef * value_loss

				optimizer.zero_grad()
				loss.backward()
				nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
				optimizer.step()

		writer.add_scalar("losses/policy_loss", policy_loss.item(), global_step)
		writer.add_scalar("losses/value_loss", value_loss.item(), global_step)
		writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
		steps_per_second = global_step / (time.time() - start_time)
		writer.add_scalar("charts/SPS", steps_per_second, global_step)
		if update % 10 == 0:
			print(f"update={update} step={global_step:,} policy_loss={policy_loss.item():.4f} value_loss={value_loss.item():.4f} SPS={steps_per_second:.1f}")

	torch.save({"env_id": ENV_ID, "model_state_dict": agent.state_dict()}, MODEL_PATH)
	print(f"Training complete. Model saved to {MODEL_PATH}")
	envs.close()
	writer.close()


if __name__ == "__main__":
	train(parse_args())
