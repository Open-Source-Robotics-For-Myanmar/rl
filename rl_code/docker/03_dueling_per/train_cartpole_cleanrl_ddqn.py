import argparse
import random
import time
from pathlib import Path
from typing import NamedTuple

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from stable_baselines3.common.buffers import ReplayBuffer
from torch.utils.tensorboard import SummaryWriter

USE_DUELING = True
USE_PER = True
TOTAL_TIMESTEPS = 500_000  # training loop ကို total ဘယ်နှစ်ကြိမ် (environment step) run မလဲ
PER_ALPHA = 0.6  # priority ကို sampling probability ထဲ ဘယ်လောက်အသားပေးမလဲ (0 = uniform sampling, 1 = priority အတိုင်းအပြည့်)
PER_BETA_START = 0.4  # training အစမှာ importance-sampling correction ကို ဘယ်လောက်နည်းနည်းပဲ ပြင်မလဲ
PER_BETA_END = 1.0  # training ပြီးခါနီးမှာ bias ကို အပြည့်ပြင်ဖို့ 1.0 အထိ linear ဖြည်းဖြည်း တိုးသွားမယ်
PER_EPS = 1e-6  # TD-error 0 ဖြစ်နေရင်တောင် priority က 0 မဖြစ်အောင် ထည့်ပေးထားတဲ့ minimum value


class QNetwork(nn.Module):
	def __init__(self, env: gym.vector.SyncVectorEnv, use_dueling: bool = False) -> None:
		super().__init__()
		observation_size = int(np.prod(env.single_observation_space.shape))
		action_count = env.single_action_space.n
		self.use_dueling = use_dueling
		self.feature = nn.Sequential(
			nn.Linear(observation_size, 120),
			nn.ReLU(),
			nn.Linear(120, 84),
			nn.ReLU(),
		)
		if use_dueling:
			self.value_stream = nn.Linear(84, 1)
			self.advantage_stream = nn.Linear(84, action_count)
		else:
			self.output_layer = nn.Linear(84, action_count)

	def forward(self, observation: torch.Tensor) -> torch.Tensor:
		features = self.feature(observation)
		if self.use_dueling:
			value = self.value_stream(features)
			advantage = self.advantage_stream(features)
			return value + advantage - advantage.mean(dim=1, keepdim=True)
		return self.output_layer(features)


class PERSamples(NamedTuple):
	observations: torch.Tensor
	actions: torch.Tensor
	next_observations: torch.Tensor
	dones: torch.Tensor
	rewards: torch.Tensor


class PrioritizedReplayBuffer:
	"""Proportional-priority experience replay (Schaul et al., 2016)."""

	def __init__(
		self,
		buffer_size: int,
		observation_space: gym.Space,
		action_space: gym.Space,
		device: torch.device,
		alpha: float = 0.6,
	) -> None:
		self.buffer_size = buffer_size
		self.device = device
		self.alpha = alpha
		self.pos = 0
		self.full = False
		obs_shape = observation_space.shape
		self.observations = np.zeros((buffer_size, *obs_shape), dtype=np.float32)
		self.next_observations = np.zeros((buffer_size, *obs_shape), dtype=np.float32)
		self.actions = np.zeros((buffer_size, 1), dtype=np.int64)
		self.rewards = np.zeros((buffer_size, 1), dtype=np.float32)
		self.dones = np.zeros((buffer_size, 1), dtype=np.float32)
		self.priorities = np.zeros((buffer_size,), dtype=np.float32)

	def __len__(self) -> int:
		return self.buffer_size if self.full else self.pos

	def add(self, obs, next_obs, action, reward, done, infos) -> None:
		max_priority = self.priorities[: len(self)].max() if len(self) > 0 else 1.0
		self.observations[self.pos] = obs[0]
		self.next_observations[self.pos] = next_obs[0]
		self.actions[self.pos] = action[0]
		self.rewards[self.pos] = reward[0]
		self.dones[self.pos] = done[0]
		self.priorities[self.pos] = max_priority
		self.pos += 1
		if self.pos == self.buffer_size:
			self.full = True
			self.pos = 0

	def sample(self, batch_size: int, beta: float = 0.4):
		size = len(self)
		scaled_priorities = self.priorities[:size] ** self.alpha
		probabilities = scaled_priorities / scaled_priorities.sum()
		indices = np.random.choice(size, batch_size, p=probabilities)

		weights = (size * probabilities[indices]) ** (-beta)
		weights /= weights.max()
		weights_tensor = torch.as_tensor(weights, dtype=torch.float32, device=self.device)

		data = PERSamples(
			observations=torch.as_tensor(self.observations[indices], device=self.device),
			actions=torch.as_tensor(self.actions[indices], device=self.device),
			next_observations=torch.as_tensor(self.next_observations[indices], device=self.device),
			dones=torch.as_tensor(self.dones[indices], device=self.device),
			rewards=torch.as_tensor(self.rewards[indices], device=self.device),
		)
		return data, indices, weights_tensor

	def update_priorities(self, indices: np.ndarray, priorities: np.ndarray) -> None:
		self.priorities[indices] = priorities


def linear_schedule(start: float, end: float, duration: int, step: int) -> float:
	progress = min(step / duration, 1.0)
	return start + progress * (end - start)


def make_env() -> gym.Env:
	return gym.wrappers.RecordEpisodeStatistics(gym.make("CartPole-v1"))


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Train CleanRL-style DDQN on CartPole.")
	parser.add_argument("--learning-rate", type=float, default=2.5e-4)
	parser.add_argument("--buffer-size", type=int, default=10_000)
	parser.add_argument("--gamma", type=float, default=0.99)
	parser.add_argument("--batch-size", type=int, default=128)
	parser.add_argument("--learning-starts", type=int, default=10_000)
	parser.add_argument("--train-frequency", type=int, default=10)
	parser.add_argument("--target-network-frequency", type=int, default=500)
	parser.add_argument("--start-epsilon", type=float, default=1.0)
	parser.add_argument("--end-epsilon", type=float, default=0.05)
	parser.add_argument("--exploration-fraction", type=float, default=0.5)
	parser.add_argument("--seed", type=int, default=1)
	parser.add_argument("--cuda", action=argparse.BooleanOptionalAction, default=True)
	return parser.parse_args()


def train(args: argparse.Namespace) -> None:
	random.seed(args.seed)
	np.random.seed(args.seed)
	torch.manual_seed(args.seed)
	torch.backends.cudnn.deterministic = True

	device = torch.device("cuda" if args.cuda and torch.cuda.is_available() else "cpu")
	run_name = f"CartPole-v1__cleanrl_ddqn__{args.seed}__{int(time.time())}"
	writer = SummaryWriter(Path("cleanrl_ddqn_cartpole_tensorboard") / run_name)
	envs = gym.vector.SyncVectorEnv([make_env])
	envs.single_action_space.seed(args.seed)

	q_network = QNetwork(envs, use_dueling=USE_DUELING).to(device)
	target_network = QNetwork(envs, use_dueling=USE_DUELING).to(device)
	target_network.load_state_dict(q_network.state_dict())
	optimizer = optim.Adam(q_network.parameters(), lr=args.learning_rate)
	if USE_PER:
		replay_buffer = PrioritizedReplayBuffer(
			args.buffer_size,
			envs.single_observation_space,
			envs.single_action_space,
			device,
			alpha=PER_ALPHA,
		)
	else:
		replay_buffer = ReplayBuffer(
			args.buffer_size,
			envs.single_observation_space,
			envs.single_action_space,
			device,
			handle_timeout_termination=False,
		)

	observations, _ = envs.reset(seed=args.seed)
	start_time = time.time()
	exploration_steps = max(int(args.exploration_fraction * TOTAL_TIMESTEPS), 1)

	for global_step in range(TOTAL_TIMESTEPS):
		epsilon = linear_schedule(args.start_epsilon, args.end_epsilon, exploration_steps, global_step)
		if random.random() < epsilon:
			actions = np.array([envs.single_action_space.sample()])
		else:
			with torch.no_grad():
				actions = torch.argmax(
					q_network(torch.as_tensor(observations, device=device)), dim=1
				).cpu().numpy()

		next_observations, rewards, terminations, truncations, infos = envs.step(actions)
		real_next_observations = next_observations.copy()
		for index, truncated in enumerate(truncations):
			if truncated:
				real_next_observations[index] = infos["final_observation"][index]
		replay_buffer.add(
			observations, real_next_observations, actions, rewards, terminations, infos
		)
		observations = next_observations

		if "final_info" in infos:
			for info in infos["final_info"]:
				if info and "episode" in info:
					episode_return = float(np.asarray(info["episode"]["r"]).item())
					episode_length = int(np.asarray(info["episode"]["l"]).item())
					writer.add_scalar("charts/episodic_return", episode_return, global_step)
					writer.add_scalar("charts/episodic_length", episode_length, global_step)
					print(
						f"step={global_step}, return={episode_return:.1f}, "
						f"length={episode_length}"
					)

		if global_step > args.learning_starts and global_step % args.train_frequency == 0:
			if USE_PER:
				beta = linear_schedule(PER_BETA_START, PER_BETA_END, TOTAL_TIMESTEPS, global_step)
				data, indices, weights = replay_buffer.sample(args.batch_size, beta=beta)
			else:
				data = replay_buffer.sample(args.batch_size)

			with torch.no_grad():
				next_actions = q_network(data.next_observations).argmax(dim=1, keepdim=True)
				next_values = target_network(data.next_observations).gather(1, next_actions).squeeze(1)
				td_target = data.rewards.flatten() + args.gamma * next_values * (
					1 - data.dones.flatten()
				)
			current_values = q_network(data.observations).gather(
				1, data.actions.long()
			).squeeze(1)

			if USE_PER:
				elementwise_loss = nn.functional.smooth_l1_loss(current_values, td_target, reduction="none")
				loss = (weights * elementwise_loss).mean()
				td_errors = (td_target - current_values).detach().abs().cpu().numpy()
				replay_buffer.update_priorities(indices, td_errors + PER_EPS)
			else:
				loss = nn.functional.smooth_l1_loss(current_values, td_target)

			optimizer.zero_grad()
			loss.backward()
			optimizer.step()
			if global_step % 100 == 0:
				writer.add_scalar("losses/td_loss", loss.item(), global_step)
				writer.add_scalar("charts/epsilon", epsilon, global_step)
				writer.add_scalar("charts/SPS", global_step / (time.time() - start_time), global_step)

		if global_step % args.target_network_frequency == 0:
			target_network.load_state_dict(q_network.state_dict())

	model_path = Path("rom_cleanrl_ddqn_cartpole.cleanrl_model")
	torch.save(
		{
			"env_id": "CartPole-v1",
			"model_state_dict": q_network.state_dict(),
			"use_dueling": USE_DUELING,
		},
		model_path,
	)
	print(f"Training complete. Model saved to {model_path}")
	envs.close()
	writer.close()


if __name__ == "__main__":
	train(parse_args())
