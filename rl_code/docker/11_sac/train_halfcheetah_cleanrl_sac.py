import argparse
import random
import time
from pathlib import Path

import gymnasium as gym
import numpy as np
import pybullet_envs_gymnasium  # noqa: F401  (registers HalfCheetahBulletEnv-v0)
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from stable_baselines3.common.buffers import ReplayBuffer
from torch.utils.tensorboard import SummaryWriter

ENV_ID = "HalfCheetahBulletEnv-v0"
TOTAL_TIMESTEPS = 1_000_000
POLICY_LEARNING_RATE = 3e-4
Q_LEARNING_RATE = 1e-3
BUFFER_SIZE = 1_000_000
GAMMA = 0.99
TAU = 0.005
BATCH_SIZE = 256
LEARNING_STARTS = 5_000
POLICY_FREQUENCY = 2
TARGET_NETWORK_FREQUENCY = 1
ALPHA = 0.2
AUTOTUNE = True
HIDDEN_SIZE = 256
LOG_STD_MIN = -5.0
LOG_STD_MAX = 2.0
SEED = 1
LOG_FREQUENCY = 1_000
MODEL_PATH = Path("rom_cleanrl_sac_halfcheetah.cleanrl_model")
TENSORBOARD_LOG_DIR = Path("cleanrl_sac_halfcheetah_tensorboard")


class Actor(nn.Module):
	def __init__(self, envs: gym.vector.SyncVectorEnv) -> None:
		super().__init__()
		observation_size = int(np.prod(envs.single_observation_space.shape))
		action_size = int(np.prod(envs.single_action_space.shape))
		self.net = nn.Sequential(
			nn.Linear(observation_size, HIDDEN_SIZE), nn.ReLU(),
			nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE), nn.ReLU(),
		)
		self.mean_head = nn.Linear(HIDDEN_SIZE, action_size)
		self.log_std_head = nn.Linear(HIDDEN_SIZE, action_size)
		action_high = torch.as_tensor(envs.single_action_space.high, dtype=torch.float32)
		action_low = torch.as_tensor(envs.single_action_space.low, dtype=torch.float32)
		self.register_buffer("action_scale", (action_high - action_low) / 2.0)
		self.register_buffer("action_bias", (action_high + action_low) / 2.0)

	def forward(self, observation: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
		hidden = self.net(observation)
		mean = self.mean_head(hidden)
		log_std = torch.tanh(self.log_std_head(hidden))
		log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1.0)
		return mean, log_std

	def get_action(self, observation: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
		mean, log_std = self(observation)
		std = log_std.exp()
		normal = torch.distributions.Normal(mean, std)
		raw_action = normal.rsample()
		squashed_action = torch.tanh(raw_action)
		action = squashed_action * self.action_scale + self.action_bias
		log_prob = normal.log_prob(raw_action)
		log_prob -= torch.log(self.action_scale * (1.0 - squashed_action.pow(2)) + 1e-6)
		log_prob = log_prob.sum(1, keepdim=True)
		deterministic_mean = torch.tanh(mean) * self.action_scale + self.action_bias
		return action, log_prob, deterministic_mean


class QNetwork(nn.Module):
	def __init__(self, envs: gym.vector.SyncVectorEnv) -> None:
		super().__init__()
		observation_size = int(np.prod(envs.single_observation_space.shape))
		action_size = int(np.prod(envs.single_action_space.shape))
		self.net = nn.Sequential(
			nn.Linear(observation_size + action_size, HIDDEN_SIZE), nn.ReLU(),
			nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE), nn.ReLU(),
			nn.Linear(HIDDEN_SIZE, 1),
		)

	def forward(self, observation: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
		return self.net(torch.cat([observation, action], dim=1)).squeeze(-1)


def make_env() -> gym.Env:
	return gym.wrappers.RecordEpisodeStatistics(gym.make(ENV_ID))


def soft_update(target: nn.Module, source: nn.Module, tau: float) -> None:
	for target_param, source_param in zip(target.parameters(), source.parameters()):
		target_param.data.copy_(tau * source_param.data + (1.0 - tau) * target_param.data)


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Train CleanRL-style SAC on HalfCheetahBulletEnv-v0.")
	parser.add_argument("--policy-learning-rate", type=float, default=POLICY_LEARNING_RATE)
	parser.add_argument("--q-learning-rate", type=float, default=Q_LEARNING_RATE)
	parser.add_argument("--buffer-size", type=int, default=BUFFER_SIZE)
	parser.add_argument("--gamma", type=float, default=GAMMA)
	parser.add_argument("--tau", type=float, default=TAU)
	parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
	parser.add_argument("--learning-starts", type=int, default=LEARNING_STARTS)
	parser.add_argument("--policy-frequency", type=int, default=POLICY_FREQUENCY)
	parser.add_argument("--target-network-frequency", type=int, default=TARGET_NETWORK_FREQUENCY)
	parser.add_argument("--alpha", type=float, default=ALPHA)
	parser.add_argument("--autotune", action=argparse.BooleanOptionalAction, default=AUTOTUNE)
	parser.add_argument("--seed", type=int, default=SEED)
	parser.add_argument("--cuda", action=argparse.BooleanOptionalAction, default=True)
	return parser.parse_args()


def train(args: argparse.Namespace) -> None:
	random.seed(args.seed)
	np.random.seed(args.seed)
	torch.manual_seed(args.seed)
	device = torch.device("cuda" if args.cuda and torch.cuda.is_available() else "cpu")
	run_name = f"{ENV_ID}__cleanrl_sac__{args.seed}__{int(time.time())}"
	writer = SummaryWriter(TENSORBOARD_LOG_DIR / run_name)
	envs = gym.vector.SyncVectorEnv([make_env])
	envs.single_action_space.seed(args.seed)

	actor = Actor(envs).to(device)
	qf1, qf2 = QNetwork(envs).to(device), QNetwork(envs).to(device)
	qf1_target, qf2_target = QNetwork(envs).to(device), QNetwork(envs).to(device)
	qf1_target.load_state_dict(qf1.state_dict())
	qf2_target.load_state_dict(qf2.state_dict())
	actor_optimizer = optim.Adam(actor.parameters(), lr=args.policy_learning_rate)
	q_optimizer = optim.Adam(list(qf1.parameters()) + list(qf2.parameters()), lr=args.q_learning_rate)

	if args.autotune:
		target_entropy = -float(np.prod(envs.single_action_space.shape))
		log_alpha = torch.zeros(1, requires_grad=True, device=device)
		alpha = log_alpha.exp().item()
		alpha_optimizer = optim.Adam([log_alpha], lr=args.q_learning_rate)
	else:
		alpha = args.alpha

	replay_buffer = ReplayBuffer(args.buffer_size, envs.single_observation_space, envs.single_action_space, device, handle_timeout_termination=False)
	observations, _ = envs.reset(seed=args.seed)
	start_time = time.time()

	for global_step in range(TOTAL_TIMESTEPS):
		if global_step < args.learning_starts:
			actions = np.array([envs.single_action_space.sample()])
		else:
			with torch.no_grad():
				actions, _, _ = actor.get_action(torch.as_tensor(observations, dtype=torch.float32, device=device))
			actions = actions.cpu().numpy()
		next_observations, rewards, terminations, truncations, infos = envs.step(actions)
		real_next_observations = next_observations.copy()
		for index, truncated in enumerate(truncations):
			if truncated:
				real_next_observations[index] = infos["final_observation"][index]
		replay_buffer.add(observations, real_next_observations, actions, rewards, terminations, infos)
		observations = next_observations

		if "final_info" in infos:
			for info in infos["final_info"]:
				if info and "episode" in info:
					episode_return = float(np.asarray(info["episode"]["r"]).item())
					episode_length = int(np.asarray(info["episode"]["l"]).item())
					writer.add_scalar("charts/episodic_return", episode_return, global_step)
					writer.add_scalar("charts/episodic_length", episode_length, global_step)
					print(f"step={global_step:,} return={episode_return:.1f} length={episode_length}")

		if global_step > args.learning_starts:
			data = replay_buffer.sample(args.batch_size)
			batch_observations = data.observations.float()
			batch_actions = data.actions.float()
			batch_next_observations = data.next_observations.float()
			batch_rewards = data.rewards.float().flatten()
			batch_dones = data.dones.float().flatten()

			with torch.no_grad():
				next_actions, next_log_pi, _ = actor.get_action(batch_next_observations)
				next_q_values = torch.min(qf1_target(batch_next_observations, next_actions), qf2_target(batch_next_observations, next_actions))
				next_q_values = next_q_values - alpha * next_log_pi.view(-1)
				td_target = batch_rewards + args.gamma * next_q_values * (1 - batch_dones)
			qf_loss = F.mse_loss(qf1(batch_observations, batch_actions), td_target) + F.mse_loss(qf2(batch_observations, batch_actions), td_target)
			q_optimizer.zero_grad()
			qf_loss.backward()
			q_optimizer.step()

			if global_step % args.policy_frequency == 0:
				for _ in range(args.policy_frequency):
					policy_actions, log_pi, _ = actor.get_action(batch_observations)
					min_qf_pi = torch.min(qf1(batch_observations, policy_actions), qf2(batch_observations, policy_actions))
					actor_loss = (alpha * log_pi.view(-1) - min_qf_pi).mean()
					actor_optimizer.zero_grad()
					actor_loss.backward()
					actor_optimizer.step()

					if args.autotune:
						with torch.no_grad():
							_, log_pi, _ = actor.get_action(batch_observations)
						alpha_loss = (-log_alpha.exp() * (log_pi.view(-1) + target_entropy)).mean()
						alpha_optimizer.zero_grad()
						alpha_loss.backward()
						alpha_optimizer.step()
						alpha = log_alpha.exp().item()
				writer.add_scalar("losses/actor_loss", actor_loss.item(), global_step)
				writer.add_scalar("losses/alpha", alpha, global_step)

			if global_step % args.target_network_frequency == 0:
				soft_update(qf1_target, qf1, args.tau)
				soft_update(qf2_target, qf2, args.tau)

			if global_step % LOG_FREQUENCY == 0:
				writer.add_scalar("losses/qf_loss", qf_loss.item(), global_step)
				steps_per_second = global_step / (time.time() - start_time)
				writer.add_scalar("charts/SPS", steps_per_second, global_step)
				print(f"step={global_step:,} qf_loss={qf_loss.item():.4f} alpha={alpha:.4f} SPS={steps_per_second:.1f}")

	torch.save({"env_id": ENV_ID, "model_state_dict": actor.state_dict()}, MODEL_PATH)
	print(f"Training complete. Model saved to {MODEL_PATH}")
	envs.close()
	writer.close()


if __name__ == "__main__":
	train(parse_args())
