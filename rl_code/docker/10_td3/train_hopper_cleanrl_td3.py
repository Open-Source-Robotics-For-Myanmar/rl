import argparse
import random
import time
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from stable_baselines3.common.buffers import ReplayBuffer
from torch.utils.tensorboard import SummaryWriter

ENV_ID = "Hopper-v4"
TOTAL_TIMESTEPS = 1_000_000
LEARNING_RATE = 3e-4
BUFFER_SIZE = 1_000_000
GAMMA = 0.99
TAU = 0.005
BATCH_SIZE = 256
EXPLORATION_NOISE = 0.1
POLICY_NOISE = 0.2
NOISE_CLIP = 0.5
LEARNING_STARTS = 25_000
POLICY_FREQUENCY = 2
HIDDEN_SIZE = 256
SEED = 1
LOG_FREQUENCY = 1_000
MODEL_PATH = Path("rom_cleanrl_td3_hopper.cleanrl_model")
TENSORBOARD_LOG_DIR = Path("cleanrl_td3_hopper_tensorboard")


class Actor(nn.Module):
	def __init__(self, envs: gym.vector.SyncVectorEnv) -> None:
		super().__init__()
		observation_size = int(np.prod(envs.single_observation_space.shape))
		action_size = int(np.prod(envs.single_action_space.shape))
		self.net = nn.Sequential(
			nn.Linear(observation_size, HIDDEN_SIZE), nn.ReLU(),
			nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE), nn.ReLU(),
			nn.Linear(HIDDEN_SIZE, action_size),
		)
		action_high = torch.as_tensor(envs.single_action_space.high, dtype=torch.float32)
		action_low = torch.as_tensor(envs.single_action_space.low, dtype=torch.float32)
		self.register_buffer("action_scale", (action_high - action_low) / 2.0)
		self.register_buffer("action_bias", (action_high + action_low) / 2.0)

	def forward(self, observation: torch.Tensor) -> torch.Tensor:
		return torch.tanh(self.net(observation)) * self.action_scale + self.action_bias


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
	parser = argparse.ArgumentParser(description="Train CleanRL-style TD3 on Hopper.")
	parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE)
	parser.add_argument("--buffer-size", type=int, default=BUFFER_SIZE)
	parser.add_argument("--gamma", type=float, default=GAMMA)
	parser.add_argument("--tau", type=float, default=TAU)
	parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
	parser.add_argument("--exploration-noise", type=float, default=EXPLORATION_NOISE)
	parser.add_argument("--policy-noise", type=float, default=POLICY_NOISE)
	parser.add_argument("--noise-clip", type=float, default=NOISE_CLIP)
	parser.add_argument("--learning-starts", type=int, default=LEARNING_STARTS)
	parser.add_argument("--policy-frequency", type=int, default=POLICY_FREQUENCY)
	parser.add_argument("--seed", type=int, default=SEED)
	parser.add_argument("--cuda", action=argparse.BooleanOptionalAction, default=True)
	return parser.parse_args()


def train(args: argparse.Namespace) -> None:
	random.seed(args.seed)
	np.random.seed(args.seed)
	torch.manual_seed(args.seed)
	device = torch.device("cuda" if args.cuda and torch.cuda.is_available() else "cpu")
	run_name = f"{ENV_ID}__cleanrl_td3__{args.seed}__{int(time.time())}"
	writer = SummaryWriter(TENSORBOARD_LOG_DIR / run_name)
	envs = gym.vector.SyncVectorEnv([make_env])
	envs.single_action_space.seed(args.seed)

	actor, target_actor = Actor(envs).to(device), Actor(envs).to(device)
	target_actor.load_state_dict(actor.state_dict())
	qf1, qf2 = QNetwork(envs).to(device), QNetwork(envs).to(device)
	target_qf1, target_qf2 = QNetwork(envs).to(device), QNetwork(envs).to(device)
	target_qf1.load_state_dict(qf1.state_dict())
	target_qf2.load_state_dict(qf2.state_dict())
	actor_optimizer = optim.Adam(actor.parameters(), lr=args.learning_rate)
	qf_optimizer = optim.Adam(list(qf1.parameters()) + list(qf2.parameters()), lr=args.learning_rate)
	replay_buffer = ReplayBuffer(args.buffer_size, envs.single_observation_space, envs.single_action_space, device, handle_timeout_termination=False)
	action_low, action_high = envs.single_action_space.low, envs.single_action_space.high
	action_scale = actor.action_scale.cpu().numpy()
	observations, _ = envs.reset(seed=args.seed)
	start_time = time.time()

	for global_step in range(TOTAL_TIMESTEPS):
		if global_step < args.learning_starts:
			actions = np.array([envs.single_action_space.sample()])
		else:
			with torch.no_grad():
				actions = actor(torch.as_tensor(observations, dtype=torch.float32, device=device)).cpu().numpy()
			actions = np.clip(actions + np.random.normal(0.0, args.exploration_noise * action_scale, actions.shape), action_low, action_high)
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
				target_noise = (torch.randn_like(batch_actions) * args.policy_noise * actor.action_scale).clamp(-args.noise_clip * actor.action_scale, args.noise_clip * actor.action_scale)
				next_actions = (target_actor(batch_next_observations) + target_noise).clamp(torch.as_tensor(action_low, dtype=torch.float32, device=device), torch.as_tensor(action_high, dtype=torch.float32, device=device))
				next_q_values = torch.min(target_qf1(batch_next_observations, next_actions), target_qf2(batch_next_observations, next_actions))
				td_target = batch_rewards + args.gamma * next_q_values * (1 - batch_dones)
			qf_loss = F.mse_loss(qf1(batch_observations, batch_actions), td_target) + F.mse_loss(qf2(batch_observations, batch_actions), td_target)
			qf_optimizer.zero_grad()
			qf_loss.backward()
			qf_optimizer.step()

			if global_step % args.policy_frequency == 0:
				actor_loss = -qf1(batch_observations, actor(batch_observations)).mean()
				actor_optimizer.zero_grad()
				actor_loss.backward()
				actor_optimizer.step()
				for target, source in ((target_actor, actor), (target_qf1, qf1), (target_qf2, qf2)):
					soft_update(target, source, args.tau)
				writer.add_scalar("losses/actor_loss", actor_loss.item(), global_step)
			if global_step % LOG_FREQUENCY == 0:
				writer.add_scalar("losses/qf_loss", qf_loss.item(), global_step)
				steps_per_second = global_step / (time.time() - start_time)
				writer.add_scalar("charts/SPS", steps_per_second, global_step)
				print(f"step={global_step:,} qf_loss={qf_loss.item():.4f} SPS={steps_per_second:.1f}")

	torch.save({"env_id": ENV_ID, "model_state_dict": actor.state_dict()}, MODEL_PATH)
	print(f"Training complete. Model saved to {MODEL_PATH}")
	envs.close()
	writer.close()


if __name__ == "__main__":
	train(parse_args())