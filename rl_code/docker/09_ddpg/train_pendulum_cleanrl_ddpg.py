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

TOTAL_TIMESTEPS = 100_000  # training loop ကို total ဘယ်နှစ် environment step run မလဲ
LEARNING_RATE = 3e-4
BUFFER_SIZE = 100_000
GAMMA = 0.99
TAU = 0.005  # target network ကို soft-update လုပ်တဲ့ Polyak averaging coefficient
BATCH_SIZE = 256
EXPLORATION_NOISE = 0.1
LEARNING_STARTS = 5_000
POLICY_FREQUENCY = 2  # actor + target networks update လုပ်မည့် interval
SEED = 1


# continuous action ကို deterministic ထုတ်ပေးတဲ့ actor (policy) network
class Actor(nn.Module):
	def __init__(self, envs: gym.vector.SyncVectorEnv) -> None:
		super().__init__()
		observation_size = int(np.prod(envs.single_observation_space.shape))
		action_size = int(np.prod(envs.single_action_space.shape))
		self.net = nn.Sequential(
			nn.Linear(observation_size, 256),
			nn.ReLU(),
			nn.Linear(256, 256),
			nn.ReLU(),
			nn.Linear(256, action_size),
		)
		action_high = torch.as_tensor(envs.single_action_space.high, dtype=torch.float32)
		action_low = torch.as_tensor(envs.single_action_space.low, dtype=torch.float32)
		# tanh output (-1, 1) ကို environment ရဲ့ actual action range ထဲ scale ပြောင်းဖို့
		self.register_buffer("action_scale", (action_high - action_low) / 2.0)
		self.register_buffer("action_bias", (action_high + action_low) / 2.0)

	def forward(self, observation: torch.Tensor) -> torch.Tensor:
		action = torch.tanh(self.net(observation))
		return action * self.action_scale + self.action_bias


# (state, action) ချရင်ပြန်ထုတ်ပေးတဲ့ Q-value critic network
class QNetwork(nn.Module):
	def __init__(self, envs: gym.vector.SyncVectorEnv) -> None:
		super().__init__()
		observation_size = int(np.prod(envs.single_observation_space.shape))
		action_size = int(np.prod(envs.single_action_space.shape))
		self.net = nn.Sequential(
			nn.Linear(observation_size + action_size, 256),
			nn.ReLU(),
			nn.Linear(256, 256),
			nn.ReLU(),
			nn.Linear(256, 1),
		)

	def forward(self, observation: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
		return self.net(torch.cat([observation, action], dim=1)).squeeze(-1)


# Pendulum environment (continuous action) ကို episode statistics wrapper နဲ့ ဖန်တီးတယ်
def make_env() -> gym.Env:
	return gym.wrappers.RecordEpisodeStatistics(gym.make("Pendulum-v1"))


# online network parameters ကို target network ဆီ Polyak averaging (soft update) နဲ့ copy ချတယ်
def soft_update(target: nn.Module, source: nn.Module, tau: float) -> None:
	for target_param, source_param in zip(target.parameters(), source.parameters()):
		target_param.data.copy_(tau * source_param.data + (1.0 - tau) * target_param.data)


# command-line arguments (learning rate, buffer size, gamma, tau, batch size, exploration noise, seed) တွေကို parse လုပ်တယ်
def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Train CleanRL-style DDPG on Pendulum.")
	parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE)
	parser.add_argument("--buffer-size", type=int, default=BUFFER_SIZE)
	parser.add_argument("--gamma", type=float, default=GAMMA)
	parser.add_argument("--tau", type=float, default=TAU)
	parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
	parser.add_argument("--exploration-noise", type=float, default=EXPLORATION_NOISE)
	parser.add_argument("--learning-starts", type=int, default=LEARNING_STARTS)
	parser.add_argument("--policy-frequency", type=int, default=POLICY_FREQUENCY)
	parser.add_argument("--seed", type=int, default=SEED)
	parser.add_argument("--cuda", action=argparse.BooleanOptionalAction, default=True)
	return parser.parse_args()


# replay buffer + deterministic actor-critic (target networks + Polyak soft-update) ဖြင့် DDPG train လုပ်တယ်
def train(args: argparse.Namespace) -> None:
	random.seed(args.seed)
	np.random.seed(args.seed)
	torch.manual_seed(args.seed)

	device = torch.device("cuda" if args.cuda and torch.cuda.is_available() else "cpu")
	run_name = f"Pendulum-v1__cleanrl_ddpg__{args.seed}__{int(time.time())}"
	writer = SummaryWriter(Path("cleanrl_ddpg_pendulum_tensorboard") / run_name)

	envs = gym.vector.SyncVectorEnv([make_env])
	envs.single_action_space.seed(args.seed)

	actor = Actor(envs).to(device)
	target_actor = Actor(envs).to(device)
	target_actor.load_state_dict(actor.state_dict())
	qf = QNetwork(envs).to(device)
	target_qf = QNetwork(envs).to(device)
	target_qf.load_state_dict(qf.state_dict())

	actor_optimizer = optim.Adam(actor.parameters(), lr=args.learning_rate)
	qf_optimizer = optim.Adam(qf.parameters(), lr=args.learning_rate)

	replay_buffer = ReplayBuffer(
		args.buffer_size,
		envs.single_observation_space,
		envs.single_action_space,
		device,
		handle_timeout_termination=False,
	)

	action_low = envs.single_action_space.low
	action_high = envs.single_action_space.high
	action_scale_np = actor.action_scale.cpu().numpy()

	observations, _ = envs.reset(seed=args.seed)
	start_time = time.time()

	for global_step in range(TOTAL_TIMESTEPS):
		# learning-starts မပြည့်ခင် random action များနဲ့ replay buffer ကို warm-up လုပ်တယ်
		if global_step < args.learning_starts:
			actions = np.array([envs.single_action_space.sample()])
		else:
			with torch.no_grad():
				actions = actor(torch.as_tensor(observations, dtype=torch.float32, device=device)).cpu().numpy()
			noise = np.random.normal(0.0, args.exploration_noise * action_scale_np, size=actions.shape)
			actions = np.clip(actions + noise, action_low, action_high)

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
					print(f"step={global_step}, return={episode_return:.1f}, length={episode_length}")

		if global_step > args.learning_starts:
			data = replay_buffer.sample(args.batch_size)

			with torch.no_grad():
				next_state_actions = target_actor(data.next_observations)
				next_qf_values = target_qf(data.next_observations, next_state_actions)
				td_target = data.rewards.flatten() + args.gamma * next_qf_values * (1 - data.dones.flatten())
			qf_values = qf(data.observations, data.actions)
			qf_loss = F.mse_loss(qf_values, td_target)

			qf_optimizer.zero_grad()
			qf_loss.backward()
			qf_optimizer.step()

			# policy_frequency ခြားပြီးမှ actor + target networks ကို update လုပ်တယ် (delayed policy update)
			if global_step % args.policy_frequency == 0:
				actor_loss = -qf(data.observations, actor(data.observations)).mean()

				actor_optimizer.zero_grad()
				actor_loss.backward()
				actor_optimizer.step()

				soft_update(target_actor, actor, args.tau)
				soft_update(target_qf, qf, args.tau)

				if global_step % 100 == 0:
					writer.add_scalar("losses/actor_loss", actor_loss.item(), global_step)

			if global_step % 100 == 0:
				writer.add_scalar("losses/qf_loss", qf_loss.item(), global_step)
				writer.add_scalar("charts/SPS", global_step / (time.time() - start_time), global_step)

	model_path = Path("rom_cleanrl_ddpg_pendulum.cleanrl_model")
	torch.save({"env_id": "Pendulum-v1", "model_state_dict": actor.state_dict()}, model_path)
	print(f"Training complete. Model saved to {model_path}")
	envs.close()
	writer.close()


if __name__ == "__main__":
	train(parse_args())
