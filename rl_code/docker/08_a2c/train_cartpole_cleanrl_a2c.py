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
from torch.distributions import Categorical
from torch.utils.tensorboard import SummaryWriter

TOTAL_TIMESTEPS = 500_000  # training loop ကို total ဘယ်နှစ် environment step run မလဲ


# policy(actor) နဲ့ value(critic) head နှှစ်ခုကို shared layer တစ်ခုတည်းက ထုတ်ပေးတဲ့ actor-critic network
class ActorCriticNetwork(nn.Module):
	def __init__(self, envs: gym.vector.SyncVectorEnv) -> None:
		super().__init__()
		observation_size = int(np.prod(envs.single_observation_space.shape))
		action_count = envs.single_action_space.n
		self.shared = nn.Sequential(
			nn.Linear(observation_size, 128),
			nn.ReLU(),
		)
		self.policy_head = nn.Linear(128, action_count)
		self.value_head = nn.Linear(128, 1)

	# observation ကနေ (policy logits, state value) နှှစ်ခုကို တစ်ခုတည်း ထုတ်ပေးတဲ့ forward pass
	def forward(self, observation: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
		features = self.shared(observation)
		return self.policy_head(features), self.value_head(features).squeeze(-1)


# CartPole environment ကို episode statistics wrapper နဲ့ ဖန်တီးတယ်
def make_env() -> gym.Env:
	return gym.wrappers.RecordEpisodeStatistics(gym.make("CartPole-v1"))


# command-line arguments (num-envs, num-steps, learning rate, gamma, gae-lambda, entropy/value coef, seed) တွေကို parse လုပ်တယ်
def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description="Train CleanRL-style synchronous A2C (parallel envs + GAE) on CartPole."
	)
	parser.add_argument("--num-envs", type=int, default=8)
	parser.add_argument("--num-steps", type=int, default=20)
	parser.add_argument("--learning-rate", type=float, default=7e-4)
	parser.add_argument("--gamma", type=float, default=0.99)
	parser.add_argument("--gae-lambda", type=float, default=0.95)
	parser.add_argument("--entropy-coef", type=float, default=0.01)
	parser.add_argument("--value-loss-coef", type=float, default=0.5)
	parser.add_argument("--max-grad-norm", type=float, default=0.5)
	parser.add_argument("--seed", type=int, default=1)
	parser.add_argument("--cuda", action=argparse.BooleanOptionalAction, default=True)
	return parser.parse_args()


# parallel vectorized environments (num_envs ခု) ကို synchronous rollout + GAE နဲ့ train လုပ်တဲ့ A2C
def train(args: argparse.Namespace) -> None:
	random.seed(args.seed)
	np.random.seed(args.seed)
	torch.manual_seed(args.seed)

	device = torch.device("cuda" if args.cuda and torch.cuda.is_available() else "cpu")
	run_name = f"CartPole-v1__cleanrl_a2c__{args.seed}__{int(time.time())}"
	writer = SummaryWriter(Path("cleanrl_a2c_cartpole_tensorboard") / run_name)

	envs = gym.vector.SyncVectorEnv([make_env for _ in range(args.num_envs)])
	envs.single_action_space.seed(args.seed)
	network = ActorCriticNetwork(envs).to(device)
	optimizer = optim.Adam(network.parameters(), lr=args.learning_rate)

	observations, _ = envs.reset(seed=args.seed)
	global_step = 0
	start_time = time.time()
	num_updates = TOTAL_TIMESTEPS // (args.num_envs * args.num_steps)

	for update in range(num_updates):
		log_probs_list, values_list, entropies_list, rewards_list, dones_list = [], [], [], [], []

		# num_steps step အထိ parallel environment အတွေ့တွေ့ကို run ပြီး trajectory ကောက်ယူ
		for _ in range(args.num_steps):
			observation_tensor = torch.as_tensor(observations, dtype=torch.float32, device=device)
			logits, value = network(observation_tensor)
			distribution = Categorical(logits=logits)
			action = distribution.sample()

			next_observations, rewards, terminations, truncations, infos = envs.step(action.cpu().numpy())
			dones = np.logical_or(terminations, truncations)

			log_probs_list.append(distribution.log_prob(action))
			values_list.append(value)
			entropies_list.append(distribution.entropy())
			rewards_list.append(torch.as_tensor(rewards, dtype=torch.float32, device=device))
			dones_list.append(torch.as_tensor(dones, dtype=torch.float32, device=device))

			observations = next_observations
			global_step += args.num_envs

			if "final_info" in infos:
				for info in infos["final_info"]:
					if info and "episode" in info:
						episode_return = float(np.asarray(info["episode"]["r"]).item())
						episode_length = int(np.asarray(info["episode"]["l"]).item())
						writer.add_scalar("charts/episodic_return", episode_return, global_step)
						writer.add_scalar("charts/episodic_length", episode_length, global_step)
						print(
							f"step={global_step}, return={episode_return:.1f}, length={episode_length}"
						)

		with torch.no_grad():
			observation_tensor = torch.as_tensor(observations, dtype=torch.float32, device=device)
			_, bootstrap_value = network(observation_tensor)

		values_tensor = torch.stack(values_list)
		rewards_tensor = torch.stack(rewards_list)
		dones_tensor = torch.stack(dones_list)
		log_probs_tensor = torch.stack(log_probs_list)
		entropies_tensor = torch.stack(entropies_list)

		advantages = torch.zeros_like(rewards_tensor)
		last_gae = torch.zeros(args.num_envs, device=device)
		next_value = bootstrap_value
		# GAE (Generalized Advantage Estimation) ကို timestep နောက်ကနေ ရှေ့ဆီ (reverse) တွက်ချက်
		for t in reversed(range(args.num_steps)):
			next_non_terminal = 1.0 - dones_tensor[t]
			delta = rewards_tensor[t] + args.gamma * next_value * next_non_terminal - values_tensor[t]
			last_gae = delta + args.gamma * args.gae_lambda * next_non_terminal * last_gae
			advantages[t] = last_gae
			next_value = values_tensor[t]
		returns = advantages + values_tensor.detach()

		# policy loss (advantage-weighted log_prob) + value loss (MSE) + entropy bonus အတွက် total loss ပေါင်းတယ်
		policy_loss = -(log_probs_tensor * advantages.detach()).mean()
		value_loss = F.mse_loss(values_tensor, returns)
		entropy_loss = entropies_tensor.mean()
		loss = policy_loss + args.value_loss_coef * value_loss - args.entropy_coef * entropy_loss

		optimizer.zero_grad()
		loss.backward()
		nn.utils.clip_grad_norm_(network.parameters(), args.max_grad_norm)
		optimizer.step()

		if update % 10 == 0:
			writer.add_scalar("losses/policy_loss", policy_loss.item(), global_step)
			writer.add_scalar("losses/value_loss", value_loss.item(), global_step)
			writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
			writer.add_scalar("charts/SPS", global_step / (time.time() - start_time), global_step)

	model_path = Path("rom_cleanrl_a2c_cartpole.cleanrl_model")
	torch.save({"env_id": "CartPole-v1", "model_state_dict": network.state_dict()}, model_path)
	print(f"Training complete. Model saved to {model_path}")
	envs.close()
	writer.close()


if __name__ == "__main__":
	train(parse_args())
