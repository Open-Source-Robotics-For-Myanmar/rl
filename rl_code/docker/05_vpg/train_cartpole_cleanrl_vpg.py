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

ENTROPY_COEF = 0.01  # exploration ကို အားပေးဖို့ entropy bonus ရဲ့ weight
TOTAL_EPISODES = 1_500  # training loop ကို total ဘယ်နှစ် episode run မလဲ


# state ကနေ action logits ထုတ်ပေးတဲ့ actor (policy) network
class PolicyNetwork(nn.Module):
	def __init__(self, env: gym.Env) -> None:
		super().__init__()
		observation_size = int(np.prod(env.observation_space.shape))
		action_count = env.action_space.n
		self.network = nn.Sequential(
			nn.Linear(observation_size, 120),
			nn.ReLU(),
			nn.Linear(120, 84),
			nn.ReLU(),
			nn.Linear(84, action_count),
		)

	# observation ကနေ action logits ထုတ်ပေးတဲ့ forward pass
	def forward(self, observation: torch.Tensor) -> torch.Tensor:
		return self.network(observation)

	# action sample ယူပြီး log_prob (loss အတွက်) နဲ့ entropy (exploration bonus အတွက်) ပြန်ပေးတယ်
	def get_action(self, observation: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
		logits = self(observation)
		distribution = Categorical(logits=logits)
		action = distribution.sample()
		return action, distribution.log_prob(action), distribution.entropy()


# state value V(s) ကို ခန့်မှန်းပေးတဲ့ baseline (critic) network — VPG ရဲ့ variance လျှော့ချဖို့ သုံးတယ်
class ValueNetwork(nn.Module):
	def __init__(self, env: gym.Env) -> None:
		super().__init__()
		observation_size = int(np.prod(env.observation_space.shape))
		self.network = nn.Sequential(
			nn.Linear(observation_size, 120),
			nn.ReLU(),
			nn.Linear(120, 84),
			nn.ReLU(),
			nn.Linear(84, 1),
		)

	# observation ကနေ scalar value estimate တစ်ခုတည်း ထုတ်ပေးတဲ့ forward pass
	def forward(self, observation: torch.Tensor) -> torch.Tensor:
		return self.network(observation).squeeze(-1)


# episode ရဲ့ reward list ကနေ gamma-discounted cumulative return ကို နောက်ကနေရှေ့ (reverse) တွက်ချက်တယ်
def discounted_returns(rewards: list[float], gamma: float) -> np.ndarray:
	returns = np.zeros(len(rewards), dtype=np.float32)
	running_return = 0.0
	for step in reversed(range(len(rewards))):
		running_return = rewards[step] + gamma * running_return
		returns[step] = running_return
	return returns


# CartPole environment ကို episode statistics wrapper နဲ့ ဖန်တီးတယ်
def make_env() -> gym.Env:
	return gym.wrappers.RecordEpisodeStatistics(gym.make("CartPole-v1"))


# command-line arguments (learning rate နှစ်ခု၊ gamma၊ seed စသည်) တွေကို parse လုပ်တယ်
def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Train CleanRL-style VPG (REINFORCE + baseline) on CartPole.")
	parser.add_argument("--policy-learning-rate", type=float, default=1e-3)
	parser.add_argument("--value-learning-rate", type=float, default=1e-3)
	parser.add_argument("--gamma", type=float, default=0.99)
	parser.add_argument("--normalize-advantages", action=argparse.BooleanOptionalAction, default=True)
	parser.add_argument("--seed", type=int, default=1)
	parser.add_argument("--cuda", action=argparse.BooleanOptionalAction, default=True)
	return parser.parse_args()


# VPG (REINFORCE + learned value baseline) ဖြင့် policy network နဲ့ value network နှစ်ခုလုံးကို train တယ်
def train(args: argparse.Namespace) -> None:
	random.seed(args.seed)
	np.random.seed(args.seed)
	torch.manual_seed(args.seed)

	device = torch.device("cuda" if args.cuda and torch.cuda.is_available() else "cpu")
	run_name = f"CartPole-v1__cleanrl_vpg__{args.seed}__{int(time.time())}"
	writer = SummaryWriter(Path("cleanrl_vpg_cartpole_tensorboard") / run_name)

	env = make_env()
	env.action_space.seed(args.seed)
	policy_network = PolicyNetwork(env).to(device)
	value_network = ValueNetwork(env).to(device)
	policy_optimizer = optim.Adam(policy_network.parameters(), lr=args.policy_learning_rate)
	value_optimizer = optim.Adam(value_network.parameters(), lr=args.value_learning_rate)

	global_step = 0
	start_time = time.time()
	for episode in range(TOTAL_EPISODES):
		observation, _ = env.reset(seed=args.seed + episode)
		observations: list[np.ndarray] = []
		log_probs: list[torch.Tensor] = []
		entropies: list[torch.Tensor] = []
		rewards: list[float] = []
		done = False
		info: dict = {}

		while not done:
			observation_tensor = torch.as_tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)
			action, log_prob, entropy = policy_network.get_action(observation_tensor)
			observations.append(observation)
			log_probs.append(log_prob)
			entropies.append(entropy)
			observation, reward, terminated, truncated, info = env.step(int(action.item()))
			rewards.append(float(reward))
			done = terminated or truncated
			global_step += 1

		# discounted return တွက်ပြီး value network ရဲ့ prediction ကို baseline အဖြစ်နှုတ်ပြီး advantage ရှာတယ်
		returns = discounted_returns(rewards, args.gamma)
		returns_tensor = torch.as_tensor(returns, device=device)
		observations_tensor = torch.as_tensor(np.array(observations), dtype=torch.float32, device=device)

		values = value_network(observations_tensor)
		advantages = returns_tensor - values.detach()
		if args.normalize_advantages:
			advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

		# policy loss မှာ entropy bonus ပါဝင်ပြီး၊ value loss ကတော့ value network ကို return နဲ့ regress လုပ်တယ်
		policy_loss = -(torch.cat(log_probs) * advantages).sum() - ENTROPY_COEF * torch.cat(entropies).sum()
		value_loss = F.mse_loss(values, returns_tensor)

		policy_optimizer.zero_grad()
		policy_loss.backward()
		policy_optimizer.step()

		value_optimizer.zero_grad()
		value_loss.backward()
		value_optimizer.step()

		if "episode" in info:
			episode_return = float(np.asarray(info["episode"]["r"]).item())
			episode_length = int(np.asarray(info["episode"]["l"]).item())
			writer.add_scalar("charts/episodic_return", episode_return, global_step)
			writer.add_scalar("charts/episodic_length", episode_length, global_step)
			writer.add_scalar("losses/policy_loss", policy_loss.item(), global_step)
			writer.add_scalar("losses/value_loss", value_loss.item(), global_step)
			writer.add_scalar("charts/SPS", global_step / (time.time() - start_time), global_step)
			print(
				f"episode={episode}, step={global_step}, return={episode_return:.1f}, "
				f"length={episode_length}"
			)

	model_path = Path("rom_cleanrl_vpg_cartpole.cleanrl_model")
	torch.save({"env_id": "CartPole-v1", "model_state_dict": policy_network.state_dict()}, model_path)
	print(f"Training complete. Model saved to {model_path}")
	env.close()
	writer.close()


if __name__ == "__main__":
	train(parse_args())
