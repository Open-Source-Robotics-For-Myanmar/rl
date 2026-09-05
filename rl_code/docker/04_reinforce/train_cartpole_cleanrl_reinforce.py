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

TOTAL_EPISODES = 2_000  # training loop ကို total ဘယ်နှစ် episode run မလဲ


# action ကို ရွေးချယ်ဖို့ probability ထုတ်ပေးတဲ့ policy network (state -> action logits)
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

	# observation ကနေ action logits တွေကို forward pass ဖြင့် တွက်ချက်ခြင်း
	def forward(self, observation: torch.Tensor) -> torch.Tensor:
		return self.network(observation)

	# logits ကနေ probability distribution ဆောက်ပြီး action တစ်ခု sample ယူပြီး log_prob ကို ပြန်ပေးခြင်း
	def get_action(self, observation: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
		logits = self(observation)
		distribution = Categorical(logits=logits)
		action = distribution.sample()
		return action, distribution.log_prob(action)


# episode တစ်ခုလုံးရဲ့ reward list ကနေ gamma-discounted cumulative return တွေကို နောက်ကနေရှေ့ (reverse) တွက်ခြင်း
def discounted_returns(rewards: list[float], gamma: float) -> np.ndarray:
	returns = np.zeros(len(rewards), dtype=np.float32)
	running_return = 0.0
	for step in reversed(range(len(rewards))):
		running_return = rewards[step] + gamma * running_return
		returns[step] = running_return
	return returns


# CartPole environment ကို episode statistics wrapper နဲ့ ဖန်တီးခြင်း
def make_env() -> gym.Env:
	return gym.wrappers.RecordEpisodeStatistics(gym.make("CartPole-v1"))


# command-line arguments (learning rate, gamma, seed, စသဖြင့်) တွေကို parse လုပ်ခြင်း
def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Train CleanRL-style REINFORCE on CartPole.")
	parser.add_argument("--learning-rate", type=float, default=1e-3)
	parser.add_argument("--gamma", type=float, default=0.99)
	parser.add_argument("--normalize-returns", action=argparse.BooleanOptionalAction, default=True)
	parser.add_argument("--seed", type=int, default=1)
	parser.add_argument("--cuda", action=argparse.BooleanOptionalAction, default=True)
	return parser.parse_args()


# REINFORCE algorithm ဖြင့် policy network ကို train လုပ်တဲ့ main function
def train(args: argparse.Namespace) -> None:
	random.seed(args.seed)
	np.random.seed(args.seed)
	torch.manual_seed(args.seed)

	device = torch.device("cuda" if args.cuda and torch.cuda.is_available() else "cpu")
	run_name = f"CartPole-v1__cleanrl_reinforce__{args.seed}__{int(time.time())}"
	writer = SummaryWriter(Path("cleanrl_reinforce_cartpole_tensorboard") / run_name)

	env = make_env()
	env.action_space.seed(args.seed)
	policy_network = PolicyNetwork(env).to(device)
	optimizer = optim.Adam(policy_network.parameters(), lr=args.learning_rate)

	global_step = 0
	start_time = time.time()
	for episode in range(TOTAL_EPISODES):
		observation, _ = env.reset(seed=args.seed + episode)
		log_probs: list[torch.Tensor] = []
		rewards: list[float] = []
		done = False
		info: dict = {}

		# episode ပြီးဆုံးတဲ့အထိ action ရွေးချယ်ပြီး log_prob နဲ့ reward တွေကို စုသိမ်းခြင်း
		while not done:
			observation_tensor = torch.as_tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)
			action, log_prob = policy_network.get_action(observation_tensor)
			observation, reward, terminated, truncated, info = env.step(int(action.item()))
			log_probs.append(log_prob)
			rewards.append(float(reward))
			done = terminated or truncated
			global_step += 1

		# discounted return တွက်ပြီး optional ဖြစ်တဲ့ normalization လုပ်ခြင်း
		returns = discounted_returns(rewards, args.gamma)
		returns_tensor = torch.as_tensor(returns, device=device)
		if args.normalize_returns:
			returns_tensor = (returns_tensor - returns_tensor.mean()) / (returns_tensor.std() + 1e-8)

		# policy gradient loss ကို တွက်ချက်ပြီး backpropagation ဖြင့် policy network ကို update လုပ်ခြင်း
		policy_loss = -(torch.cat(log_probs) * returns_tensor).sum()
		optimizer.zero_grad()
		policy_loss.backward()
		optimizer.step()

		if "episode" in info:
			episode_return = float(np.asarray(info["episode"]["r"]).item())
			episode_length = int(np.asarray(info["episode"]["l"]).item())
			writer.add_scalar("charts/episodic_return", episode_return, global_step)
			writer.add_scalar("charts/episodic_length", episode_length, global_step)
			writer.add_scalar("losses/policy_loss", policy_loss.item(), global_step)
			writer.add_scalar("charts/SPS", global_step / (time.time() - start_time), global_step)
			print(
				f"episode={episode}, step={global_step}, return={episode_return:.1f}, "
				f"length={episode_length}"
			)

	model_path = Path("rom_cleanrl_reinforce_cartpole.cleanrl_model")
	torch.save({"env_id": "CartPole-v1", "model_state_dict": policy_network.state_dict()}, model_path)
	print(f"Training complete. Model saved to {model_path}")
	env.close()
	writer.close()


if __name__ == "__main__":
	train(parse_args())
