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


TOTAL_TIMESTEPS = 300_000  # training loop ကို total ဘယ်နှစ် environment step run မလဲ
NUM_STEPS = 20
LEARNING_RATE = 7e-4
GAMMA = 0.99
GAE_LAMBDA = 0.95

# policy(actor) နဲ့ value(critic) head နှှစ်ခုကို shared layer တစ်ခုတည်းက ထုတ်ပေးတဲ့ actor-critic network
class ActorCriticNetwork(nn.Module):
	def __init__(self, env: gym.Env) -> None:
		super().__init__()
		observation_size = int(np.prod(env.observation_space.shape))
		action_count = env.action_space.n
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


# rollout တစ်ခုလုံးပေ္းရှိတဲ့ reward/value/done တွေကို Generalized Advantage Estimation (GAE) နဲ့အတွက် advantage တွက်ချက်တယ်
def compute_gae(
	rewards: torch.Tensor,
	values: torch.Tensor,
	dones: torch.Tensor,
	bootstrap_value: torch.Tensor,
	gamma: float,
	gae_lambda: float,
) -> torch.Tensor:
	"""lambda=0 reduces to one-step TD advantage; lambda=1 reduces to Monte-Carlo advantage."""
	advantages = torch.zeros_like(rewards)
	last_gae = torch.zeros_like(bootstrap_value)
	next_value = bootstrap_value
	for t in reversed(range(len(rewards))):
		next_non_terminal = 1.0 - dones[t]
		delta = rewards[t] + gamma * next_value * next_non_terminal - values[t]
		last_gae = delta + gamma * gae_lambda * next_non_terminal * last_gae
		advantages[t] = last_gae
		next_value = values[t]
	return advantages


# CartPole environment ကို episode statistics wrapper နဲ့ ဖန်တီးတယ်
def make_env() -> gym.Env:
	return gym.wrappers.RecordEpisodeStatistics(gym.make("CartPole-v1"))


# command-line arguments (num-steps, learning rate, gamma, gae-lambda, entropy/value coef, seed) တွေကို parse လုပ်တယ်
def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Train CleanRL-style actor-critic with GAE on CartPole.")
	parser.add_argument("--num-steps", type=int, default=NUM_STEPS)
	parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE)
	parser.add_argument("--gamma", type=float, default=GAMMA)
	parser.add_argument("--gae-lambda", type=float, default=GAE_LAMBDA)
	parser.add_argument("--entropy-coef", type=float, default=0.01)
	parser.add_argument("--value-loss-coef", type=float, default=0.5)
	parser.add_argument("--max-grad-norm", type=float, default=0.5)
	parser.add_argument("--seed", type=int, default=1)
	parser.add_argument("--cuda", action=argparse.BooleanOptionalAction, default=True)
	return parser.parse_args()


# n-step rollout ကောက်ပြီး GAE ကိုပှင့်လုပ်တဲ့ single-env actor-critic ကို train တယ်
def train(args: argparse.Namespace) -> None:
	random.seed(args.seed)
	np.random.seed(args.seed)
	torch.manual_seed(args.seed)

	device = torch.device("cuda" if args.cuda and torch.cuda.is_available() else "cpu")
	run_name = f"CartPole-v1__cleanrl_gae__{args.seed}__{int(time.time())}"
	writer = SummaryWriter(Path("cleanrl_gae_cartpole_tensorboard") / run_name)

	env = make_env()
	env.action_space.seed(args.seed)
	network = ActorCriticNetwork(env).to(device)
	optimizer = optim.Adam(network.parameters(), lr=args.learning_rate)

	observation, _ = env.reset(seed=args.seed)
	global_step = 0
	start_time = time.time()
	num_updates = TOTAL_TIMESTEPS // args.num_steps

	for update in range(num_updates):
		log_probs, values, entropies, rewards, dones = [], [], [], [], []

		# num_steps step အထိ environment ကို run ပြီး trajectory ကောက်ယူ
		for _ in range(args.num_steps):
			observation_tensor = torch.as_tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)
			logits, value = network(observation_tensor)
			distribution = Categorical(logits=logits)
			action = distribution.sample()

			next_observation, reward, terminated, truncated, info = env.step(int(action.item()))
			done = terminated or truncated

			log_probs.append(distribution.log_prob(action).squeeze(0))
			values.append(value.squeeze(0))
			entropies.append(distribution.entropy().squeeze(0))
			rewards.append(reward)
			dones.append(float(done))

			observation = next_observation
			global_step += 1

			if done:
				if "episode" in info:
					episode_return = float(np.asarray(info["episode"]["r"]).item())
					episode_length = int(np.asarray(info["episode"]["l"]).item())
					writer.add_scalar("charts/episodic_return", episode_return, global_step)
					writer.add_scalar("charts/episodic_length", episode_length, global_step)
					print(
						f"step={global_step}, return={episode_return:.1f}, length={episode_length}"
					)
				observation, _ = env.reset()

		with torch.no_grad():
			observation_tensor = torch.as_tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)
			_, bootstrap_value = network(observation_tensor)
			bootstrap_value = bootstrap_value.squeeze(0)

		rewards_tensor = torch.as_tensor(rewards, dtype=torch.float32, device=device)
		dones_tensor = torch.as_tensor(dones, dtype=torch.float32, device=device)
		values_tensor = torch.stack(values)
		log_probs_tensor = torch.stack(log_probs)
		entropies_tensor = torch.stack(entropies)

		advantages = compute_gae(
			rewards_tensor, values_tensor, dones_tensor, bootstrap_value, args.gamma, args.gae_lambda
		)
		returns = advantages + values_tensor.detach()

		# policy loss (advantage-weighted log_prob) + value loss (MSE) + entropy bonus အတွက် total loss ပေါင်တယ်
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

	model_path = Path("rom_cleanrl_gae_cartpole.cleanrl_model")
	torch.save({"env_id": "CartPole-v1", "model_state_dict": network.state_dict()}, model_path)
	print(f"Training complete. Model saved to {model_path}")
	env.close()
	writer.close()


if __name__ == "__main__":
	train(parse_args())
