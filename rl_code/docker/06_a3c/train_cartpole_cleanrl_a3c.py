import argparse
import random
import time
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
import torch.multiprocessing as mp
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from torch.utils.tensorboard import SummaryWriter

TOTAL_EPISODES = 2_000  # training loop ကို total ဘယ်နှစ် episode run မလဲ


# policy(actor) နဲ့ value(critic) head နှှစ်ခုကို shared layer တစ်ခုတည်းက ထုတ်ပေးတဲ့ actor-critic network
class ActorCriticNetwork(nn.Module):
	def __init__(self, observation_size: int, action_count: int) -> None:
		super().__init__()
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


class SharedAdam(optim.Adam):
	"""Adam whose per-parameter moment buffers live in shared memory (Mnih et al., 2016)."""

	def __init__(self, params, lr: float = 1e-4) -> None:
		super().__init__(params, lr=lr)
		for group in self.param_groups:
			for parameter in group["params"]:
				state = self.state[parameter]
				state["step"] = torch.zeros(1)
				state["exp_avg"] = torch.zeros_like(parameter.data)
				state["exp_avg_sq"] = torch.zeros_like(parameter.data)

	def share_memory(self) -> None:
		for group in self.param_groups:
			for parameter in group["params"]:
				state = self.state[parameter]
				state["step"].share_memory_()
				state["exp_avg"].share_memory_()
				state["exp_avg_sq"].share_memory_()


# A3C worker (process) တစ်ခုစီက local network ကနေ n_steps rollout ကောက်ပြီး gradient ကို global network ဆီ async တင်ပို့ (Hogwild! update)
def worker(
	rank: int,
	args: argparse.Namespace,
	observation_size: int,
	action_count: int,
	global_network: ActorCriticNetwork,
	optimizer: SharedAdam,
	global_episode_counter,
	run_name: str,
) -> None:
	torch.manual_seed(args.seed + rank)
	env = gym.wrappers.RecordEpisodeStatistics(gym.make("CartPole-v1"))
	env.action_space.seed(args.seed + rank)

	local_network = ActorCriticNetwork(observation_size, action_count)
	writer = SummaryWriter(Path("cleanrl_a3c_cartpole_tensorboard") / run_name) if rank == 0 else None

	observation, _ = env.reset(seed=args.seed + rank)

	while global_episode_counter.value < TOTAL_EPISODES:
		local_network.load_state_dict(global_network.state_dict())
		log_probs, values, entropies, rewards = [], [], [], []
		done = False
		info: dict = {}

		for _ in range(args.n_steps):
			observation_tensor = torch.as_tensor(observation, dtype=torch.float32).unsqueeze(0)
			logits, value = local_network(observation_tensor)
			distribution = Categorical(logits=logits)
			action = distribution.sample()

			observation, reward, terminated, truncated, info = env.step(int(action.item()))
			done = terminated or truncated

			log_probs.append(distribution.log_prob(action).squeeze(0))
			values.append(value.squeeze(0))
			entropies.append(distribution.entropy().squeeze(0))
			rewards.append(reward)

			if done:
				with global_episode_counter.get_lock():
					global_episode_counter.value += 1
					current_episode = global_episode_counter.value
				if "episode" in info:
					episode_return = float(np.asarray(info["episode"]["r"]).item())
					episode_length = int(np.asarray(info["episode"]["l"]).item())
					print(
						f"worker={rank}, episode={current_episode}, return={episode_return:.1f}, "
						f"length={episode_length}"
					)
					if writer is not None:
						writer.add_scalar("charts/episodic_return", episode_return, current_episode)
						writer.add_scalar("charts/episodic_length", episode_length, current_episode)
				observation, _ = env.reset()
				break

		bootstrap_value = 0.0
		if not done:
			with torch.no_grad():
				observation_tensor = torch.as_tensor(observation, dtype=torch.float32).unsqueeze(0)
				_, bootstrap_value_tensor = local_network(observation_tensor)
				bootstrap_value = bootstrap_value_tensor.item()

		returns = []
		running_return = bootstrap_value
		for reward in reversed(rewards):
			running_return = reward + args.gamma * running_return
			returns.insert(0, running_return)
		returns_tensor = torch.as_tensor(returns, dtype=torch.float32)
		values_tensor = torch.stack(values)
		advantages = returns_tensor - values_tensor

		policy_loss = -(torch.stack(log_probs) * advantages.detach()).sum()
		value_loss = advantages.pow(2).sum()
		entropy_loss = torch.stack(entropies).sum()
		loss = policy_loss + args.value_loss_coef * value_loss - args.entropy_coef * entropy_loss

		optimizer.zero_grad()
		loss.backward()
		nn.utils.clip_grad_norm_(local_network.parameters(), args.max_grad_norm)

		# asynchronous "Hogwild" update: apply the local worker's gradients directly to the shared global network
		for local_parameter, global_parameter in zip(local_network.parameters(), global_network.parameters()):
			global_parameter.grad = local_parameter.grad
		optimizer.step()

	env.close()
	if writer is not None:
		writer.close()


# command-line arguments (worker count, n-steps, learning rate, entropy/value coef, seed) တို့ကို parse လုပ်တယ်
def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Train CleanRL-style A3C (async parallel actor-critic) on CartPole.")
	parser.add_argument("--num-workers", type=int, default=4)
	parser.add_argument("--n-steps", type=int, default=20)
	parser.add_argument("--learning-rate", type=float, default=1e-4)
	parser.add_argument("--gamma", type=float, default=0.99)
	parser.add_argument("--entropy-coef", type=float, default=0.01)
	parser.add_argument("--value-loss-coef", type=float, default=0.5)
	parser.add_argument("--max-grad-norm", type=float, default=40.0)
	parser.add_argument("--seed", type=int, default=1)
	return parser.parse_args()


# global network ကို shared memory ထဲထားပြီး worker process များစွာ (multiprocessing) ကို parallel run ကာ A3C အဖြစ် train တယ်
def train(args: argparse.Namespace) -> None:
	random.seed(args.seed)
	np.random.seed(args.seed)
	torch.manual_seed(args.seed)

	probe_env = gym.make("CartPole-v1")
	observation_size = int(np.prod(probe_env.observation_space.shape))
	action_count = probe_env.action_space.n
	probe_env.close()

	global_network = ActorCriticNetwork(observation_size, action_count)
	global_network.share_memory()
	optimizer = SharedAdam(global_network.parameters(), lr=args.learning_rate)
	optimizer.share_memory()
	global_episode_counter = mp.Value("i", 0)
	run_name = f"CartPole-v1__cleanrl_a3c__{args.seed}__{int(time.time())}"

	processes = []
	for rank in range(args.num_workers):
		process = mp.Process(
			target=worker,
			args=(
				rank,
				args,
				observation_size,
				action_count,
				global_network,
				optimizer,
				global_episode_counter,
				run_name,
			),
		)
		process.start()
		processes.append(process)
	for process in processes:
		process.join()

	model_path = Path("rom_cleanrl_a3c_cartpole.cleanrl_model")
	torch.save({"env_id": "CartPole-v1", "model_state_dict": global_network.state_dict()}, model_path)
	print(f"Training complete. Model saved to {model_path}")


if __name__ == "__main__":
	mp.set_start_method("spawn", force=True)
	train(parse_args())
