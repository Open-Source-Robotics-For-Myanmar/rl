import time

import gymnasium as gym
import torch

from train_cartpole_cleanrl_vpg import PolicyNetwork


rom_time_check = True
env = gym.make("CartPole-v1", render_mode="human")
policy_network = PolicyNetwork(env)
checkpoint = torch.load("rom_cleanrl_vpg_cartpole.cleanrl_model", map_location="cpu")
policy_network.load_state_dict(checkpoint["model_state_dict"])
policy_network.eval()

if rom_time_check:
	start_time = time.perf_counter()

observation, info = env.reset()
done = False
step_count = 0
total_reward = 0.0
while not done:
	observation_tensor = torch.as_tensor(observation, dtype=torch.float32).unsqueeze(0)
	with torch.no_grad():
		action = int(torch.argmax(policy_network(observation_tensor), dim=1).item())
	observation, reward, terminated, truncated, info = env.step(action)
	step_count += 1
	total_reward += reward
	done = terminated or truncated

print(f"Testing complete. Steps: {step_count}, Reward: {total_reward:.1f}")
if rom_time_check:
	elapsed_time = time.perf_counter() - start_time
	print(f"Episode duration: {step_count} environment steps ({elapsed_time:.2f} seconds)")

input("Press Enter to close...")
env.close()
