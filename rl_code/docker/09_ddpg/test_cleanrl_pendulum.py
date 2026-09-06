import time

import gymnasium as gym
import torch

from train_pendulum_cleanrl_ddpg import Actor


rom_time_check = True
env = gym.make("Pendulum-v1", render_mode="human")
network_env = gym.vector.SyncVectorEnv([lambda: gym.make("Pendulum-v1")])
actor = Actor(network_env)
network_env.close()
checkpoint = torch.load("rom_cleanrl_ddpg_pendulum.cleanrl_model", map_location="cpu")
actor.load_state_dict(checkpoint["model_state_dict"])
actor.eval()

if rom_time_check:
	start_time = time.perf_counter()

observation, info = env.reset()
done = False
step_count = 0
total_reward = 0.0
while not done:
	observation_tensor = torch.as_tensor(observation, dtype=torch.float32).unsqueeze(0)
	with torch.no_grad():
		action = actor(observation_tensor).squeeze(0).numpy()
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
