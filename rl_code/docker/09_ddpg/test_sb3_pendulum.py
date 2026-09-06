import time

import gymnasium as gym
from stable_baselines3 import DDPG


rom_time_check = True
env = gym.make("Pendulum-v1", render_mode="human")
model = DDPG.load("rom_sb3_ddpg_pendulum")

if rom_time_check:
	start_time = time.perf_counter()

observation, info = env.reset()
done = False
step_count = 0
total_reward = 0.0
while not done:
	action, _states = model.predict(observation, deterministic=True)
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
