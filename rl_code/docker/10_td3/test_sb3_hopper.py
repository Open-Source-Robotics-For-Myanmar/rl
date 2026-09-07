import time

import gymnasium as gym
from stable_baselines3 import TD3

ENV_ID = "Hopper-v4"
MODEL_PATH = "rom_sb3_td3_hopper"
ROM_TIME_CHECK = True

env = gym.make(ENV_ID, render_mode="human")
model = TD3.load(MODEL_PATH)
start_time = time.perf_counter() if ROM_TIME_CHECK else None
observation, _ = env.reset()
done = False
step_count = 0
total_reward = 0.0
while not done:
	action, _states = model.predict(observation, deterministic=True)
	observation, reward, terminated, truncated, _ = env.step(action)
	step_count += 1
	total_reward += reward
	done = terminated or truncated

print(f"Testing complete. Steps: {step_count}, Reward: {total_reward:.1f}")
if ROM_TIME_CHECK:
	print(f"Episode duration: {step_count} environment steps ({time.perf_counter() - start_time:.2f} seconds)")
input("Press Enter to close...")
env.close()