import time

import gymnasium as gym
import pybullet_envs_gymnasium  # noqa: F401  (registers HalfCheetahBulletEnv-v0)
import torch

from train_halfcheetah_cleanrl_sac import Actor, ENV_ID, MODEL_PATH

ROM_TIME_CHECK = True
SIMULATION_DELAY = 1 / 30
env = gym.make(ENV_ID, render_mode="human")
network_env = gym.vector.SyncVectorEnv([lambda: gym.make(ENV_ID)])
actor = Actor(network_env)
network_env.close()
checkpoint = torch.load(MODEL_PATH, map_location="cpu")
actor.load_state_dict(checkpoint["model_state_dict"])
actor.eval()

start_time = time.perf_counter() if ROM_TIME_CHECK else None
observation, _ = env.reset()
done = False
step_count = 0
total_reward = 0.0
while not done:
	with torch.no_grad():
		_, _, action = actor.get_action(torch.as_tensor(observation, dtype=torch.float32).unsqueeze(0))
	observation, reward, terminated, truncated, _ = env.step(action.squeeze(0).numpy())
	time.sleep(SIMULATION_DELAY)
	step_count += 1
	total_reward += reward
	done = terminated or truncated

print(f"Testing complete. Steps: {step_count}, Reward: {total_reward:.1f}")
if ROM_TIME_CHECK:
	print(f"Episode duration: {step_count} environment steps ({time.perf_counter() - start_time:.2f} seconds)")
input("Press Enter to close...")
env.close()
