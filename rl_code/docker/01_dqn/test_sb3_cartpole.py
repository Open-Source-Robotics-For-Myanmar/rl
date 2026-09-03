from stable_baselines3 import DQN
import gymnasium as gym
import time

rom_time_check = True

# Window ဖြင့် CartPole gameplay ကို ပြရန် environment ဖန်တီးသည်။
env = gym.make("CartPole-v1", render_mode="human")
# သိမ်းထားသော trained DQN model ကို load လုပ်သည်။
model = DQN.load("rom_dqn_cartpole")

# Episode အသစ် စတင်ပြီး initial observation ကို ရယူသည်။
if rom_time_check:
    start_time = time.perf_counter()
observation, info = env.reset()
done = False
step_count = 0
total_reward = 0
while not done:
    # Model က လက်ရှိ observation အပေါ်မူတည်၍ အကောင်းဆုံး action ကို ရွေးသည်။
    action = model.predict(observation, deterministic=True)[0]
    # Action ကို environment သို့ပို့ပြီး state အသစ်နှင့် episode အခြေအနေကို ရယူသည်။
    observation, reward, terminated, truncated, info = env.step(action)
    step_count += 1
    total_reward += reward
    # Pole ကျခြင်း သို့မဟုတ် maximum steps ရောက်လျှင် episode ပြီးဆုံးသည်။
    done = terminated or truncated

# Test ပြီးဆုံးချိန်တွင် episode ကြာချိန်နှင့် reward ကို ပြသည်။
print(f"Testing complete. Steps: {step_count}, Reward: {total_reward}")
if rom_time_check:
    elapsed_time = time.perf_counter() - start_time
    print(f"Episode duration: {step_count} environment steps ({elapsed_time:.2f} seconds)")
input("Press Enter to close...")
# Environment နှင့် CartPole window ကို ပိတ်သည်။
env.close()