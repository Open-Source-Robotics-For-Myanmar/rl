from stable_baselines3 import DQN
import gymnasium as gym

# Window ဖြင့် CartPole gameplay ကို ပြရန် environment ဖန်တီးသည်။
env = gym.make("CartPole-v1", render_mode="human")
# သိမ်းထားသော trained DQN model ကို load လုပ်သည်။
model = DQN.load("rom_dqn_cartpole")

# Episode အသစ် စတင်ပြီး initial observation ကို ရယူသည်။
observation, info = env.reset()
done = False
while not done:
    # Model က လက်ရှိ observation အပေါ်မူတည်၍ အကောင်းဆုံး action ကို ရွေးသည်။
    action = model.predict(observation, deterministic=True)[0]
    # Action ကို environment သို့ပို့ပြီး state အသစ်နှင့် episode အခြေအနေကို ရယူသည်။
    observation, reward, terminated, truncated, info = env.step(action)
    # Pole ကျခြင်း သို့မဟုတ် maximum steps ရောက်လျှင် episode ပြီးဆုံးသည်။
    done = terminated or truncated

# Test ပြီးဆုံးကြောင်း ပြပြီး window မပိတ်မီ Enter စောင့်သည်။
print("Testing complete.")
input("Press Enter to close...")
# Environment နှင့် CartPole window ကို ပိတ်သည်။
env.close()