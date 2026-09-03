import gymnasium as gym
from stable_baselines3 import DQN

# CartPole environment ကို ဖန်တီးသည်။
env = gym.make("CartPole-v1")

# MLP policy သုံးသော DQN model ကို TensorBoard log path နှင့် ဖန်တီးသည်။
model = DQN("MlpPolicy", env, verbose=1, tensorboard_log="./dqn_cartpole_tensorboard/")

# Environment နှင့် action 100,000 ကြိမ် အပြန်အလှန်လုပ်၍ model ကို train လုပ်သည်။
model.learn(total_timesteps=100000)

# Train ပြီးသော model ကို နောက်မှ test လုပ်ရန် သိမ်းဆည်းသည်။
model.save("rom_dqn_cartpole")

# Training ပြီးဆုံးကြောင်း terminal တွင် ပြသည်။
print("Training complete.")