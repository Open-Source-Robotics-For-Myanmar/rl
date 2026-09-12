# 03 — MuJoCo Continuous Control ပြန်လေ့လာခြင်း

`rl_code/docker` ထဲမှာ DDPG/TD3/SAC/PPO ကို Pendulum/LunarLander လို "light" envs တွေနဲ့ပဲ train ခဲ့တာမို့—ဒီ directory မှာတော့ **MuJoCo** ရဲ့ higher-dimensional continuous control envs (Hopper, HalfCheetah, Walker2d, Ant, Humanoid) နဲ့ algorithm တွေကို ပြန်စမ်းပါမယ်။ Isaac Lab (04) ကို မစခင် MuJoCo physics/API ကို အရင်ရင်းနှီးအောင် လုပ်ဖို့ ရည်ရွယ်ထားတဲ့ အဆင့်ပါ။

## ဒီ directory မှာ လုပ်မယ့်အရာများ

1. **Install & sanity check**
   - `pip install "gymnasium[mujoco]"` ထည့်ပြီး `Hopper-v4`, `HalfCheetah-v4`, `Ant-v4` စတာတွေကို random policy နဲ့ run ကြည့်ပြီး render/observation/action space စစ်။
2. **Existing algorithms ကို MuJoCo envs ပေါ် ပြောင်းသုံး**
   - `09_ddpg`, `10_td3`, `11_sac`, `12_ppo` code style ကို base ယူပြီး `HalfCheetah-v4` / `Hopper-v4` အတွက် train script ရေး (CleanRL-style + SB3 comparison ပုံစံအတိုင်း)။
3. **Observation/reward scaling**
   - MuJoCo envs တွေမှာ observation normalization (running mean/std) နဲ့ reward scaling က training stability အတွက် အရေးကြီးမှုကို လက်တွေ့ compare လုပ်ကြည့်ခြင်း (normalize on/off ၂ မျိုး)။
4. **Harder env** — `Humanoid-v4` ကို train ကြည့်ပြီး dimensionality မြင့်လာတာနဲ့အမျှ training time/stability ဘယ်လိုပြောင်းလဲသလဲ မှတ်ချက်ချ။
5. **Comparison notes** — TensorBoard reward curves ကို algorithm အလိုက်/env အလိုက် နှိုင်းယှဉ်ပြီး learning ရသလဲ မှတ်တမ်းတင်။

## အဓိက concepts

- Continuous action clipping & squashing (`tanh` for SAC/TD3 deterministic policies)
- Gaussian policy head (mean + log-std) vs deterministic policy + exploration noise
- Observation/reward normalization wrapper (`gymnasium.wrappers.NormalizeObservation`, `NormalizeReward`)
- Episode termination vs truncation (`terminated` / `truncated`) semantics MuJoCo envs အတွက်

## Deliverables

- [ ] MuJoCo envs (Hopper, HalfCheetah, Ant, Humanoid) train script အနည်းဆုံး ၂ ခု (algorithm ၂ မျိုးဖြင့်)
- [ ] Reward curve comparison notes/screenshots
- [ ] "MuJoCo vs simple gym env" ခက်ခဲချက် ကွာခြားချက် အတွက် မှတ်စု (short summary)
