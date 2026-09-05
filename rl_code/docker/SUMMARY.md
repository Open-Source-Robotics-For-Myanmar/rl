# Summary

## CartPole ရဲ့ observation space က continuous လား?

CartPole-v1 ရဲ့ observation space က **continuous (Box)** ပါ — ဒါပေမဲ့ action space ကတော့ **discrete** ပါတယ်။

- **Observation**: `Box(4,)` — cart position, cart velocity, pole angle, pole angular velocity (float values, continuous range)
- **Action**: `Discrete(2)` — left (0) သို့မဟုတ် right (1) ပဲရှိတယ်

ဒါကြောင့် policy network ရဲ့ output ဟာ **`Categorical` distribution** (softmax logits) ဖြစ်နေတာပါ — input (observation) က continuous vector ဖြစ်နေပေမဲ့ output (action) က discrete ဖြစ်လို့ပါ။

MuJoCo ရဲ့ envs (Hopper, HalfCheetah, InvertedPendulum စသည်) ကတော့ observation **နှင့်** action နှစ်ခုစလုံး continuous ဖြစ်ပါတယ် — အဲ့တော့ policy network output ကို `Categorical` အစား `Normal` (Gaussian, mean + std) distribution သုံးရမှာ ဖြစ်ပါတယ်။

## အခုထိ (01_dqn ~ 08_a2c) MuJoCo သုံးရသေးလား?

မသုံးရသေးပါ — အခုထိ (`01_dqn` ကနေ `08_a2c` အထိ) **MuJoCo လုံးဝမသုံးရသေးပါဘူး**။ Algorithm အားလုံး **CartPole-v1** (classic control, discrete action) ပေါ်မှာပဲ implement လုပ်ထားတာပါ:

| Folder | Algorithm | Environment |
|---|---|---|
| `01_dqn` | DQN | CartPole-v1 |
| `02_ddqn` | DDQN | CartPole-v1 |
| `03_dueling_per` | Dueling DDQN + PER | CartPole-v1 |
| `04_reinforce` | REINFORCE | CartPole-v1 |
| `05_vpg` | VPG (+ SB3 A2C comparison) | CartPole-v1 |
| `06_a3c` | A3C | CartPole-v1 |
| `07_gae` | GAE actor-critic | CartPole-v1 |
| `08_a2c` | A2C (+ SB3 A2C comparison) | CartPole-v1 |

MuJoCo (continuous observation + continuous action, e.g. Hopper, HalfCheetah, InvertedPendulum) က ဒီနောက်ပိုင်း chapters (DDPG, TD3, SAC, PPO continuous-control စသည်) အတွက် သင့်တော်ပါတယ် — အဲ့အချိန်ကျမှ Gaussian policy head ပြောင်းသုံးရပါမယ်။
