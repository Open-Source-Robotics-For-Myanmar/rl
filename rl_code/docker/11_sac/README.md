# SAC - Soft Actor-Critic on PyBullet HalfCheetah

ဒီ folder မှာ PyBullet ရဲ့ `HalfCheetahBulletEnv-v0` continuous-control environment အတွက် **Soft Actor-Critic (SAC)** implementation နှစ်မျိုးပါဝင်ပါတယ်။

- `train_halfcheetah_cleanrl_sac.py`: PyTorch + CleanRL-style SAC (automatic entropy tuning ပါ)
- `train_halfcheetah_sb3_sac.py`: Stable-Baselines3 built-in SAC

## PyBullet Install

`HalfCheetahBulletEnv-v0` သည် MuJoCo မဟုတ်ဘဲ **PyBullet** engine ပေါ်တွင် implement လုပ်ထားသော environment ဖြစ်ပါတယ်။ Original `pybullet_envs` package က old `gym` API ကိုသုံးထားလို့ `gymnasium` နှင့်တိုက်ရိုက်မတွဲသောကြောင့် community-maintained fork ဖြစ်တဲ့ `pybullet_envs_gymnasium` ကိုသုံးပါတယ်။ Docker image အသစ် build လုပ်လျှင် Dockerfile က install လုပ်ပေးပါတယ်။ Local environment တွင် run မည်ဆိုလျှင်:

```bash
pip install pybullet==3.2.6 pybullet_envs_gymnasium==0.5.0 "gymnasium==0.29.1" stable-baselines3==2.3.2 tensorboard
```

Script တစ်ခုစီရဲ့ အပေါ်ဆုံးမှာ `import pybullet_envs_gymnasium` ကို side-effect import အဖြစ်ထားထားပြီး၊ ဒါက `HalfCheetahBulletEnv-v0` ကို gymnasium registry ထဲ register လုပ်ပေးပါတယ်။

## HalfCheetahBulletEnv-v0 Spaces

HalfCheetah သည် 2D planar cheetah-like robot ဖြစ်ပြီး joint torque 6 ခုကို actuator အဖြစ်ထုတ်ပေးရပါတယ်။ ရှေ့ကို အမြန်ဆုံးပြေးနိုင်အောင် energy cost နည်းနည်းနှင့် reward ပေးသည်။

| Space | Shape | Range | အဓိပ္ပါယ် |
|---|---:|---|---|
| Observation | `(26,)` | `Box(-inf, inf, (26,))` | Torso height/orientation + joint angles/velocities + foot-contact sensors |
| Action | `(6,)` | `Box(-1.0, 1.0, (6,))` | Thigh/shin/foot (front + back) actuator torque controls |

## SAC အဓိက အကြံဉာဏ်

SAC သည် TD3 ကဲ့သို့ twin critics နှင့် target networks ကိုသုံးပေမယ့် DDPG/TD3 ၏ deterministic policy အစား **maximum-entropy** stochastic (squashed-Gaussian) policy ကိုသုံးပါတယ်။ Objective တွင် expected return အပြင် policy entropy ကိုပါ maximize လုပ်ပြီး exploration ကို အလိုအလျောက်ထိန်းထားနိုင်စေပါတယ်။

$$
J(\pi) = \mathbb{E}\left[\sum_t r(s_t, a_t) + \alpha \mathcal{H}(\pi(\cdot|s_t))\right]
$$

1. **Squashed Gaussian actor:** Policy က mean/log-std ထုတ်ပြီး `tanh` squashing ဖြင့် action ကို bound `[-1, 1]` အတွင်းထားသည် (log-prob ကို `tanh` Jacobian correction ဖြင့်ပြင်ဆင်သည်)။
2. **Twin critics:** $Q_1$, $Q_2$ နှစ်ခု train လုပ်ပြီး target အတွက် entropy term နုတ်ပြီးနောက် အနိမ့်ဆုံးကိုသုံးသည်။
3. **Automatic entropy tuning:** $\alpha$ ကို fixed constant မဟုတ်ဘဲ target entropy (`-action_dim`) ကို ရောက်အောင် gradient ဖြင့် autotune လုပ်သည် (`log_alpha` parameter)။

$$
y = r + \gamma(1-d) \left[\min_{i=1,2} Q'_i(s', \tilde{a}') - \alpha \log \pi(\tilde{a}'|s')\right]
$$

## Global Parameters

Training parameters များအားလုံးကို training script တစ်ခုစီ၏ အပေါ်ဆုံးတွင် global constants အဖြစ်ထားထားပြီး command line argument ဖြင့် override လုပ်နိုင်သော value များလည်းပါရှိပါတယ်။

| Parameter | Default | အဓိပ္ပါယ် |
|---|---:|---|
| `ENV_ID` | `HalfCheetahBulletEnv-v0` | PyBullet environment |
| `TOTAL_TIMESTEPS` | `1,000,000` | Training environment steps |
| `POLICY_LEARNING_RATE` | `3e-4` | Actor Adam learning rate |
| `Q_LEARNING_RATE` | `1e-3` | Critic (+ alpha) Adam learning rate |
| `BUFFER_SIZE` | `1,000,000` | Replay-buffer capacity |
| `GAMMA` | `0.99` | Discount factor |
| `TAU` | `0.005` | Polyak target soft-update coefficient |
| `BATCH_SIZE` | `256` | Replay batch size |
| `LEARNING_STARTS` | `5,000` | Random-action warm-up steps |
| `POLICY_FREQUENCY` | `2` | Actor/alpha update interval (critic updates every step) |
| `TARGET_NETWORK_FREQUENCY` | `1` | Target-network soft-update interval |
| `ALPHA` | `0.2` | Fixed entropy coefficient (only used if `--no-autotune`) |
| `AUTOTUNE` | `True` | Automatically learn entropy coefficient $\alpha$ |
| `HIDDEN_SIZE` | `256` | CleanRL actor/critic hidden units |
| `LOG_STD_MIN` / `LOG_STD_MAX` | `-5` / `2` | Actor log-std clamping bounds |
| `SEED` | `1` | Random seed |

## Run Training

```bash
cd docker/11_sac
python3 train_halfcheetah_cleanrl_sac.py
python3 train_halfcheetah_sb3_sac.py
```

ဥပမာ CleanRL hyperparameter override:

```bash
python3 train_halfcheetah_cleanrl_sac.py --policy-learning-rate 3e-4 --no-autotune --alpha 0.2
```

CleanRL model ကို `rom_cleanrl_sac_halfcheetah.cleanrl_model`၊ SB3 model ကို `rom_sb3_sac_halfcheetah.zip` အဖြစ်သိမ်းပါတယ်။ TensorBoard logs များကို သက်ဆိုင်ရာ `*_sac_halfcheetah_tensorboard/` folders ထဲမှာရေးပါတယ်။

## Run Evaluation

```bash
cd docker/11_sac
python3 test_cleanrl_halfcheetah.py
python3 test_sb3_halfcheetah.py
```

Evaluation တွင် `render_mode="human"` ဖြင့် HalfCheetah ကိုပြပြီး deterministic action (squashed policy mean) ကိုသုံးပါတယ်။
