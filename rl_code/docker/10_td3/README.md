# TD3 - Twin Delayed DDPG on MuJoCo Hopper

ဒီ folder မှာ MuJoCo ရဲ့ `Hopper-v4` continuous-control environment အတွက် **Twin Delayed Deep Deterministic Policy Gradient (TD3)** implementation နှစ်မျိုးပါဝင်ပါတယ်။

- `train_hopper_cleanrl_td3.py`: PyTorch + CleanRL-style TD3
- `train_hopper_sb3_td3.py`: Stable-Baselines3 built-in TD3

## MuJoCo Install

Docker image အသစ် build လုပ်လျှင် `gymnasium[mujoco]` ကို Dockerfile က install လုပ်ပေးပါတယ်။ Local environment တွင် run မည်ဆိုလျှင်:

```bash
pip install "gymnasium[mujoco]==0.29.1" "mujoco==2.3.7" stable-baselines3==2.3.2 tensorboard
```

`Gymnasium 0.29.1` ၏ human renderer သည် MuJoCo 3.x API နှင့်မကိုက်နိုင်သောကြောင့် `mujoco==2.3.7` ကို pin လုပ်ထားပါတယ်။

## Hopper-v4 Spaces

Hopper သည် torso, thigh, leg, foot ပါသော 2D one-legged robot ဖြစ်ပြီး action 3 ခုကို torque အဖြစ်ထုတ်ပေးရပါတယ်။ Episode တစ်ခုလျှင် အများဆုံး 1,000 steps ရှိပါတယ်။

| Space | Shape | Range | အဓိပ္ပါယ် |
|---|---:|---|---|
| Full simulator state | `(12,)` | `Box(-inf, inf, (12,))` | `qpos` 6 ခု + `qvel` 6 ခု |
| Observation | `(11,)` | `Box(-inf, inf, (11,))` | Full state မှ forward x-position ကိုဖယ်ထားသော `qpos[1:]` 5 ခု + `qvel` 6 ခု |
| Action | `(3,)` | `Box(-1.0, 1.0, (3,))` | thigh, leg, foot actuator torque controls |

Environment API တွင် state/observation bound များက unbounded (`-inf` မှ `inf`) ဖြစ်ပါတယ်။ Reset အစတွင် simulator state components များကို default pose အနီး uniform noise `[-0.005, 0.005]` ဖြင့် initialize လုပ်ပါတယ်။

## TD3 အဓိက အကြံဉာဏ်

TD3 သည် DDPG ကို instability လျော့အောင် အောက်ပါ 3 ခုထည့်သွင်းထားပါတယ်။

1. **Twin critics:** $Q_1$, $Q_2$ နှစ်ခု train လုပ်ပြီး target အတွက် အနိမ့်ဆုံးကိုသုံးသည်။
2. **Delayed policy update:** Critic update 2 ကြိမ်လျှင် actor နှင့် target networks ကို 1 ကြိမ်သာ update လုပ်သည်။
3. **Target policy smoothing:** Target actor action ပေါ် clipped Gaussian noise ထည့်သည်။

$$
y = r + \gamma(1-d) \min_{i=1,2} Q'_i(s', \mathrm{clip}(\mu'(s') + \epsilon))
$$

## Global Parameters

Training parameters များအားလုံးကို training script တစ်ခုစီ၏ အပေါ်ဆုံးတွင် global constants အဖြစ်ထားထားပြီး command line argument ဖြင့် override လုပ်နိုင်သော value များလည်းပါရှိပါတယ်။

| Parameter | Default | အဓိပ္ပါယ် |
|---|---:|---|
| `ENV_ID` | `Hopper-v4` | MuJoCo environment |
| `TOTAL_TIMESTEPS` | `1,000,000` | Training environment steps |
| `LEARNING_RATE` | `3e-4` | Adam learning rate |
| `BUFFER_SIZE` | `1,000,000` | Replay-buffer capacity |
| `GAMMA` | `0.99` | Discount factor |
| `TAU` | `0.005` | Polyak target soft-update coefficient |
| `BATCH_SIZE` | `256` | Replay batch size |
| `EXPLORATION_NOISE` | `0.1` | Online action Gaussian noise |
| `POLICY_NOISE` | `0.2` | Target-policy smoothing noise |
| `NOISE_CLIP` | `0.5` | Target-policy noise clipping bound |
| `LEARNING_STARTS` | `25,000` | Random-action warm-up steps |
| `POLICY_FREQUENCY` / `POLICY_DELAY` | `2` | Actor/target update interval |
| `HIDDEN_SIZE` | `256` | CleanRL actor/critic hidden units |
| `SEED` | `1` | Random seed |

## Run Training

```bash
cd docker/10_td3
python3 train_hopper_cleanrl_td3.py
python3 train_hopper_sb3_td3.py
```

ဥပမာ CleanRL hyperparameter override:

```bash
python3 train_hopper_cleanrl_td3.py --learning-rate 3e-4 --policy-noise 0.2 --noise-clip 0.5
```

CleanRL model ကို `rom_cleanrl_td3_hopper.cleanrl_model`၊ SB3 model ကို `rom_sb3_td3_hopper.zip` အဖြစ်သိမ်းပါတယ်။ TensorBoard logs များကို သက်ဆိုင်ရာ `*_td3_hopper_tensorboard/` folders ထဲမှာရေးပါတယ်။

## Run Evaluation

```bash
cd docker/10_td3
python3 test_cleanrl_hopper.py
python3 test_sb3_hopper.py
```

Evaluation တွင် `render_mode="human"` ဖြင့် Hopper ကိုပြပြီး deterministic policy action ကိုသုံးပါတယ်။