# PPO - Proximal Policy Optimization on LunarLander

ဒီ folder မှာ `LunarLander-v2` discrete-action environment အတွက် **Proximal Policy Optimization (PPO)** implementation နှစ်မျိုးပါဝင်ပါတယ်။

- `train_lunarlander_cleanrl_ppo.py`: PyTorch + CleanRL-style PPO (clipped surrogate objective + GAE + multi-epoch minibatch update)
- `train_lunarlander_sb3_ppo.py`: Stable-Baselines3 built-in PPO

## Box2D Install

`LunarLander-v2` သည် `gymnasium` ရဲ့ `box2d` extra ပေါ်တွင် မှီခိုနေပါတယ်။ Docker image အသစ် build လုပ်လျှင် Dockerfile က install လုပ်ပေးပါတယ်။ Local environment တွင် run မည်ဆိုလျှင် (build အတွက် `swig` လိုအပ်ပါတယ်):

```bash
sudo apt-get install -y swig
pip install "gymnasium[box2d]==0.29.1" stable-baselines3==2.3.2 tensorboard
```

## LunarLander-v2 Spaces

Agent သည် lander ကို flag နှစ်ခုကြားရှိ landing pad ပေါ်တွင် လုံခြုံစွာ ချရောက်စေရမည်ဖြစ်သည်။ Reward သည် landing pad နှင့်ကွာဟမှု၊ velocity၊ angle နှင့် fuel သုံးစွဲမှုပေါ်မူတည်၍ ပေးပြီး environment ကို "solve" ပြီဟုမှတ်ယူရန် average reward **200** ရရန် လိုအပ်ပါတယ်။ Episode step limit သည် **1,000** ဖြစ်သည်။

| Space | Shape | Range/Values | အဓိပ္ပါယ် |
|---|---:|---|---|
| Observation | `(8,)` | `Box(-inf, inf, (8,))` | x/y coordinates, x/y velocity, angle, angular velocity, leg-1/leg-2 ground-contact booleans |
| Action | `Discrete(4)` | `{0, 1, 2, 3}` | `0`=do nothing, `1`=fire left engine, `2`=fire main engine, `3`=fire right engine |

## PPO အဓိက အကြံဉာဏ်

PPO သည် A2C ကဲ့သို့ပင် on-policy actor-critic ဖြစ်ပြီး GAE ဖြင့် advantage တွက်ပါတယ်၊ ဒါပေမယ့် rollout တစ်ခုတည်းကို gradient step **တစ်ခါတည်း** မလုပ်ဘဲ **clipped surrogate objective** ဖြင့် rollout data ကို epoch/minibatch များစွာပြန်သုံးပြီး update လုပ်ခြင်းဖြင့် policy ကို stable စွာ improve စေပါတယ်။

Probability ratio

$$
r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{old}}(a_t|s_t)}
$$

ကို clip လုပ်ပြီး clipped surrogate objective ကို

$$
L^{CLIP}(\theta) = \mathbb{E}_t\left[\min\left(r_t(\theta)A_t,\ \mathrm{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon)A_t\right)\right]
$$

လို့တွက်ပါတယ်။ Advantage $A_t$ ကို GAE ဖြင့်တွက်ပြီး minibatch တစ်ခုစီအတွင်း normalize လုပ်ပါတယ်။

1. **Separate actor/critic networks:** Actor (policy logits) နှင့် critic (state value) ကို orthogonal-initialized network သီးခြားစီသုံးသည် (official CleanRL PPO convention)။
2. **Multiple epochs + minibatches:** Rollout တစ်ခုကို `update_epochs` ကြိမ်ပြန်သုံးပြီး `num_minibatches` အုပ်စုခွဲ shuffle-train လုပ်သည်။
3. **Clipped value loss မလုပ်ပါ (simple MSE):** ရိုးရှင်းအောင် value loss ကို clip မလုပ်ဘဲ `0.5 * MSE(V(s), R)` ကိုသုံးသည်။
4. **Entropy bonus:** Exploration ထိန်းရန် `entropy_coef` ဖြင့် entropy ကို loss ထဲ အနုတ်ပေါင်းထည့်သည်။

Total loss:

$$
L = -L^{CLIP} + c_v L^{VF} - c_e H
$$

## Global Parameters

| Parameter | Default | အဓိပ္ပါယ် |
|---|---:|---|
| `ENV_ID` | `LunarLander-v2` | Box2D discrete-action environment |
| `TOTAL_TIMESTEPS` | `1,000,000` | Training environment steps |
| `NUM_ENVS` | `8` | Parallel environments |
| `NUM_STEPS` | `128` | Update တစ်ကြိမ်မလုပ်ခင် environment တစ်ခုစီ၏ rollout length (`batch_size = NUM_ENVS * NUM_STEPS = 1,024`) |
| `LEARNING_RATE` | `2.5e-4` | Adam learning rate |
| `GAMMA` | `0.99` | Discount factor |
| `GAE_LAMBDA` | `0.95` | GAE bias-variance trade-off |
| `UPDATE_EPOCHS` | `4` | Rollout data ကို ပြန်သုံးမည့် epoch အရေအတွက် |
| `NUM_MINIBATCHES` | `4` | Epoch တစ်ခုစီကို ခွဲမည့် minibatch အရေအတွက် |
| `CLIP_COEF` | `0.2` | Clipped surrogate objective clip range $\epsilon$ |
| `ENTROPY_COEF` | `0.01` | Entropy bonus weight |
| `VALUE_LOSS_COEF` | `0.5` | Value loss weight |
| `MAX_GRAD_NORM` | `0.5` | Gradient clipping norm |
| `HIDDEN_SIZE` | `64` | Actor/critic hidden layer units |
| `SEED` | `1` | Random seed |

## Run Training

```bash
cd docker/12_ppo
python3 train_lunarlander_cleanrl_ppo.py
python3 train_lunarlander_sb3_ppo.py
```

ဥပမာ CleanRL hyperparameter override:

```bash
python3 train_lunarlander_cleanrl_ppo.py --num-steps 128 --clip-coef 0.2 --update-epochs 4
```

CleanRL model ကို `rom_cleanrl_ppo_lunarlander.cleanrl_model`၊ SB3 model ကို `rom_sb3_ppo_lunarlander.zip` အဖြစ်သိမ်းပါတယ်။ TensorBoard logs များကို သက်ဆိုင်ရာ `*_ppo_lunarlander_tensorboard/` folders ထဲမှာရေးပါတယ်။

## Run Evaluation

```bash
cd docker/12_ppo
python3 test_cleanrl_lunarlander.py
python3 test_sb3_lunarlander.py
```

Evaluation တွင် `render_mode="human"` ဖြင့် LunarLander ကိုပြပြီး policy logits ထဲက အမြင့်ဆုံး action (CleanRL) သို့မဟုတ် `deterministic=True` (SB3) ကိုသုံးပါတယ်။
