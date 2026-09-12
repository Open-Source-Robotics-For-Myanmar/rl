# 04 — Isaac Lab / Isaac Sim: GPU-Parallel Robot Learning

Lou ရဲ့ interview ထဲမှာ ပြောထားသလို locomotion research မှာ industry-standard training framework က **Isaac Lab** (Isaac Sim အပေါ်တည်ဆောက်ထားတဲ့) ဖြစ်ပါတယ် — GPU ပေါ်မှာ env ထောင်ချီကို parallel run လုပ်ပြီး PPO training ကို massively scale လုပ်နိုင်တာက အဓိက အားသာချက်။ ဒီ directory ရဲ့ ရည်ရွယ်ချက်က MuJoCo (single-env, CPU) mindset ကနေ Isaac Lab (thousands-of-envs, GPU) mindset ဆီ ကူးပြောင်းဖို့ပါ။

## ဒီ directory မှာ လုပ်မယ့်အရာများ

1. **Prerequisites စစ်**
   - NVIDIA GPU (RTX-class, VRAM 8GB+) ရှိမရှိ၊ driver/CUDA version စစ်။ Isaac Sim ရဲ့ system requirements နဲ့ ကိုက်ညီမှု confirm။
2. **Install**
   - Isaac Lab ကို pip/conda (သို့) Docker container နဲ့ install လုပ် (existing `rl_code/docker` pattern ကို reference ယူပြီး Isaac Lab အတွက် container/venv သီးခြား ထားနိုင်တယ်)။
3. **Built-in tasks run**
   - `Isaac-Cartpole-v0`, `Isaac-Ant-v0`, `Isaac-Humanoid-v0`, `Isaac-Velocity-Anymal-C-v0` စတဲ့ default tasks တွေကို default PPO (rsl_rl / skrl) config နဲ့ run ကြည့်။
4. **Parallel scaling စမ်းသပ်**
   - `num_envs` ကို 16 → 256 → 4096 စသည်ဖြင့် တိုးကြည့်ပြီး training throughput (steps/sec) နဲ့ GPU memory usage ကို မှတ်တမ်းတင်။
5. **Custom reward/task**
   - built-in task config ကို copy ပြီး reward function ကို ကိုယ်တိုင်ပြင်ကြည့် (ဥပမာ velocity tracking reward ပြောင်း)။
6. **Logging**
   - TensorBoard/W&B integration ကို setup ပြီး MuJoCo run တွေနဲ့ training curve ကို side-by-side comparison လုပ်။

## အဓိက concepts

- Vectorized/parallel environments (CPU gym `VecEnv` vs GPU-native PhysX parallel envs)
- USD (Universal Scene Description) asset format — robot ကို `.usd`/`.urdf` ကနေ import
- PhysX GPU simulation pipeline
- `rsl_rl` / `skrl` / `rl_games` — Isaac Lab နဲ့အတူ သုံးလေ့ရှိတဲ့ RL libraries

## Deliverables

- [ ] Isaac Lab install ပြီး built-in task အနည်းဆုံး ၂ ခု training run
- [ ] `num_envs` scaling experiment result (throughput vs GPU memory table)
- [ ] Custom reward function ပြင်ဆင်ထားတဲ့ task တစ်ခု
- [ ] "Isaac Lab vs MuJoCo/gym" training-speed/workflow ကွာခြားချက် အတွက် မှတ်စု
