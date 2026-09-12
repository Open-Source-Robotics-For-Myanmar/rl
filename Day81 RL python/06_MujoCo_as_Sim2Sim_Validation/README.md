# 06 — MuJoCo ကို Sim-to-Sim Validation Tool အဖြစ်သုံးခြင်း

Lou ရဲ့ interview ထဲက pipeline ကို ကိုယ်တိုင် reproduce လုပ်တဲ့ directory ပါ — **Isaac Lab ထဲ train ထားတဲ့ policy ကို MuJoCo ထဲ deploy/validate** လုပ်ခြင်း (real robot မတင်ခင် higher-fidelity simulator ထဲမှာ ပထမဆုံး စမ်းသပ်တဲ့ "sim-to-sim" အဆင့်)။

## ဒီ directory မှာ လုပ်မယ့်အရာများ

1. **Policy export**
   - Isaac Lab (04) ထဲ train ထားတဲ့ checkpoint ကို portable format (`.onnx` သို့ TorchScript) အဖြစ် export လုပ်နည်း လေ့လာ။
2. **Same task ကို MuJoCo XML နဲ့ ပြန်ဆောက်**
   - Isaac Lab task ထဲက robot/task (ဥပမာ Ant, simple quadruped) ကို MuJoCo `.xml` model file အနေနဲ့ ကိုက်ညီအောင် ပြန်တည်ဆောက် (observation/action ordering ကို ၂ ဖက်စလုံး တူညီအောင် သေချာစစ်)။
3. **Inference-only cross-simulator run**
   - Exported policy ကို MuJoCo env ထဲ load ပြီး train **မလုပ်ဘဲ** inference ပဲ run ပြီး behavior/reward ကို Isaac ထဲက result နဲ့ နှိုင်းယှဉ်။
4. **Dynamics gap စစ်ဆေးခြင်း**
   - Reward/success-rate ကွာခြားချက်ရှိရင် ဘယ် parameter (friction, contact model, timestep) ကြောင့်ဖြစ်တာလဲ debug လုပ်ကြည့်။
5. **Latency/observation timing simulation**
   - Lou ပြောခဲ့တဲ့ "vision/sensor latency ကို MuJoCo ထဲမှာ ပိုမှန်ကန်စွာ တွေ့ရတယ်" concept အတိုင်း observation buffer/delay ကို MuJoCo deployment loop ထဲ ထည့်ကြည့်ခြင်း။

## အဓိက concepts

- Cross-simulator policy transfer (observation/action space alignment)
- Contact modeling fidelity difference (Isaac/PhysX GPU vs MuJoCo)
- Sim-to-sim validation ရဲ့ ရည်ရွယ်ချက် — real hardware deploy မလုပ်ခင် extra safety-check layer
- Deployment code testing (inference loop ကို simulation ထဲမှာ အရင်တည့်ခြင်း)

## Deliverables

- [ ] Exported policy file (`.onnx`/TorchScript) + loading script
- [ ] MuJoCo XML model (Isaac task နဲ့ observation/action ကိုက်ညီအောင်ပြင်ထားသည်)
- [ ] Isaac vs MuJoCo reward/behavior comparison report
- [ ] Dynamics gap ရှိပါက root-cause မှတ်စု
