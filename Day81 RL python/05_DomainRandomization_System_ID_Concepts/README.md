# 05 — Domain Randomization & System Identification Concepts

Interview ထဲမှာ Lou ရှင်းပြထားတဲ့ sim-to-real pipeline ရဲ့ core technique နှစ်ခုကို ဒီ directory မှာ လက်တွေ့ စမ်းသပ်ပါမယ်: **system identification** (simulation parameters ကို real robot နဲ့ ကိုက်ညီအောင်ချိန်ညှိခြင်း) နဲ့ **domain randomization** (nominal parameters ပတ်ဝန်းကျင်မှာ randomize ပြီး policy ကို robust ဖြစ်အောင် train ခြင်း)။

## ဒီ directory မှာ လုပ်မယ့်အရာများ

1. **Domain randomization API လေ့လာ**
   - Isaac Lab (04) ထဲက domain randomization config (mass, friction coefficient, motor strength/gain, random push force, sensor noise) ကို ဖတ်ပြီး ဘယ် parameters တွေကို randomize လို့ရလဲ list ချ။
2. **Randomization range experiment**
   - Task တစ်ခုကို randomization **မပါ** နဲ့ **ပါ** ၂ မျိုး train ပြီး trained policy ကို perturbation (push/mass change) အောက်မှာ test — robustness ကွာခြားချက်ကို ကိန်းဂဏန်းနဲ့ မှတ်တမ်းတင်။
3. **Randomization range ကို progressively ကျဉ်း/ကျယ်ချိန်ညှိကြည့်ခြင်း**
   - Range ကျဉ်းလွန်း/ကျယ်လွန်းရင် ဖြစ်လာနိုင်တဲ့ training instability ဒါမှမဟုတ် performance ကျဆင်းမှုကို မြင်ကြည့်ခြင်း (ADR — Automatic Domain Randomization idea)။
4. **System identification toy exercise**
   - ရိုးရှင်းတဲ့ simulated system (ဥပမာ pendulum/cartpole) တစ်ခုအတွက် unknown parameter (mass, damping) ကို data ကနေ least-squares/optimization နဲ့ fit လုပ်ကြည့် — Lou ပြောတဲ့ "multi-round calibration + mean/std ယူပြီး randomization range သတ်မှတ်ခြင်း" concept ကို ကိုယ်တိုင် reproduce။
5. **Sensor/actuator noise modeling**
   - IMU/encoder noise, motor torque limit, latency (observation delay buffer) ကို simulation ထဲ inject လုပ်နည်းကို လေ့လာ (Isaac Lab ရဲ့ noise config ဒါမှမဟုတ် custom wrapper ရေးနည်း)။

## အဓိက concepts

- Domain Randomization (DR) vs Automatic Domain Randomization (ADR)
- System Identification (parameter estimation from data)
- Sim-to-real gap sources: mass/inertia error, motor curve mismatch, sensor latency/noise
- Mean/std-based randomization range determination

## Deliverables

- [ ] Domain randomization config file (parameters + ranges list)
- [ ] "With DR" vs "Without DR" robustness comparison result
- [ ] System ID toy exercise script + fitted parameter result
- [ ] Sim-to-real gap sources အတွက် checklist/မှတ်စု
