# 07 — Humanoid Whole-Body Control & Kinematic Retargeting (Advanced)

Roadmap ရဲ့ နောက်ဆုံးအဆင့် — Lou ရဲ့ research area ဖြစ်တဲ့ **kinematic retargeting** (လူ့ motion ကို humanoid robot motion အဖြစ်ပြောင်းခြင်း) နဲ့ **humanoid whole-body control** concepts ကို optional/advanced topic အနေနဲ့ လေ့လာတဲ့ directory ပါ။

## ဒီ directory မှာ လုပ်မယ့်အရာများ

1. **Isaac Lab humanoid tasks လေ့လာ**
   - `Isaac-Humanoid-v0`, Unitree H1/G1 built-in tasks တွေကို run ကြည့်ပြီး humanoid-specific reward terms (upright bonus, energy penalty, foot contact) တွေကို ဖတ်။
2. **Motion imitation / AMP concept**
   - Adversarial Motion Priors (AMP) ဒါမှမဟုတ် reference-motion-tracking reward ဆိုတာ ဘာလဲ၊ Lou ပြောခဲ့တဲ့ "reference motion ကို RL initialization + guidance အဖြစ်သုံးခြင်း" concept နဲ့ ဘယ်လိုဆက်စပ်လဲ လေ့လာ/မှတ်စုချ။
3. **Mocap data sample နဲ့ keypoint matching**
   - Public mocap dataset (ဥပမာ AMASS) ကနေ sample clip တစ်ခုယူပြီး keypoint (joint positions) ကို ဖတ်ထုတ်ကြည့်ခြင်း။
4. **Toy retargeting exercise**
   - Full OmniRetarget optimization pipeline မဟုတ်ဘဲ **simplified 2D/simple-skeleton toy example** တစ်ခုနဲ့ keypoint matching + basic non-penetration constraint concept ကို ကိုယ်တိုင် implement ကြည့်ခြင်း (scale-difference artifact ကိုမြင်ကြည့်ရန်)။
5. **Interaction mesh concept notes**
   - Human–object relative-position graph ("interaction mesh") idea ကို conceptual level မှာ မှတ်စုရေး — full implementation မလိုအပ်ပါ။
6. **Paper/resource reading**
   - OmniRetarget, AMP (Adversarial Motion Priors) ဆိုင်ရာ papers ကို ဖတ်ပြီး key idea summary ချရေး (repo ရဲ့ existing chapter-summary pattern အတိုင်း).

## အဓိက concepts

- Kinematic retargeting (keypoint matching, scale mismatch, penetration artifacts)
- Interaction mesh / graph-based human-object relationship representation
- Adversarial Motion Priors (AMP) — motion-style reward via discriminator
- Reference motion as RL initialization + guidance (hierarchical optimization + RL pipeline)

## Deliverables

- [ ] Isaac Lab humanoid task run result + reward-term notes
- [ ] AMP/motion-imitation concept summary (မှတ်စု)
- [ ] Toy keypoint-retargeting script (simple skeleton, scale mismatch demo)
- [ ] OmniRetarget/AMP paper summary notes
