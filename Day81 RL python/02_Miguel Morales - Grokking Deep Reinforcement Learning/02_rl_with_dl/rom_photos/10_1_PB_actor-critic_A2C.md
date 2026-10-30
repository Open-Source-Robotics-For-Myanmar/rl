# A2C — Synchronous Advantage Actor-Critic

[A3C](./08_1_PB_actor-critic_A3C.md) ရဲ့ **architecture (actor + critic heads)** ကို ထိန်းထားပြီး **async Hogwild! ကို ဖြုတ်**လိုက်တာက A2C ဖြစ်ပါတယ်။ Worker တွေက environment ကို parallel step လုပ်ပြီး rollout ပြည့်မှ **network တစ်ခုတည်း** ကို batch update လုပ်တယ်။ GPU နဲ့ ကိုက်တယ်။

> **အနှစ်ချုပ်:** Shared actor-critic **တစ်ခုတည်း**။ `num_envs` ကို sync vector အနေနဲ့ step။ n-step (သို့ GAE) advantage။ Target network မရှိ။ Loss တစ်ခုထဲမှာ policy + value − entropy ပေါင်းတယ်။

## Original Paper နှင့် ဆက်စပ် Papers

A2C က peer-reviewed “A2C paper” သီးသန့် မဟုတ်ပါ။ A3C ကို engineering ပြန်ချိန်ထားတဲ့ **OpenAI Baselines** နာမည်ဖြစ်ပါတယ်။

1. **Mnih et al. (2016) ICML** — A3C original; A2C က ဒီ architecture/loss ကို sync ပြန်ရေး။
2. **OpenAI Baselines blog (2017)**, *OpenAI Baselines: ACKTR & A2C* (Wu, Espeholt, Andrychowicz, Zaremba, …) — **A2C နာမည် စတင်ကျယ်သုံး**။ Parallel env + one learner က A3C နဲ့ ယှဉ်ပြီး ပိုတည်ငြိမ် / ပိုမြန်နိုင်ကြောင်း။
3. **Schulman et al. (2015)**, *GAE* — ခေတ်သစ် A2C (SB3, CleanRL, Grokking နောက်ပိုင်း) အများအပြားက n-step အစား **GAE(λ)** သုံး။
4. **Schulman et al. (2017)**, *PPO* — A2C-style rollout + clipped surrogate; နောက်အခန်း။
5. **Morales (2020)** Grokking Ch. 11 — `FCAC` single net, `MultiprocessEnv`။
6. **Espeholt et al. (2018) IMPALA** — sync A2C ထက် actor/learner ခွဲ၊ off-policy correction။

## Diagram အရောင်များ

| အရောင် | Fill / border | အဓိပ္ပာယ် |
|---|---|---|
| အပြာနု | `#cce5ff` / `#004085` | Parallel envs၊ shared backbone |
| အစိမ်းနု | `#d4edda` / `#155724` | Policy head |
| အဝါနု | `#fff3cd` / `#856404` | Advantage / GAE |
| ခရမ်းနု | `#e2d9f3` / `#4b0082` | Value head / bootstrap |
| ပန်းရောင်နု | `#f8d7da` / `#721c24` | Combined loss / Adam |
| လိမ္မော်နု | `#ffe5cc` / `#8a3b00` | Sync batch, vector env (A3C worker နေရာ) |

## ဘာပြဿနာကို ဖြေရှင်းတာလဲ

A3C ရဲ့ lock-free update က implementation ခက်၊ GPU အလဟဿ၊ gradient stale ဖြစ်နိုင်။ A2C က:

- Env N ခု **တစ်ပြိုင်နက်** `step`
- Experience ကို **စုပြီးမှ** learner တစ်ခု update
- Decorrelation ကို async နဲ့မဟုတ်ဘဲ **parallel env တွေရဲ့ မတူတဲ့ state** နဲ့ ရ

Advantage နှစ်မျိုးတွေ့ရတယ်:

```text
# Paper-style / classic n-step (A3C formula, batched)
G_t^{(n)} = n-step return with V(s_{t+n}) bootstrap
A_t       = G_t^{(n)} - V(s_t)

# Common modern A2C (GAE)
δ_t = r_t + γ V(s_{t+1})(1-d_t) - V(s_t)
A_t = GAE(γ, λ) from δ
R_t = A_t + V(s_t)          # value target
L   = -mean(log π · A) + c_v MSE(V, R) - c_e mean(H)
```

GAE က A2C ရဲ့ မဖြစ်မနေ အစိတ်အပိုင်း မဟုတ်။ Orthogonal estimator ဖြစ်ပါတယ်။

## Diagram 1 — Paper Lineage

```mermaid
%%{init: {'theme': 'base', 'themeVariables': { 'primaryColor': '#ffffff', 'lineColor': '#218bff'}}}%%
graph TD
    P1[Mnih 2016 A3C - async actor-learners]
    P2[Schulman 2015 GAE - optional advantage estimator]
    P3[OpenAI 2017 A2C - synchronous learners, vectorized envs]
    P4[Grokking FCAC + MultiprocessEnv]
    P5[PPO 2017 - clip on top of A2C-style rollouts]
    P6[IMPALA 2018 - another parallelism design]

    P1 -->|remove Hogwild, keep parallel data| P3
    P2 -.->|often plugged into A2C code| P3
    P3 --> P4
    P3 --> P5
    P1 --> P6

    classDef a3c fill:#e2d9f3,stroke:#4b0082,stroke-width:1px,color:#4b0082;
    classDef gae fill:#fff3cd,stroke:#856404,stroke-width:1px,color:#856404;
    classDef a2c fill:#d4edda,stroke:#155724,stroke-width:1px,color:#155724;
    classDef later fill:#ffe5cc,stroke:#8a3b00,stroke-width:1px,color:#8a3b00;

    class P1 a3c;
    class P2 gae;
    class P3,P4 a2c;
    class P5,P6 later;
```

## Diagram 2 — Network Architecture (အရေးကြီးဆုံး)

A2C architecture က A3C နဲ့ **တူ**တယ်။ ကွာတာက **copy အရေအတွက်** — A2C မှာ live weights တစ်စုံတည်း။ Grokking `FCAC` က Dueling လို hidden မျှပြီး head နှစ်ခု၊ ဒါပေမယ့် output က `Q` မဟုတ် **`(logits, V)` tuple**။

```mermaid
%%{init: {'theme': 'base', 'themeVariables': { 'primaryColor': '#ffffff', 'lineColor': '#218bff'}}}%%
graph TD
    subgraph Phase1 [Single shared actor-critic - FCAC]
        Batch[Batched states from N envs] --> Hidden[Shared FC or Conv + ReLU]
        Hidden --> PHead[Policy head Linear to n_actions]
        Hidden --> VHead[Value head Linear to 1]
        PHead --> Logits[Logits batch]
        Logits --> Dist[Categorical per env]
        Dist --> Act[Actions for all envs at once]
        Dist --> LP[log pi batch]
        Dist --> Ent[Entropy batch]
        VHead --> Vs["V(s) batch"]
    end

    classDef shared fill:#cce5ff,stroke:#004085,stroke-width:1px,color:#004085;
    classDef policy fill:#d4edda,stroke:#155724,stroke-width:1px,color:#155724;
    classDef sample fill:#fff3cd,stroke:#856404,stroke-width:1px,color:#856404;
    classDef value fill:#e2d9f3,stroke:#4b0082,stroke-width:1px,color:#4b0082;

    class Batch,Hidden shared;
    class PHead,Logits,LP policy;
    class Dist,Act,Ent sample;
    class VHead,Vs value;

    style Phase1 fill:#eaf7ec,stroke:#155724,stroke-width:2px;
```

CartPole docker `08_a2c`: `4 → 128 ReLU → {logits: 2, V: 1}`၊ `SyncVectorEnv` default `num_envs=8`။ Grokking: hidden `(32,32)` ပြီး `policy_output` + `value_output`။

```text
forward(s) -> (logits, V)
full_pass  -> action, log_prob, entropy, V
eval       -> argmax(logits)   # value head unused
```

**Target network မရှိ။** Rollout အဆုံး `V(s_T)` ကို **လက်ရှိ** value head နဲ့ တွက်ပြီး `.detach()` လုပ်တယ်။ ဒါက freeze clone မဟုတ်၊ graph ပြန်မဝင်အောင်ပဲ။

## Diagram 3 — A3C vs A2C parallelism

```mermaid
%%{init: {'theme': 'base', 'themeVariables': { 'primaryColor': '#ffffff', 'lineColor': '#218bff'}}}%%
graph LR
    subgraph A3Cside [A3C - many actor-learners]
        L0[Local net 0] -->|async grads| G[Global net]
        L1[Local net 1] -->|async grads| G
        L2[Local net 2] -->|async grads| G
    end

    subgraph A2Cside [A2C - one learner, many envs]
        E0[Env 0]
        E1[Env 1]
        E2[Env 2]
        E0 --> NET[Single actor-critic]
        E1 --> NET
        E2 --> NET
        NET -->|batched sync update| NET
    end

    classDef a3c fill:#ffe5cc,stroke:#8a3b00,stroke-width:1px,color:#8a3b00;
    classDef a2c fill:#d4edda,stroke:#155724,stroke-width:1px,color:#155724;
    classDef env fill:#cce5ff,stroke:#004085,stroke-width:1px,color:#004085;

    class L0,L1,L2,G a3c;
    class NET a2c;
    class E0,E1,E2 env;

    style A3Cside fill:#fff1e5,stroke:#8a3b00,stroke-width:2px;
    style A2Cside fill:#eaf7ec,stroke:#155724,stroke-width:2px;
```

## Diagram 4 — Sync training loop

```mermaid
%%{init: {'theme': 'base', 'themeVariables': { 'primaryColor': '#ffffff', 'lineColor': '#218bff'}}}%%
graph TD
    subgraph Phase1 [Vectorized rollout]
        Net[Single AC net] -->|sample a for each env| Envs[N environments]
        Envs -->|r, s, done for all| Buf[(Rollout tensor: T times N)]
        Buf --> Net
    end

    subgraph Phase2 [Advantage]
        Buf --> Boot["Detach V(s_T) bootstrap"]
        Boot --> Adv[n-step or GAE advantages]
        Adv --> Ret["returns = A + V  value targets"]
    end

    subgraph Phase3 [One combined optimizer step]
        Adv --> PL[Policy loss]
        Ret --> VL[Value loss]
        Buf --> Ent[Entropy bonus]
        PL --> L["L = L_pi + c_v L_V - c_e H"]
        VL --> L
        Ent --> L
        L --> Opt[Adam + grad clip]
        Opt --> Net
    end

    classDef dataGathering fill:#cce5ff,stroke:#004085,stroke-width:1px,color:#004085;
    classDef sampling fill:#fff3cd,stroke:#856404,stroke-width:1px,color:#856404;
    classDef value fill:#e2d9f3,stroke:#4b0082,stroke-width:1px,color:#4b0082;
    classDef optimization fill:#f8d7da,stroke:#721c24,stroke-width:1px,color:#721c24;
    classDef policy fill:#d4edda,stroke:#155724,stroke-width:1px,color:#155724;

    class Envs,Buf dataGathering;
    class Adv sampling;
    class Boot,Ret,VL value;
    class L,Opt,Ent optimization;
    class Net,PL policy;

    style Phase1 fill:#e7f1ff,stroke:#004085,stroke-width:2px;
    style Phase2 fill:#fffbea,stroke:#856404,stroke-width:2px;
    style Phase3 fill:#fdecea,stroke:#721c24,stroke-width:2px;
```

## Architecture နှိုင်းယှဉ် — REINFORCE → A2C

| | REINFORCE | VPG | A3C | A2C |
|---|---|---|---|---|
| Policy head | ရှိ | ရှိ | ရှိ | ရှိ |
| Value head | မရှိ | သီးခြား net | head သို့မဟုတ် သီးခြား | **shared net head** |
| Bootstrap | မရှိ | မရှိ | n-step `V` | n-step သို့မဟုတ် GAE |
| Actor-critic | မဟုတ် | မဟုတ် | ဟုတ် | **ဟုတ်** |
| Copies in RAM | 1 | 2 nets | 1 global + N local | **1** |
| Parallel | မရှိ | မရှိ | async learners | **sync envs** |
| Target Q | မရှိ | မရှိ | မရှိ | **မရှိ** |
| Typical loss | `-logπ G` | `-logπ A`, MSE `V` | same, Hogwild | **weighted sum, one step** |

## Q&A

### မေး ၁။ Network ဘယ်နှခုလဲ?

**တစ်ခု။** Shared backbone + policy head + value head။ A3C worker copy တွေ မလို။ DQN target မလို။

### မေး ၂။ A3C နဲ့ architecture ကွာသလား?

**မကွာသင့်ပါ** — နှစ်ခုလုံး actor-critic heads။ ကွာတာ parallelism နဲ့ update timing။ Code တချို့က A3C မှာ nets ခွဲ၊ A2C မှာ share — ဒါက implementation choice၊ algorithm definition မဟုတ်။

### မေး ၃။ GAE မထည့်ရင် A2C မဟုတ်ဘူးလား?

**ဟုတ်ပါတယ်။** A2C = synchronous advantage actor-critic။ GAE က advantage တွက်ပုံ option။

### မေး ၄။ Output က action တိုက်ရိုက်လား?

Logits → distribution။ Train: sample။ Eval: `argmax`။ `V(s)` က critic estimate၊ acting မပါ။

## အနှစ်ချုပ်

```text
A2C net     → shared backbone + π head + V head
data        → N envs, sync rollout T steps
advantage   → n-step or GAE (optional)
update      → one Adam step on combined loss
vs A3C      → no local clones, no Hogwild!
vs DQN      → no θ⁻, no replay; on-policy actor-critic
next typical → PPO clip / entropy / GAE on this skeleton
```
