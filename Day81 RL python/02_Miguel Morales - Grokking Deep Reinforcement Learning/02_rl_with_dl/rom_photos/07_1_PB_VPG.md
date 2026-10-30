# VPG (Vanilla Policy Gradient) — REINFORCE with Baseline

[REINFORCE](./06_1_PB_REINFORCE.md) ပေါ်မှာ **learned value baseline `V_φ(s)`** ထည့်ထားတာက VPG ဖြစ်ပါတယ်။ Return `G_t` အစား advantage `A_t = G_t − V(s_t)` ကို policy gradient မှာ သုံးလို့ variance လျော့တယ်။ **Bootstrap မလုပ်သေးလို့ actor-critic မဟုတ်သေးပါ။**

> **အနှစ်ချုပ်:** Network ၂ ခု — Policy (actor) နဲ့ Value (baseline)။ နှစ်ခုလုံး episode ပြီးမှ MC `G_t` နဲ့ သင်တယ်။ Value က `G` ကို predict၊ Policy က `G − V` နဲ့ သင်တယ်။ Entropy bonus က exploration ထိန်းတယ်။

## Original Paper နှင့် ဆက်စပ် Papers

VPG ဆိုတာ paper ခေါင်းစဉ်တစ်ခုတည်း မဟုတ်ပါ။ OpenAI Spinning Up / Grokking က **REINFORCE + baseline** ကို ဒီနာမည်နဲ့ ခေါ်တာပါ။

1. **Williams (1992)** — original REINFORCE; **baseline ကိုယ်တိုင် ဒီ paper မှာ ပါပြီးသား** (constant သို့မဟုတ် state-dependent offset, unbiased)။
2. **Sutton et al. (1999)** — Policy Gradient Theorem; baseline က expectation မပြောင်းဘဲ variance ပဲလျှော့။
3. **Greensmith, Bartlett & Baxter (2004)**, *Variance Reduction Techniques for Gradient Estimates in RL*။
4. **Schulman et al. (2015)**, *GAE* — `G − V` က estimator မိသားစုထဲက တစ်ခု; နောက်ခြေမှာ [GAE](./10_1_PB_GAE.md)။
5. **OpenAI Spinning Up (2018), Vanilla Policy Gradient** — entropy regularization အလေ့အထ။
6. **Morales (2020)**, Grokking Ch. 11 — `FCDAP` + `FCV`, separate optimizers။

## Diagram အရောင်များ

GitHub အတွက် `%%{init}%%` မသုံး။ `flowchart` + `:::class`။

| အရောင် | Fill / stroke | အဓိပ္ပာယ် |
|---|---|---|
| အပြာနု | `#cce5ff` / `#004085` | Env၊ trajectory |
| အစိမ်းနု | `#d4edda` / `#155724` | Policy / Actor |
| အဝါနု | `#fff3cd` / `#856404` | Advantage `G − V` |
| ခရမ်းနု | `#e2d9f3` / `#4b0082` | Value baseline MC only |
| ပန်းရောင်နု | `#f8d7da` / `#721c24` | Loss / optimizer |
| လိမ္မော်နု | `#ffe5cc` / `#8a3b00` | Entropy |

## ဘာပြဿနာကို ဖြေရှင်းတာလဲ

REINFORCE ရဲ့ `G_t` က အပေါင်းကိန်းကြီး ဖြစ်တတ်တယ်။ ကောင်း/ဆိုး ခွဲဖို့ **ပျမ်းမျှထက် ဘယ်လောက်ကွာလဲ** ပဲလိုတယ်။

```text
A_t = G_t - V_φ(S_t)
L_π = - mean( log π(A_t|S_t;θ) · A_t.detach() ) - β_e · H[π]
L_V = mean( (G_t - V_φ(S_t))^2 )
```

`V` ကို detach လုပ်မှ baseline က policy gradient ကို bias မဖြစ်စေပါ။

## Diagram 1 — Paper Lineage

```mermaid
flowchart TD
    P1["Williams 1992 REINFORCE plus optional baseline"]:::policy
    P2["Sutton 1999 PG theorem baseline does not change expectation"]:::shared
    P3["Greensmith 2004 variance reduction"]:::sample
    P4["Schulman 2015 GAE places G minus V on a spectrum"]:::sample
    P5["Spinning Up / Grokking VPG learned V_phi entropy still MC"]:::value
    P6["Next A3C n-step bootstrap plus workers"]:::extra

    P1 --> P5
    P2 --> P5
    P3 --> P5
    P4 -.->|"related estimator family"| P5
    P5 --> P6

    classDef shared fill:#cce5ff,stroke:#004085,color:#004085
    classDef policy fill:#d4edda,stroke:#155724,color:#155724
    classDef sample fill:#fff3cd,stroke:#856404,color:#856404
    classDef value fill:#e2d9f3,stroke:#4b0082,color:#4b0082
    classDef extra fill:#ffe5cc,stroke:#8a3b00,color:#8a3b00
```

## Diagram 2 — Network Architectures

Grokking / docker `05_vpg`: **policy နဲ့ value က network သီးခြား ၂ ခု**။ Optimizer ၂ ခု။ DQN clone ၂ ခုနဲ့ မတူ — output shape ကွာတယ်၊ weight snapshot မဟုတ်။

```mermaid
flowchart TD
    S["State s"]:::shared

    subgraph PolicyNet["Policy network theta FCDAP"]
        S --> PH1["Hidden FC plus ReLU"]:::policy
        PH1 --> PH2["Hidden FC plus ReLU"]:::policy
        PH2 --> Logits["Logits n_actions"]:::policy
        Logits --> Dist["Categorical"]:::sample
        Dist --> Act["Sampled action"]:::sample
        Dist --> LP["log pi"]:::policy
        Dist --> Ent["Entropy H"]:::extra
    end

    subgraph ValueNet["Value network phi FCV baseline"]
        S --> VH1["Hidden FC plus ReLU"]:::value
        VH1 --> VH2["Hidden FC plus ReLU"]:::value
        VH2 --> Vs["V(s) one scalar"]:::value
    end

    classDef shared fill:#cce5ff,stroke:#004085,color:#004085
    classDef policy fill:#d4edda,stroke:#155724,color:#155724
    classDef sample fill:#fff3cd,stroke:#856404,color:#856404
    classDef value fill:#e2d9f3,stroke:#4b0082,color:#4b0082
    classDef extra fill:#ffe5cc,stroke:#8a3b00,color:#8a3b00

    style PolicyNet fill:#eaf7ec,stroke:#155724
    style ValueNet fill:#f3e9fc,stroke:#4b0082
```

| Head | Input | Output | Role |
|---|---|---|---|
| Policy | `s` | logits → `π(a|s)` | sample + `log π` |
| Value | `s` | scalar `V(s)` | baseline only — **next-state bootstrap မလုပ်** |

CartPole: Policy `4 → 120 → 84 → 2`, Value `4 → 120 → 84 → 1`။ Grokking hidden `(32, 32)`။ Dueling `V`/`A` နဲ့ မရော — ဒီ `V` က သီးခြား net၊ Q ပြန်မပေါင်း။

## Diagram 3 — REINFORCE vs VPG

```mermaid
flowchart LR
    subgraph Plain["REINFORCE one net"]
        R1["Episode"]:::train --> R2["G_t"]:::train
        R2 --> R3["L = minus log pi times G_t"]:::train
    end

    subgraph WithBase["VPG two nets"]
        V1["Episode"]:::policy --> V2["G_t"]:::sample
        V1 --> V3["V_phi(s)"]:::value
        V2 --> V4["A = G minus V detach V"]:::sample
        V3 --> V4
        V4 --> V5["L_pi = minus log pi times A minus entropy"]:::policy
        V2 --> V6["L_V = MSE of V and G"]:::value
        V3 --> V6
    end

    classDef policy fill:#d4edda,stroke:#155724,color:#155724
    classDef sample fill:#fff3cd,stroke:#856404,color:#856404
    classDef value fill:#e2d9f3,stroke:#4b0082,color:#4b0082
    classDef train fill:#f8d7da,stroke:#721c24,color:#721c24

    style Plain fill:#fdecea,stroke:#721c24
    style WithBase fill:#eaf7ec,stroke:#155724
```

## Diagram 4 — Training Loop

```mermaid
flowchart TD
    subgraph Phase1["On-policy episode"]
        Env["Environment"]:::shared -->|"s"| Pi["Policy theta"]:::policy
        Env -->|"s"| Vn["Value phi"]:::value
        Pi -->|"a"| Env
        Pi -->|"log_pi and entropy"| Buf["Trajectory"]:::shared
        Vn -->|"V(s)"| Buf
        Env -->|"r, done"| Buf
    end

    subgraph Phase2["MC returns then advantage"]
        Buf --> Gt["Reward-to-go G_t"]:::sample
        Gt --> A["A_t = G_t minus V(s_t) detach"]:::sample
        Buf --> A
    end

    subgraph Phase3["Two losses two optimizers"]
        A --> PL["Policy loss plus entropy bonus"]:::policy
        Gt --> VL["Value MSE vs G"]:::value
        Buf --> PL
        Buf --> VL
        PL --> OptP["Update theta only"]:::train
        VL --> OptV["Update phi only"]:::train
        OptP --> Pi
        OptV --> Vn
    end

    classDef shared fill:#cce5ff,stroke:#004085,color:#004085
    classDef policy fill:#d4edda,stroke:#155724,color:#155724
    classDef sample fill:#fff3cd,stroke:#856404,color:#856404
    classDef value fill:#e2d9f3,stroke:#4b0082,color:#4b0082
    classDef train fill:#f8d7da,stroke:#721c24,color:#721c24

    style Phase1 fill:#e7f1ff,stroke:#004085
    style Phase2 fill:#fffbea,stroke:#856404
    style Phase3 fill:#fdecea,stroke:#721c24
```

## REINFORCE vs VPG

| | REINFORCE | VPG |
|---|---|---|
| Policy net | ရှိ | ရှိ |
| Value net | မရှိ | **ရှိ — baseline** |
| Score | `G_t` | `G_t − V(s_t)` |
| Bootstrap `V(s')` | မရှိ | **မရှိ** |
| Actor-critic? | မဟုတ် | **မဟုတ်** |
| Entropy | မလိုအပ် | Grokking/Spinning Up မှာ ထည့်လေ့ရှိ |
| Target net / replay | မရှိ | မရှိ |

## Q&A

### မေး ၁။ Network နှစ်ခုလား? DQN clone လား?

**သီးခြား architecture ၂ ခု** — logits နဲ့ scalar `V`။ DQN θ / θ⁻ မဟုတ်။ Episode တိုင်း နှစ်ခုလုံး update။

### မေး ၂။ ဘာကြောင့် actor-critic မခေါ်သေးလဲ?

Critic က **bootstrapped** `r + γV(s')` သုံးတယ်။ VPG ရဲ့ `V` က MC `G` ကို regress ရုံ။

### မေး ၃။ `V` gradient က policy ဆီ စီးသလား?

`A = G − V.detach()` ဆိုရင် **မစီးပါ**။

## အနှစ်ချုပ်

```text
VPG        → policy θ + baseline φ
advantage  → G - V(s)   (MC, no bootstrap)
entropy    → keep π from collapsing
not AC yet → wait for A3C n-step V(s_{t+n})
```
