# NFQ (Neural Fitted Q-Iteration) Original Paper Diagrams

NFQ algorithm ရဲ့ data flow ကို diagram ၂ ခုနဲ့ ဖော်ပြထားပါတယ်။ ပထမတစ်ခုက overview (basic) version ဖြစ်ပြီး ဒုတိယတစ်ခုက Q-Network နဲ့ Target Network ကို ပိုမိုရှင်းလင်းစွာ ခွဲခြားပြထားတဲ့ detailed version ဖြစ်ပါတယ်။

## Diagram 1 — Basic Overview

```mermaid
%%{init: {'theme': 'base', 'themeVariables': { 'primaryColor': '#ffffff'}}}%%
graph TD
    subgraph Phase1 [Data Gathering]
        Env[Environment] -->|State and Reward| Agent[Agent]
        Agent -->|Action| Env
        Agent -->|Collect Transition| Buffer[(Replay Buffer D)]
    end

    subgraph Phase2 [Mini Batch Sampling]
        Buffer -->|Sample Batch| Batch[Transition Sample]
    end

    subgraph Phase3 [Q Network and Target]
        Batch -->|State and Action| MLP[Current Q Network]
        MLP --> Q_pred[Predicted Q Value]

        Batch -->|Next State| Target_Net[Target Q Network]
        Target_Net --> Max_Q[Max Q Value]
        Batch -->|Reward| Target_Y[Target Value Y]
        Max_Q --> Target_Y
    end

    subgraph Phase4 [Optimization]
        Q_pred --> Loss[MSE Loss]
        Target_Y --> Loss
        Loss --> Opt[Gradient Descent Update]
        Opt -->|Update Weights| MLP
    end

    classDef dataGathering fill:#cce5ff,stroke:#004085,stroke-width:1px,color:#004085;
    classDef sampling fill:#fff3cd,stroke:#856404,stroke-width:1px,color:#856404;
    classDef qnetwork fill:#d4edda,stroke:#155724,stroke-width:1px,color:#155724;
    classDef optimization fill:#f8d7da,stroke:#721c24,stroke-width:1px,color:#721c24;

    class Env,Agent,Buffer dataGathering;
    class Batch sampling;
    class MLP,Q_pred,Target_Net,Max_Q,Target_Y qnetwork;
    class Loss,Opt optimization;

    style Phase1 fill:#e7f1ff,stroke:#004085,stroke-width:2px;
    style Phase2 fill:#fffbea,stroke:#856404,stroke-width:2px;
    style Phase3 fill:#eaf7ec,stroke:#155724,stroke-width:2px;
    style Phase4 fill:#fdecea,stroke:#721c24,stroke-width:2px;
```

## Diagram 2 — Detailed Version (Current Q-Network vs. Target Generation)

```mermaid
%%{init: {'theme': 'base', 'themeVariables': { 'primaryColor': '#ffffff'}}}%%
graph TD
    subgraph Phase1 [Data Gathering]
        Env2[Environment] -->|State and Reward| Agent2[Agent]
        Agent2 -->|Action| Env2
        Agent2 -->|Collect Transition| Buffer2[(Replay Buffer D)]
    end

    subgraph Phase2 [Mini Batch Sampling]
        Buffer2 -->|Sample Batch| Batch2[Transition Sample]
    end

    subgraph Phase3 [Q Networks and Target Calculation]
        Batch2 -->|State s and Action a| Q_Current[Current Q-Network<br/>Updates via Gradient Descent]
        Q_Current --> Q_pred2[Predicted Q Value]

        Batch2 -->|Next State s'| Q_Target[Target Generation<br/>Fixed weights from previous iteration]
        Q_Target --> Max_Q2[Max Q Value]
        Batch2 -->|Reward r| Target_Y2[Target Value Y]
        Max_Q2 --> Target_Y2
    end

    subgraph Phase4 [Optimization]
        Q_pred2 --> Loss2[MSE Loss]
        Target_Y2 --> Loss2
        Loss2 --> Opt2[Gradient Descent Update]
        Opt2 -->|Updates Weights| Q_Current
    end

    classDef dataGathering fill:#cce5ff,stroke:#004085,stroke-width:1px,color:#004085;
    classDef sampling fill:#fff3cd,stroke:#856404,stroke-width:1px,color:#856404;
    classDef current fill:#d4edda,stroke:#155724,stroke-width:1px,color:#155724;
    classDef target fill:#e2d9f3,stroke:#4b0082,stroke-width:1px,color:#4b0082;
    classDef optimization fill:#f8d7da,stroke:#721c24,stroke-width:1px,color:#721c24;

    class Env2,Agent2,Buffer2 dataGathering;
    class Batch2 sampling;
    class Q_Current,Q_pred2 current;
    class Q_Target,Max_Q2,Target_Y2 target;
    class Loss2,Opt2 optimization;

    style Phase1 fill:#e7f1ff,stroke:#004085,stroke-width:2px;
    style Phase2 fill:#fffbea,stroke:#856404,stroke-width:2px;
    style Phase3 fill:#eaf7ec,stroke:#155724,stroke-width:2px;
    style Phase4 fill:#fdecea,stroke:#721c24,stroke-width:2px;
```

### Diagram 1 vs Diagram 2 ခြားနားချက်

- Diagram 1 က `MLP` / `Target_Net` ဆိုပြီး node name အတိုချုပ်ပဲ သုံးထားပြီး၊ Diagram 2 ကတော့ `Current Q-Network` (gradient descent နဲ့ update လုပ်တဲ့ network) နဲ့ `Target Generation` (ယခင် iteration ရဲ့ fixed weights သုံးတဲ့ network) ဆိုပြီး ရှင်းလင်းစွာ label တပ်ထားပါတယ်။
- Color coding အားလုံးမှာ **Data Gathering** = ပြာ၊ **Sampling** = အဝါ၊ **Current/Q Network** = အစိမ်း၊ **Target Network** (Diagram 2) = ခရမ်းရောင်၊ **Optimization** = အနီ ဖြစ်ပါတယ်။

## Q&A — Current Q-Network vs Target Q-Network

### မေး ၁။ Network နှစ်ခုလား? တစ်ခုထဲလား?

**Architecture (layer structure) အရ တစ်မျိုးတည်းပဲ ဖြစ်ပေမယ့်, memory ထဲမှာ weight set ၂ ခု သီးခြားစီ ရှိနေရပါတယ်။**

- **Current Q-Network** (weight θ) — gradient descent နဲ့ step တိုင်း update ဖြစ်နေတဲ့ "active" network။
- **Target Q-Network** (weight θ⁻) — Current Network ရဲ့ **weight snapshot (copy) တစ်ခု** ဖြစ်ပြီး၊ target value y တွက်ဖို့အတွက်ပဲ သုံးတာ၊ ဒီ weight တွေကို frozen ထားလို့ gradient မစီးဘူး။

ဒီတော့ "network ၂ ခု" လို့ ပြောလို့ရပေမယ့်, ၎င်းတို့ဟာ **architecture တူတဲ့ clone copy ၂ ခု** ဖြစ်ပါတယ် — model class သီးခြား ၂ မျိုး မဟုတ်ပါဘူး။ (NFQ original paper, Riedmiller 2005 မှာတော့ "outer loop" တစ်ခုချင်းစီအတွက် ယခင် iteration ရဲ့ Q-network ကို frozen snapshot အဖြစ် သုံးပြီး, ထို snapshot ပေါ်မှာ target တွေတွက်ကာ network အသစ်ကို (Rprop နဲ့ batch training) fit လုပ်တဲ့ idea အတူတူပါပဲ — "target network" ဆိုတဲ့ အသုံးအနှုန်းကို နောက်ပိုင်း DQN (Mnih et al., 2015) က ပိုရှင်းအောင် သီးခြား object နှစ်ခုခွဲပြထားတာပါ။)

### မေး ၂။ Target ကို weight freeze လုပ်ပြီး, loop တစ်ခါပြီးမှ current အသစ်ယူကာ loss တွက်တာလား?

**မှန်ပါတယ်, စဉ်းစားပုံအတိအကျ ဖြစ်ပါတယ်။** အဆင့်ဆင့် ဖြစ်စဉ်:

```mermaid
%%{init: {'theme': 'base'}}%%
sequenceDiagram
    participant Cur as Current Q Network theta
    participant Tgt as Target Q Network theta_minus
    participant L as Loss and Optimizer

    Note over Tgt: sync clone theta_minus = theta at start of outer iteration
    rect rgb(226, 217, 243)
    Note right of Tgt: Target side is frozen, no gradient
    L->>Tgt: send next state s_next
    Tgt-->>L: y = reward + gamma times max Q of s_next
    end
    rect rgb(212, 237, 218)
    Note right of Cur: Current side gets gradient
    L->>Cur: send state s and action a
    Cur-->>L: Q_pred
    end
    L->>L: Loss = MSE(Q_pred, y)
    L->>Cur: Backprop, update theta only
    Note over Cur,Tgt: repeat this for many batches or epochs, theta_minus unchanged
    Note over Tgt: next outer iteration starts, theta_minus = theta again
```

- **Target (θ⁻)** ကို ရေတိုမှာ **freeze** ထားပြီး gradient descent **မတွက်ဘူး** — y တွက်ဖို့ forward pass အတွက်ပဲ သုံးတယ် (PyTorch မှာဆိုရင် `.detach()` / `torch.no_grad()` သုံးတာနဲ့ အတူတူ)။
- **Current (θ)** ကိုပဲ gradient descent နဲ့ update လုပ်တယ် — θ⁻ ကို ထိမထားဘူး။
- Loss = (Q(s,a;θ) − y)² ကို θ အတွက်ပဲ backprop လုပ်တယ်။
- ဒီ cycle ကို (NFQ မှာဆို whole pattern set ကို Rprop နဲ့ epoch အများကြီး fit ဖြစ်အောင်, DQN မှာဆို steps အများကြီး) ထပ်ခါထပ်ခါ လုပ်ပြီးမှ — **outer loop/iteration တစ်ခါ ပြီးမှ** θ⁻ ကို θ ရဲ့ value အသစ်နဲ့ **sync/clone** ပြန်လုပ်တယ်။ ဒီနောက်မှ target value တွေ ပြန်တွက်ပြီး iteration အသစ် စလုပ်တာပါ။
- ဒီ "target ကို ခဏ freeze ထားတာ" ကပဲ training ကို stable ဖြစ်စေတဲ့ core idea ဖြစ်ပါတယ် — target ကိုလည်း current အတူတူ step တိုင်း update ရင် "moving target" ပြဿနာ ဖြစ်ပြီး loss ဟာ oscillate/diverge လွယ်သွားမှာပါ။

### မေး ၃။ Network output က ဘာကို ထုတ်ပေးတာလဲ — evaluate လုပ်တာလား, action ရွေးချယ်တာလား?

**Network output ဟာ "evaluation" (value estimate) ပဲ ဖြစ်ပါတယ် — action ကို တိုက်ရိုက် ရွေးချယ်ပေးတာ မဟုတ်ပါဘူး။**

- **Input:** state `s` (NFQ original architecture မှာ state+action `(s,a)` ကို input အတူတကွ ထည့်ပြီး scalar Q value ထုတ်တယ်; DQN-style architecture မှာတော့ state `s` ချည်းပဲ input ပေးပြီး, possible action တစ်ခုစီအတွက် Q value array တစ်ခုလုံးကို တစ်ပြိုင်နက် output ထုတ်ပေးတယ်)။
- **Output:** `Q(s,a)` — "ဒီ state `s` မှာ action `a` ကို ယူပြီး အဲဒီနောက်မှာ (optimal/greedy ဆက်လုပ်မယ်ဆိုရင်) ဘယ်လောက် expected discounted return ရနိုင်မလဲ" ဆိုတဲ့ **value evaluation** ပါ။ ဒါကြောင့် network ကိုယ်တိုင်က "ဘယ် action လုပ်မယ်" ဆိုတာ ဒီဇိုင်းအနေနဲ့ ထုတ်ပေးတာ မဟုတ်ဘဲ, action တစ်ခုချင်းစီရဲ့ "ကောင်းမှု" ကိုပဲ အကဲဖြတ်ပေးတာပါ။
- **Action ရွေးချယ်ခြင်း** ကတော့ ဒီ Q-value evaluation ပေါ်မှာ **နောက်ထပ်တစ်ဆင့်** ထပ်လုပ်တာပါ:
  - Exploitation အချိန်မှာ — `argmax_a Q(s,a)` (Q value အများဆုံး action ကို ယူတယ်)
  - Training/exploration အချိန်မှာ — **ε-greedy** (ε probability နဲ့ random action, (1-ε) နဲ့ argmax action) စတဲ့ policy rule တစ်ခုခုနဲ့ ဆုံးဖြတ်တယ်။
- အတိုချုပ်ပြောရရင်: **Q-network = "critic/evaluator"**, action selection rule (argmax / ε-greedy) = **"policy"** ။ NFQ/DQN မျိုးဟာ value-based method ဖြစ်လို့ policy ကို သီးခြား network နဲ့ မသင်ဘူး — value function ပေါ်မှာ derive ထားတဲ့ implicit policy ကိုပဲ သုံးတာပါ (ဒါကြောင့် actor-critic/policy-gradient method တွေနဲ့ ကွာခြားတာ ဖြစ်ပါတယ်)။
