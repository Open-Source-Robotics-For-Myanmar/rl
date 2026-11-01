# A3C — Asynchronous Advantage Actor-Critic: Network၊ Workers နှင့် Algorithm

[VPG ရဲ့ MC baseline](./07_1_PB_VPG.md) က full episode ကို စောင့်ရပြီး data ကလည်း environment တစ်ခုထဲက ဆက်တိုက်လာပါတယ်။ **A3C** က **short n-step bootstrap + parallel environment workers + asynchronous gradient updates** ကို ပေါင်းသုံးပါတယ်။

## ၁။ Original Paper နှင့် Related Sources

| ရင်းမြစ် | ဖတ်ရန် / ဆက်စပ်ပုံ |
|---|---|
| **Mnih et al., ICML 2016 — [Asynchronous Methods for Deep Reinforcement Learning](https://proceedings.mlr.press/v48/mniha16.html)** | မူရင်း paper; **Section 4 ရဲ့ asynchronous advantage actor-critic** အပိုင်း |
| **[arXiv full paper + supplementary](https://arxiv.org/pdf/1602.01783)** | **Algorithm S3**: worker တစ်ခုရဲ့ sync၊ rollout၊ return၊ gradient၊ global update အစဉ်; PDF version အလိုက် algorithm numbering ကွာနိုင်ပါတယ် |
| **Sutton et al., NeurIPS 1999 — [Policy Gradient Methods with Function Approximation](https://papers.nips.cc/paper_files/paper/1999/hash/464d828b85b0bed98e80ade0a5c43b0f-Abstract.html)** | Policy-gradient / value-function သီအိုရီအခြေခံ |
| **[OpenAI Baselines: A2C, 2017](https://openai.com/index/openai-baselines-acktr-a2c/)** | A3C ရဲ့ synchronization ကို ပြောင်းထားတဲ့ နောက်ဆက်တွဲ implementation |

Paper ထဲမှာ async Q-learning / Sarsa နည်းတွေပါရှိပါတယ်။ **ဒီစာက A3C ကိုသာ ရည်ညွှန်းပါတယ်**; အခြား async value-based variants ရဲ့ target network ကို A3C ဆီ မရောထည့်ပါနှင့်။

### အရောင်အသုံးပြုပုံ (Color Rule)

**အစိမ်း၊ လိမ္မော်၊ ခရမ်း ၃ ရောင်ကို network / trainable layer / head block များအတွက်သာ သီးသန့်သုံးပါတယ်။** သက်ဆိုင်ရာ network ပါသော diagram မှာသာ အဲဒီအရောင်ကို တွေ့ရပါမယ်။

| အသုံးပြုပုံ | အရောင် | အရောင်နမူနာ | Fill | Border / text |
|---|---|---|---|---|
| Policy / actor network; Online / current Q-network | အစိမ်းနု | ![အစိမ်းနု နောက်ခံ၊ ဘောင်နှင့်စာသား](./color_swatches/green.svg) | `#d4edda` | `#155724` |
| Value / baseline / critic network နှင့် trainable value head | လိမ္မော်နု | ![လိမ္မော်နု နောက်ခံ၊ ဘောင်နှင့်စာသား](./color_swatches/orange.svg) | `#ffe5cc` | `#8a3b00` |
| Frozen target Q-network | ခရမ်းနု | ![ခရမ်းနု နောက်ခံ၊ ဘောင်နှင့်စာသား](./color_swatches/purple.svg) | `#e2d9f3` | `#4b0082` |
| State / environment / data / shared backbone | အပြာနု | ![အပြာနု နောက်ခံ၊ ဘောင်နှင့်စာသား](./color_swatches/blue.svg) | `#cce5ff` | `#004085` |
| Batch / sampling / advantage | အဝါနု | ![အဝါနု နောက်ခံ၊ ဘောင်နှင့်စာသား](./color_swatches/yellow.svg) | `#fff3cd` | `#856404` |
| Preprocessing / PER / action selection / synchronization | စိမ်းပြာနု | ![စိမ်းပြာနု နောက်ခံ၊ ဘောင်နှင့်စာသား](./color_swatches/cyan.svg) | `#d1ecf1` | `#0c5460` |
| Outputs / probabilities / returns / target values / aggregation | မီးခိုးနု | ![မီးခိုးနု နောက်ခံ၊ ဘောင်နှင့်စာသား](./color_swatches/gray.svg) | `#e2e3e5` | `#383d41` |
| Loss / gradient / optimizer update | ပန်းရောင်နု | ![ပန်းရောင်နု နောက်ခံ၊ ဘောင်နှင့်စာသား](./color_swatches/pink.svg) | `#f8d7da` | `#721c24` |

Arrow = ![အပြာရောင် မြား](./color_swatches/arrow.svg) **အပြာ `#218bff`**။

Network က ထုတ်သော Q / V တန်ဖိုး၊ action probability၊ sample လုပ်သော action၊ return၊ bootstrap target နဲ့ loss block တွေမှာ network အရောင် ၃ ရောင်ကို မသုံးပါဘူး။ **Target value / return က မီးခိုး၊ target network ကသာ ခရမ်း** ဖြစ်ပါတယ်။

Actor နဲ့ critic နှစ်ခုလုံးပါသော shared network / parameter-copy block ကို အပြာဖြင့်ပြပြီး သီးခြား actor / critic heads ကို အစိမ်း / လိမ္မော် ခွဲထားပါတယ်။

## Diagram 1 — Worker တစ်ခု၏ Network Architecture

![Diagram 1 — Worker တစ်ခု၏ Network Architecture](./diagrams/08_1_PB_actor-critic_A3C_01.svg)

<details>
<summary>Mermaid source (ပြင်ဆင်ရန်)</summary>

```text
%%{init: {'theme': 'base', 'themeVariables': { 'primaryColor': '#ffffff', 'lineColor': '#218bff'}}}%%
flowchart TD
    S["Worker i ၏ state s"] --> F["Shared feature extractor<br/>Pixels: CNN; vectors: MLP"]
    F --> M["Hidden features<br/>Recurrent variant: optional LSTM"]
    M --> P["Actor head: K logits<br/>softmax policy pi_local"]
    M --> V["Critic head: linear scalar<br/>V_local of s"]
    P --> A["Sample action a"]
    A --> E["Worker i ၏ own environment"]
    E -->|"next state"| S
    E --> B["Short rollout: rewards and transitions"]
    B --> R["n-step return R_t<br/>bootstrap at rollout endpoint"]
    V --> ADV["A_hat = R_t minus V_local of s"]
    R --> ADV
    ADV -->|"detach"| LP["Actor loss + entropy bonus"]
    P --> LP
    V --> LV["Critic regression loss"]
    R --> LV
    LP --> GR["Local gradients<br/>heads + shared feature extractor"]
    LV --> GR

    classDef data fill:#cce5ff,stroke:#004085,stroke-width:1px,color:#004085;
    classDef batch fill:#fff3cd,stroke:#856404,stroke-width:1px,color:#856404;
    classDef train fill:#f8d7da,stroke:#721c24,stroke-width:1px,color:#721c24;
    classDef process fill:#d1ecf1,stroke:#0c5460,stroke-width:1px,color:#0c5460;
    classDef result fill:#e2e3e5,stroke:#383d41,stroke-width:1px,color:#383d41;
    classDef policyNetwork fill:#d4edda,stroke:#155724,stroke-width:1px,color:#155724;
    classDef valueNetwork fill:#ffe5cc,stroke:#8a3b00,stroke-width:1px,color:#8a3b00;

    class S,F,M,E data;
    class B,ADV batch;
    class LP,LV,GR train;
    class A process;
    class R result;
    class P policyNetwork;
    class V valueNetwork;
```

</details>

**မူရင်း paper ရဲ့ အသုံးများသော architecture** က hidden layers ကို share ပြီး **policy softmax head + linear value head** ခွဲထားတာပါ။ Recurrent variant မှာ LSTM ပါနိုင်ပေမယ့် A3C အတွက် မဖြစ်မနေ မလိုပါဘူး။ **[Chapter 11 ဥပမာ](../chapter_11_summarized.md)** က policy/value MLP နှစ်ခု သီးခြားထားတဲ့ implementation ကို ပြထားပါတယ်; အဲဒါလည်း A3C ဖြစ်နိုင်ပါတယ်။

**Actor head က action probability၊ critic head က scalar `V(s)`** ထုတ်ပါတယ်။ `Â` ကို return နဲ့ `V(s)` ကနေ တွက်တာဖြစ်ပြီး Dueling လို advantage head မရှိပါဘူး။

## Diagram 2 — Global Network နှင့် Asynchronous Workers

![Diagram 2 — Global Network နှင့် Asynchronous Workers](./diagrams/08_1_PB_actor-critic_A3C_02.svg)

<details>
<summary>Mermaid source (ပြင်ဆင်ရန်)</summary>

```text
%%{init: {'theme': 'base', 'themeVariables': { 'primaryColor': '#ffffff', 'lineColor': '#218bff'}}}%%
flowchart TD
    G["Global trainable actor-critic parameters<br/>shared optimizer state"]
    G -->|"pull weights"| W1["Worker 1: local actor-critic copy"]
    G -->|"pull weights"| W2["Worker 2: local actor-critic copy"]
    G -->|"pull weights"| WN["Worker N: local actor-critic copy"]
    W1 <-->|"actions / observations"| E1["Environment 1"]
    W2 <-->|"actions / observations"| E2["Environment 2"]
    WN <-->|"actions / observations"| EN["Environment N"]
    W1 --> B1["Own rollout, up to t_max steps"]
    W2 --> B2["Own rollout, up to t_max steps"]
    WN --> BN["Own rollout, up to t_max steps"]
    B1 --> U1["Compute local gradients"]
    B2 --> U2["Compute local gradients"]
    BN --> UN["Compute local gradients"]
    U1 -->|"apply when ready"| G
    U2 -->|"apply when ready"| G
    UN -->|"apply when ready"| G

    classDef data fill:#cce5ff,stroke:#004085,stroke-width:1px,color:#004085;
    classDef batch fill:#fff3cd,stroke:#856404,stroke-width:1px,color:#856404;
    classDef train fill:#f8d7da,stroke:#721c24,stroke-width:1px,color:#721c24;

    class E1,E2,EN,G,W1,W2,WN data;
    class B1,B2,BN batch;
    class U1,U2,UN train;
```

</details>

**Worker အားလုံးပြီးတဲ့အထိ မစောင့်ပါဘူး။** တစ်ခုအဆင်သင့်ဖြစ်ရင် သူ့ gradient နဲ့ global parameters ကို update လုပ်ပြီး global weights ပြန်ယူပါတယ်။ Local weights ကို global ဆီ တန်း overwrite လုပ်တာ မဟုတ်ပါဘူး။ ပြိုင်တူ workers ကြောင့် data correlation လျော့နိုင်ပေမယ့် data အားလုံး independent ဖြစ်သွားတယ်လို့ မဆိုလိုပါဘူး။

Local copy က rollout အတွင်း acting / gradient တွက်ဖို့ပါ။ **DQN target network အဖြစ် သုံးတာ မဟုတ်ပါဘူး**။ အခြား workers က global ကို ပြင်နေချိန်မှာ local gradient က အနည်းငယ် **stale** ဖြစ်နိုင်ပါတယ်။

## Diagram 3 — Worker Algorithm: Sync → Rollout → Bootstrap → Global Update

![Diagram 3 — Worker Algorithm: Sync → Rollout → Bootstrap → Global Update](./diagrams/08_1_PB_actor-critic_A3C_03.svg)

<details>
<summary>Mermaid source (ပြင်ဆင်ရန်)</summary>

```text
%%{init: {'theme': 'base', 'themeVariables': { 'primaryColor': '#ffffff', 'lineColor': '#218bff'}}}%%
flowchart TD
    SY["Global weights ကို local ဆီ copy"] --> C["Local policy ဖြင့် rollout စု"]
    C --> B["t_max steps သို့မဟုတ် episode boundary"]
    B --> END["Endpoint value<br/>true terminal: 0<br/>otherwise: detached V_local of endpoint"]
    END --> R["Backward recursion<br/>R_t = r_t + gamma times R_next"]
    B --> R
    R --> A["A_hat_t = R_t minus V_local of s_t"]
    A -->|"detach"| LP["Policy loss + entropy bonus"]
    R --> LV["Value loss against detached R_t"]
    LP --> GR["Accumulate local gradients"]
    LV --> GR
    GR --> U["Apply gradients to GLOBAL parameters<br/>အခြား workers ကို မစောင့်"]
    U --> SY

    classDef data fill:#cce5ff,stroke:#004085,stroke-width:1px,color:#004085;
    classDef batch fill:#fff3cd,stroke:#856404,stroke-width:1px,color:#856404;
    classDef train fill:#f8d7da,stroke:#721c24,stroke-width:1px,color:#721c24;
    classDef result fill:#e2e3e5,stroke:#383d41,stroke-width:1px,color:#383d41;

    class C data;
    class B,A batch;
    class LP,LV,GR,U train;
    class SY,END,R result;
```

</details>

`r_t` = action ပြီးနောက် reward၊ `n` = ဒီ state ကနေ rollout endpoint အထိ steps ဖြစ်ပါတယ်။ True terminal ရောက်ရင် endpoint value ကို သုညထားပြီး:

$$
R_t^{(n)}=\sum_{k=0}^{n-1}\gamma^k r_{t+k}+\gamma^n V_{\text{local}}(s_{t+n}),
\qquad \hat A_t=R_t^{(n)}-V_{\text{local}}(s_t).
$$

$$
\mathcal L_\pi=-\mathrm{mean}_t[\log\pi_{\text{local}}(a_t\mid s_t)\mathrm{sg}(\hat A_t)]
-\beta\mathrm{mean}_t H[\pi_{\text{local}}],
\quad
\mathcal L_V=\tfrac12\mathrm{mean}_t\left[(V_{\text{local}}(s_t)-\mathrm{sg}(R_t^{(n)}))^2\right].
$$

ဒီ loss က practical A3C sampled-state convention ဖြစ်ပြီး အပြင် `γ^t` မထည့်ထားပါဘူး; return အတွင်း discount ကတော့ ပါပါတယ်။ Shared trunk ဆိုရင် `L = Lπ + cV LV` ကနေ heads နဲ့ trunk ဆီ gradient စီးပါတယ်။

**Rollout အတွင်း `n` တူမနေပါဘူး** — 5-step chunk ဆိုရင် အစဆုံး state က 5-step၊ နောက်တစ်ခုက 4-step၊ နောက်ဆုံးက 1-step return သုံးပါတယ်။ **Time-limit truncation / chunk cutoff** မှာ task ဆက်နိုင်သေးရင် final observation ရဲ့ value နဲ့ bootstrap လုပ်ပါ; reset ပြီး observation ကို endpoint အဖြစ် မသုံးပါနှင့်။

## ၂။ Q&A

### မေး ၁။ Network ဘယ်နှခုရှိသလဲ?

Shared-trunk design ဆိုရင် **global actor-critic model တစ်ခု + worker တစ်ခုစီရဲ့ local copy** ရှိတယ်။ Each model မှာ actor/value heads နှစ်ခုပါ။ Separate-policy/value design ဆိုရင် global နဲ့ local တစ်နေရာစီမှာ model နှစ်ခုစီ ရှိနိုင်ပါတယ်။ **Head count နဲ့ model-copy count ကို ခွဲမှတ်ပါ။**

### မေး ၂။ Critic က action ကို ရွေးပေးသလား?

မရွေးပါဘူး။ Actor policy ထဲက sample လုပ်ပါတယ်။ Critic `V(s)` က return / advantage တွက်ရာမှာ ကူညီပြီး actor ကို learning signal ပေးပါတယ်။

### မေး ၃။ Advantage head မရှိဘဲ A3C ဘာကြောင့် “Advantage” လို့ခေါ်သလဲ?

Policy gradient ကို **`n-step return − V(s)`** နဲ့ weight ပေးလို့ပါ။ နာမည်က training signal ကို ရည်ညွှန်းတာဖြစ်ပါတယ်။

### မေး ၄။ Replay buffer မသုံးဘဲ ဘာလို့ train လို့ရသလဲ?

Fresh short rollouts ကို worker အများကြီးက စုလို့ data diversity ရပါတယ်။ Standard A3C က persistent replay မသုံးပါဘူး။ Worker တစ်ခုရဲ့ ခဏသိမ်းတဲ့ rollout နဲ့ experience replay ကို မရောပါနှင့်။

### မေး ၅။ A3C က on-policy လား?

အများအားဖြင့် **on-policy actor-critic family** လို့ သတ်မှတ်ပါတယ်။ Data က worker ရဲ့ local behavior policy နဲ့ ကိုက်ပေမယ့် global update လုပ်ချိန်မှာ async policy lag ရှိနိုင်ပါတယ်။ Global latest policy နဲ့ အမြဲအတိအကျတူတယ်လို့ မဆိုလိုပါဘူး။

### မေး ၆။ LSTM၊ shared trunk၊ workers 16 ခု မဖြစ်မနေလိုသလား?

မလိုပါဘူး။ Observation/task နဲ့ compute ပေါ်မူတည်ပါတယ်။ **Asynchronous actor-learners + advantage-based bootstrapped updates** ဆိုတဲ့ mechanism ကို အဓိကမှတ်ပါ။

### မေး ၇။ GAE က A3C ရဲ့ မဖြစ်မနေအပိုင်းလား?

မဟုတ်ပါဘူး။ Original A3C က n-step advantage သုံးပါတယ်။ [GAE](./09_1_PB_GAE.md) နဲ့ အစားထိုးနိုင်ပြီး၊ workers အားလုံးကို စောင့်ပြီး batch update လုပ်တဲ့နည်းကို [A2C](./10_1_PB_actor-critic_A2C.md) မှာ ဆက်ဖတ်နိုင်ပါတယ်။
