# VPG — REINFORCE with Value Baseline၊ Network နှင့် Algorithm Diagrams

[REINFORCE](./06_1_PB_REINFORCE.md) မှာ return `G_t` ကို policy score အဖြစ် တိုက်ရိုက်သုံးခဲ့ပါတယ်။ ဒီစာရဲ့ **Vanilla Policy Gradient (VPG)** က **“ဒီ state မှာ မျှော်လင့်ထားတာထက် ကောင်းသလား?”** ဆိုတဲ့ `Â_t = G_t − Vφ(s_t)` ကို သုံးပြီး gradient variance လျှော့ဖို့ ရည်ရွယ်ပါတယ်။

## ၁။ Original Foundations နှင့် Related Papers

**VPG အတွက် မူရင်း paper တစ်စောင်တည်း သီးခြားသတ်မှတ်ထားတာ မရှိပါဘူး။** Plain policy-gradient methods ကို ခေါ်တဲ့ နာမည်ဖြစ်ပြီး ဒီ repo ရဲ့ Chapter 11 အတိုင်း **MC return + learned baseline** version ကို အဓိကပြထားပါတယ်။

| ရင်းမြစ် | ဆက်စပ်ပုံ |
|---|---|
| **Williams, 1992 — [REINFORCE](https://link.springer.com/article/10.1007/BF00992696)** | Score-function policy-gradient နဲ့ baseline အခြေခံ |
| **Sutton et al., NeurIPS 1999 — [Policy Gradient Methods with Function Approximation](https://papers.nips.cc/paper_files/paper/1999/hash/464d828b85b0bed98e80ade0a5c43b0f-Abstract.html)** | Policy-gradient theorem၊ value-function approximation ဆက်စပ်သီအိုရီ |
| **[Spinning Up: Vanilla Policy Gradient](https://spinningup.openai.com/en/latest/algorithms/vpg.html)** | On-policy VPG ရဲ့ official algorithm reference; implementation မှာ GAE လည်း သုံးထားတယ် |
| **Schulman et al., 2015 — [TRPO](https://proceedings.mlr.press/v37/schulman15.html)** | Policy step အလွန်ကြီးသွားတာကို trust region နဲ့ ထိန်းတဲ့ ဆက်စပ်နည်း |
| **[Chapter 11 မှတ်စု](../chapter_11_summarized.md)** | ဒီမှာသုံးတဲ့ separate `FCDAP` / `FCV` networks နဲ့ MC baseline version |

### အရောင်အသုံးပြုပုံ (Color Rule)

[အရောင် ၈ မျိုးနှင့် network label သတ်မှတ်ချက် — 00_1_color_define.md](./00_1_color_define.md)

| အသုံးပြုပုံ | အရောင် | နမူနာ | Fill | Text |
|---|---|---|---|---|
| Policy Network / Actor; Online / Current Q Network | **Green / အစိမ်း** | ![Green / အစိမ်း](./color_swatches/green.svg) | `#008000` | `#FFFFFF` |
| Value Network / Critic / Baseline; Dueling value head | **Orange / လိမ္မော်** | ![Orange / လိမ္မော်](./color_swatches/orange.svg) | `#FF8C00` | `#111111` |
| Frozen Target Network | **Purple / ခရမ်း** | ![Purple / ခရမ်း](./color_swatches/purple.svg) | `#800080` | `#FFFFFF` |
| State / environment / data / shared backbone | **Blue / အပြာ** | ![Blue / အပြာ](./color_swatches/blue.svg) | `#0057D9` | `#FFFFFF` |
| Batch / sampling / advantage | **Yellow / အဝါ** | ![Yellow / အဝါ](./color_swatches/yellow.svg) | `#FFD700` | `#111111` |
| Preprocessing / PER / action selection / synchronization | **Cyan / စိမ်းပြာ** | ![Cyan / စိမ်းပြာ](./color_swatches/cyan.svg) | `#00CED1` | `#111111` |
| Loss / gradient / optimizer | **Red / အနီ** | ![Red / အနီ](./color_swatches/red.svg) | `#DC143C` | `#FFFFFF` |
| Outputs / probabilities / returns / target values / aggregation | **Gray / မီးခိုး** | ![Gray / မီးခိုး](./color_swatches/gray.svg) | `#555555` | `#FFFFFF` |

**Green / orange / purple ကို network၊ trainable layer / head တွေအတွက်သာ သီးသန့်သုံးပါတယ်။** Network block ထဲမှာ `Policy Network`၊ `Value Network`၊ `Online Network` သို့မဟုတ် `Target Network` ဆိုပြီး အမျိုးအစားပါပါတယ်။ Neuron များကို စုပြထားသောပုံမှာ အမျိုးအစားကို enclosing network / layer ခေါင်းစဉ်မှာ ရေးထားပါတယ်။

Shared actor-critic block မှာ **Policy Network + Value Network** နှစ်ခုလုံးကို ဖော်ပြပြီး shared backbone / copy ကို blue သုံးထားပါတယ်။ Shared heads ဖြစ်ရင် သီးခြား model နှစ်ခုရှိတယ်လို့ မဆိုလိုပါဘူး။ Dueling ရဲ့ value / advantage heads က Q Network ထဲက အစိတ်အပိုင်းများဖြစ်ပြီး သီးခြား policy network မဟုတ်ပါဘူး။

**Target value / return = gray၊ Target Network = purple** ဖြစ်ပါတယ်။ Network outputs၊ action sampling၊ loss နဲ့ paper-lineage block များကို network အရောင် ၃ ရောင်နဲ့ မပြပါဘူး။ Group နောက်ခံကို transparent ထားပြီး arrow ကို blue သုံးထားပါတယ်။

## Diagram 1 — Network Architecture: Policy နှင့် Value Baseline သီးခြားနှစ်ခု

![Diagram 1 — Network Architecture: Policy နှင့် Value Baseline သီးခြားနှစ်ခု](./diagrams/07_1_PB_VPG_01.svg)

<details>
<summary>Mermaid source (ပြင်ဆင်ရန်)</summary>

```text
%%{init: {"theme": "base", "themeVariables": {"primaryColor": "#555555", "primaryTextColor": "#FFFFFF", "primaryBorderColor": "#111111", "secondaryColor": "#555555", "tertiaryColor": "#FFFFFF", "lineColor": "#0057D9", "clusterBkg": "#FFFFFF", "clusterBorder": "#555555", "edgeLabelBackground": "#FFFFFF", "textColor": "#111111"}, "htmlLabels": false, "flowchart": {"htmlLabels": false}, "themeCSS": ".edgeLabel text, .edgeLabel tspan { fill: #111111 !important; } .edgeLabel { color: #111111 !important; } .cluster-label text, .cluster-label tspan { fill: #111111 !important; }"}}%%
flowchart TD
    S["State s"] --> P["Policy Network (FCDAP)<br/>theta: FC 32 - ReLU - FC 32 - ReLU"]
    P --> Z["Policy Network Output Head<br/>Linear: K action logits"]
    Z --> D["Categorical policy pi_theta"]
    D --> A["Sample action a"]
    D --> LP["log pi_theta of a given s<br/>and policy entropy H"]
    S --> V["Value Network (FCV Baseline)<br/>phi: FC 32 - ReLU - FC 32 - ReLU"]
    V --> VS["Value Network Output Head<br/>Linear: scalar V_phi(s)"]
    A --> E["Environment"]
    E --> R["Complete episode rewards"]
    R --> G["MC return G_t"]
    G --> ADV["Advantage estimate<br/>A_hat = G_t minus V_phi of s"]
    VS --> ADV
    ADV -->|"detach"| L["Policy loss + entropy bonus"]
    LP --> L
    G --> LV["Value regression loss"]
    VS --> LV
    L -.->|"update theta"| P
    LV -.->|"update phi"| V
    classDef data fill:#0057D9,stroke:#111111,stroke-width:2px,color:#FFFFFF;
    classDef batch fill:#FFD700,stroke:#111111,stroke-width:2px,color:#111111;
    classDef train fill:#DC143C,stroke:#111111,stroke-width:2px,color:#FFFFFF;
    classDef process fill:#00CED1,stroke:#111111,stroke-width:2px,color:#111111;
    classDef result fill:#555555,stroke:#111111,stroke-width:2px,color:#FFFFFF;
    classDef policyNetwork fill:#008000,stroke:#111111,stroke-width:2px,color:#FFFFFF;
    classDef valueNetwork fill:#FF8C00,stroke:#111111,stroke-width:2px,color:#111111;

    class S,E,R data;
    class ADV batch;
    class L,LV train;
    class A process;
    class D,LP,G result;
    class P,Z policyNetwork;
    class V,VS valueNetwork;
```

</details>

ဒီ **Chapter 11 ဥပမာ**မှာ trainable networks နှစ်ခု၊ weight sets `θ` နဲ့ `φ` နှစ်ခုပါ။ Architecture တူဖို့မလိုဘဲ policy output က **action logits**၊ value output က **scalar expected return** ဖြစ်ပါတယ်။ `Vφ` က frozen target-network clone မဟုတ်ပါဘူး; regression loss နဲ့ သင်နေပါတယ်။ Shared trunk + heads နှစ်ခုလည်း ရွေးနိုင်ပေမယ့် VPG ရဲ့ မဖြစ်မနေ requirement မဟုတ်ပါဘူး။

## Diagram 2 — MC Baseline VPG Algorithm

![Diagram 2 — MC Baseline VPG Algorithm](./diagrams/07_1_PB_VPG_02.svg)

<details>
<summary>Mermaid source (ပြင်ဆင်ရန်)</summary>

```text
%%{init: {"theme": "base", "themeVariables": {"primaryColor": "#555555", "primaryTextColor": "#FFFFFF", "primaryBorderColor": "#111111", "secondaryColor": "#555555", "tertiaryColor": "#FFFFFF", "lineColor": "#0057D9", "clusterBkg": "#FFFFFF", "clusterBorder": "#555555", "edgeLabelBackground": "#FFFFFF", "textColor": "#111111"}, "htmlLabels": false, "flowchart": {"htmlLabels": false}, "themeCSS": ".edgeLabel text, .edgeLabel tspan { fill: #111111 !important; } .edgeLabel { color: #111111 !important; } .cluster-label text, .cluster-label tspan { fill: #111111 !important; }"}}%%
flowchart TD
    P["Policy Network<br/>Current weights theta"] --> C["Collect fresh complete episodes"]
    C --> B["On-policy batch of states, actions, rewards"]
    B --> G["Compute reward-to-go G_t"]
    B --> V["Value Network (Baseline)<br/>Current weights phi"]
    V --> VS["Baseline predictions V_phi"]
    G --> A["A_hat = G_t minus V_phi<br/>compute before fitting baseline"]
    VS --> A
    A -->|"stop gradient"| LP["Policy loss: negative log probability<br/>weighted by advantage"]
    P --> H["Entropy bonus H"]
    H --> LP
    LP --> OP["One policy optimizer step"]
    OP -->|"theta updated"| P
    VS --> LV["Value loss: fit V_phi to G_t"]
    G --> LV
    LV --> OV["Value optimizer steps"]
    OV -->|"phi updated"| V
    OP --> N["Discard rollout and collect new data"]
    N --> C
    classDef data fill:#0057D9,stroke:#111111,stroke-width:2px,color:#FFFFFF;
    classDef batch fill:#FFD700,stroke:#111111,stroke-width:2px,color:#111111;
    classDef train fill:#DC143C,stroke:#111111,stroke-width:2px,color:#FFFFFF;
    classDef result fill:#555555,stroke:#111111,stroke-width:2px,color:#FFFFFF;
    classDef policyNetwork fill:#008000,stroke:#111111,stroke-width:2px,color:#FFFFFF;
    classDef valueNetwork fill:#FF8C00,stroke:#111111,stroke-width:2px,color:#111111;

    class C,N data;
    class B,A batch;
    class LP,OP,LV,OV train;
    class VS,H,G result;
    class P policyNetwork;
    class V valueNetwork;
```

</details>

`r_t` = action `a_t` ပြီးနောက် reward၊ `sg` = stop-gradient၊ `β ≥ 0` = entropy coefficient လို့ သတ်မှတ်ပါ။ ဒီ MC version မှာ:

$$
G_t=\sum_{k=t}^{T-1}\gamma^{k-t}r_k,
\qquad \hat A_t=G_t-V_\phi(s_t).
$$

$$
\mathcal L_\pi=-\mathrm{mean}_t\left[\gamma^t\mathrm{sg}(\hat A_t)\log\pi_\theta(a_t\mid s_t)\right]
-\beta\mathrm{mean}_t H[\pi_\theta(\cdot\mid s_t)],
\qquad
\mathcal L_V=\tfrac12\mathrm{mean}_t\left(V_\phi(s_t)-\mathrm{sg}(G_t)\right)^2.
$$

**ဒီမှာ အပြင် `γ^t` က စာအုပ် convention** ဖြစ်ပါတယ်။ [REINFORCE မှတ်စု](./06_1_PB_REINFORCE.md) မှာ ရှင်းထားသလို practical sampled-state VPG surrogate တချို့မှာ မပါပါဘူး။ Entropy bonus က optional ဖြစ်ပြီး **VPG ကို အဓိပ္ပာယ်သတ်မှတ်တဲ့ မဖြစ်မနေအပိုင်း မဟုတ်ပါဘူး**။ Value regression ကို batch တစ်ခုတည်းနဲ့ steps အများကြီးလုပ်နိုင်ပေမယ့် vanilla policy loss ကို policy ပြောင်းပြီးတိုင်း correction မပါဘဲ ထပ်ခါ optimize မလုပ်သင့်ပါဘူး။

### Baseline က ဘာကြောင့် အထောက်အကူဖြစ်သလဲ?

`V(s)=8` ဖြစ်တဲ့ state တစ်ခုမှာ `G=10` ဆိုရင် `Â=+2`၊ `G=5` ဆိုရင် `Â=−3` ပါ။ နှစ်ခုလုံး positive reward ရပေမယ့် ဒုတိယတစ်ခုက **မျှော်လင့်ချက်ထက် နိမ့်**လို့ ရွေးခဲ့တဲ့ action probability ကို လျှော့ဖို့ signal ပေးပါတယ်။

Action မပေါ်မူတည်တဲ့ baseline `b(s)` ကို fixed လို့ယူပြီး policy gradient တွက်ရင်:

$$
\mathbb E_{a\sim\pi_\theta}\!\left[b(s)\nabla_\theta\log\pi_\theta(a\mid s)\right]
=b(s)\nabla_\theta\sum_a\pi_\theta(a\mid s)=0.
$$

ဒါကြောင့် baseline subtraction က expected policy gradient ကို မပြောင်းဘဲ variance လျှော့နိုင်ပါတယ်။ **Baseline မကောင်းရင် variance အမြဲလျော့မယ်လို့ အာမမခံပါဘူး**; same-batch fitting/normalization က finite-sample effects လည်း ရှိနိုင်ပါတယ်။

## Diagram 3 — VPG၊ GAE၊ TRPO / PPO တို့၏ တာဝန်ကွာခြားပုံ

![Diagram 3 — VPG၊ GAE၊ TRPO / PPO တို့၏ တာဝန်ကွာခြားပုံ](./diagrams/07_1_PB_VPG_03.svg)

<details>
<summary>Mermaid source (ပြင်ဆင်ရန်)</summary>

```text
%%{init: {"theme": "base", "themeVariables": {"primaryColor": "#555555", "primaryTextColor": "#FFFFFF", "primaryBorderColor": "#111111", "secondaryColor": "#555555", "tertiaryColor": "#FFFFFF", "lineColor": "#0057D9", "clusterBkg": "#FFFFFF", "clusterBorder": "#555555", "edgeLabelBackground": "#FFFFFF", "textColor": "#111111"}, "htmlLabels": false, "flowchart": {"htmlLabels": false}, "themeCSS": ".edgeLabel text, .edgeLabel tspan { fill: #111111 !important; } .edgeLabel { color: #111111 !important; } .cluster-label text, .cluster-label tspan { fill: #111111 !important; }"}}%%
flowchart LR
    R["REINFORCE<br/>MC score G_t"] -->|"learned baseline"| V["ဒီစာ၏ VPG<br/>score G_t minus V"]
    V -.->|"advantage estimator ကို ပြောင်းနိုင်"| G["GAE, 2015 / ICLR 2016<br/>weighted TD residuals"]
    V -.->|"policy update rule ဆက်စပ်နည်း"| T["TRPO, 2015<br/>trust-region constraint"]
    T --> P["PPO, 2017<br/>clipped surrogate variant"]
    G -.->|"advantage estimates ပေးနိုင်"| P
    classDef batch fill:#FFD700,stroke:#111111,stroke-width:2px,color:#111111;
    classDef train fill:#DC143C,stroke:#111111,stroke-width:2px,color:#FFFFFF;
    classDef result fill:#555555,stroke:#111111,stroke-width:2px,color:#FFFFFF;

    class G batch;
    class T,P train;
    class R,V result;
```

</details>

**GAE က score တွက်ပုံ၊ TRPO/PPO က policy update ထိန်းပုံ** ဖြစ်လို့ အတူတွဲနိုင်ပါတယ်။ [PPO မူရင်း paper](https://arxiv.org/abs/1707.06347) နဲ့ [GAE မှတ်စု](./09_1_PB_GAE.md) ကို ဆက်ဖတ်နိုင်ပါတယ်။

## ၂။ Q&A

### မေး ၁။ REINFORCE နဲ့ အဓိကဘာကွာသလဲ?

ဒီအစဉ်မှာ `G_t` အစား **`G_t−V(s_t)`** ကို policy score သုံးပြီး value baseline ကိုပါ သင်တာက အဓိကကွာခြားချက်ပါ။

### မေး ၂။ Policy နဲ့ value က online / target clone နှစ်ခုလား?

မဟုတ်ပါဘူး။ **Policy က “ဘာလုပ်မလဲ”၊ value က “ဒီ state မှာ ပျမ်းမျှဘယ်လောက်ရမလဲ”** ကို သင်တယ်။ နှစ်ခုလုံး trainable ဖြစ်ပြီး output meaning မတူပါဘူး။

### မေး ၃။ Actor-critic လို့ ခေါ်လို့ရသလား?

အခေါ်အဝေါ်ပေါ် မူတည်ပါတယ်။ စာအုပ်က **MC baseline VPG** နဲ့ **bootstrapped actor-critic** ကို ခွဲခေါ်ထားပါတယ်။ တချို့ရင်းမြစ်တွေက learned policy/value ပါရင် actor-critic လို့ ကျယ်ကျယ်ပြန့်ပြန့် ခေါ်ပါတယ်။ အဓိကက **ဒီ version ရဲ့ `G_t` ထဲမှာ next-state value bootstrap မပါ**တာပါ။

### မေး ၄။ Advantage ကို ဘာကြောင့် detach လုပ်ရသလဲ?

Policy update အတွက် advantage ကို **fixed learning signal** အဖြစ် သုံးချင်လို့ပါ။ Detach မလုပ်ရင် `-log π × (G−V)` ကနေ baseline ဆီ မရည်ရွယ်တဲ့ gradient စီးသွားနိုင်ပါတယ်။ Value ကို သူ့ regression loss နဲ့ သီးခြားသင်ပါတယ်။

### မေး ၅။ `Â` ဆိုတာ Dueling ရဲ့ advantage head လား?

မဟုတ်ပါဘူး။ ဒီမှာ **return နဲ့ baseline ကနေတွက်တဲ့ scalar estimate** ပါ။ Action တိုင်းအတွက် `A(s,a)` vector ထုတ်တဲ့ network head မရှိပါဘူး။

### မေး ၆။ VPG က GAE သုံးလို့ရသလား?

ရပါတယ်။ VPG ဆိုတာ plain policy-gradient update family ဖြစ်လို့ advantage estimator ကို ပြောင်းနိုင်ပါတယ်။ Spinning Up implementation ကလည်း GAE သုံးပါတယ်။ **ဒီစာရဲ့ MC version နဲ့ အဲဒီ implementation ကို ခွဲဖတ်ပါ။**

### မေး ၇။ Entropy မြင့်ရင် အမြဲကောင်းသလား?

မဟုတ်ပါဘူး။ Bonus က policy စောစောမတင်းကျပ်အောင် ကူညီပေမယ့် coefficient ကြီးလွန်းရင် reward ကောင်းတဲ့ action ပေါ် အာရုံမစိုက်နိုင်ပါဘူး။

### မေး ၈။ Policy network ကို gradient descent လုပ်ဖို့ value network က ထုတ်တဲ့ကောင်ကို နှုတ်တာမဟုတ်လား?

**ဟုတ်ပါတယ် — episode က ရလာတဲ့ return `G_t` ထဲက value network ရဲ့ ခန့်မှန်းတန်ဖိုး `Vφ(s_t)` ကို နှုတ်တာပါ။** အဲဒီရလဒ်ကို advantage estimate လို့ ခေါ်ပါတယ်။

$$
\hat A_t = G_t - V_\phi(s_t).
$$

Policy network ရဲ့ output သို့မဟုတ် weights ထဲက value network ကို နှုတ်တာ မဟုတ်ပါဘူး။ **“တကယ်ရလာတဲ့ return − ဒီ state မှာ မျှော်လင့်ထားတဲ့ return”** ကို တွက်ပြီး ရွေးခဲ့တဲ့ action ရဲ့ log-probability ကို weight ပေးတာပါ။ Entropy bonus ကို ခဏဖယ်ကြည့်ရင် (`β = 0`):

$$
\mathcal L_\pi
=-\frac{1}{T}\sum_{t=0}^{T-1}\gamma^t\,\mathrm{sg}(\hat A_t)
\log\pi_\theta(a_t\mid s_t),
\qquad
\theta \leftarrow \theta - \alpha\nabla_\theta\mathcal L_\pi.
$$

ဥပမာ `Vφ(s_t) = 8` ဆိုရင် `G_t = 10` အတွက် `Â_t = +2` က ရွေးခဲ့တဲ့ action probability ကို တိုးဖို့ အားပေးပါတယ်။ `G_t = 5` အတွက် `Â_t = −3` က လျှော့ဖို့ signal ပေးပါတယ်။ **Gradient ကို policy ရဲ့ `log πθ` ကနေ တွက်ပြီး detached advantage ကို အလေးချိန်အဖြစ် သုံးတာပါ။**

### မေး ၉။ ဒါဆို freeze လုပ်ထားတဲ့ policy network၊ value network တွေ မရှိဘူးမဟုတ်လား?

**ဟုတ်ပါတယ်။ ဒီစာမှာပြထားတဲ့ MC baseline VPG မှာ frozen policy clone / frozen value target clone မရှိပါဘူး။** Policy `θ` နဲ့ value `φ` နှစ်ခုလုံး trainable ဖြစ်ပြီး ကိုယ့် loss နဲ့ကိုယ် update လုပ်ပါတယ်။

| Trainable network | သုံးတဲ့ loss | ဒီစာရဲ့ separate-network ပုံစံမှာ update ဖြစ်တဲ့ weights |
|---|---|---|
| Policy `πθ(a\|s)` | Detached advantage နဲ့ weight ပေးထားတဲ့ policy loss `Lπ` | `θ` |
| Value baseline `Vφ(s)` | MC return `G_t` ကို ခန့်မှန်းနိုင်အောင် သင်တဲ့ regression loss `LV` | `φ` |

ဒီမှာ **network freeze လုပ်တာ** နဲ့ **advantage ကို detach လုပ်တာ** ကို ခွဲမြင်ရပါမယ်။ `sg(Â_t)` / `detach()` က policy loss တွက်တဲ့အခါ advantage ကို ကိန်းသေအဖြစ်ယူပြီး အဲဒီလမ်းကြောင်းကနေ value network ဆီ gradient မစီးအောင် လုပ်တာပါ။ Value network ကို ဆက်မသင်တော့အောင် freeze လုပ်ထားတာ မဟုတ်ပါဘူး — သူ့ `LV` နဲ့ ဆက်သင်နေပါတယ်။

Episode / batch စုနေစဉ်မှာတော့ policy တစ်ခုတည်းက data ရဖို့ **policy weights ကို မပြောင်းဘဲ ထားပါတယ်**။ Data စုပြီးရင် baseline မသင်မီ လက်ရှိ `Vφ` နဲ့ advantage တွက်၊ detach လုပ်ပြီး policy update နဲ့ value update ကို လုပ်ပါတယ်။ ဒါက rollout စုချိန် update ခဏမလုပ်တာဖြစ်ပြီး DQN လို သီးခြား frozen target network ထားတာ မဟုတ်ပါဘူး။
