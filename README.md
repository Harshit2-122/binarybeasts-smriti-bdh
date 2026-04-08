# 🧠 Smriti — स्मृति
### *The AI that never forgets your patient*

> **"India has 1 billion people. Most of them carry their entire medical history in their head — or on a paper register that gets lost, soaked, or forgotten. Transformers cannot solve this. BDH can."**

---

## 🏆 What Makes This Different

Most hackathon submissions **show** what BDH is.
Smriti **proves** what BDH can do that nothing else can.

| Capability                    | Typical Team | Smriti           |
|-------------------------------|--------------|------------------|
| Sparsity heatmaps             | ✅           | ✅ Included       |
| Cross-session Hebbian memory  | ❌           | ✅ Core demo      |
| Inference-time learning       | ❌           | ✅ Core demo      |
| GPT-2 baseline contrast       | Rarely       | ✅ Side-by-side   |
| Real-world domain proof       | Rarely       | ✅ Healthcare     |
| Interpretable synapse audit   | ❌           | ✅ Included       |
| Quantified experiments        | ❌           | ✅ 4 experiments  |

---

## 🎯 The Problem

```
India mein 3 tarah ke log hain:

👨‍🌾 Ram — Village mein
   → Nearest doctor 40km door
   → Doctor ke paas jaata hai — 6 mahine yaad nahi
   → Critical symptoms pichle visit se connect nahi hote

👩 Priya — City mein
   → 3 alag doctors. Kisi ke paas poori history nahi.
   → Reports ka dabba ghar mein khoya hua hai.
   → Ek doctor ka prescription doosra jaanta hi nahi.

👩‍⚕️ Sunita — ASHA Worker
   → 150 patients. Paper register.
   → Critical case miss ho jaata hai.
   → Koi alert system nahi.
```

**Teeno ki ek hi problem:**
> *"Koi unki health yaad nahi rakhta."*

### Scale of Impact

| Group               | Count          | Core Problem                             |
|---------------------|----------------|------------------------------------------|
| Rural patients      | 60 crore+      | No specialist access, fragmented history |
| Urban patients      | 50 crore+      | Multiple doctors, zero continuity        |
| ASHA workers        | 10 lakh        | Paper registers, no alert system         |
| Pregnant women      | 3 crore/year   | High risk, low monitoring                |
| **Total affected**  | **~1 Billion** | **No AI memory layer exists**            |

**Transformers cannot solve this** — they wake up fresh every session.
**BDH can** — because memory is native to its architecture.

---

## 💡 Solution: Smriti

Smriti is a **BDH-powered persistent health memory system** for every Indian.
Not a diagnosis chatbot. Not a symptom checker.

> **An architectural demonstration that post-transformer AI can remember across months, learn from new medical knowledge without retraining, and explain every clinical decision — with healthcare as the proof domain.**

---

## 🔬 Core Demonstrations (4 Experiments)

Each experiment is designed to be **falsifiable, reproducible, and directly mapped to a BDH architectural claim.**

---

### Experiment 1 — Cross-Session Hebbian Memory

**What we prove:**
BDH remembers patient context across completely separate sessions with zero external database. Transformers forget everything.

**Protocol:**
```
Session 1 — January 15:
  Input : "Mujhe 3 hafte se thakaan hai, pair sujan gaye hain"
  BDH   : Responds. σ matrix saved to disk. (torch.save)
  GPT-2 : Responds. No state saved.

Session 2 — February 3 (fresh start, no context window):
  Input : "Ab sans lene mein bhi takleef ho rahi hai"

  BDH   : Loads σ matrix from disk.
           "January mein jo pair ki sujan thi —
            combined with breathlessness —
            yeh cardiac risk pattern hai."

  GPT-2 : "Breathlessness ke liye doctor se milein."
           (No memory of January. No connection made.)
```

**Metrics measured:**
- Symptom connection accuracy across sessions (BDH vs GPT-2)
- σ matrix size before and after session (prove O(n×d) constant)
- Token count processed (target: 10k+ tokens, flat memory)

**Why only BDH:**
No KV-cache. No retrieval database. Hebbian synaptic state is the memory. This is architecturally native — not engineered.

---

### Experiment 2 — Inference-Time Literature Learning

**What we prove:**
BDH absorbs new medical knowledge during inference. No backpropagation. No fine-tuning. No API call. Transformers cannot do this.

**Protocol:**
```
Step 1 — Baseline test:
  Question : "WHO 2025 maternal iron supplementation guidelines kya hain?"
  BDH      : Does not know (not in training data)
  GPT-2    : Does not know

Step 2 — Feed new knowledge at inference time:
  Paste    : [WHO 2025 guideline paragraph — 200 words]
  BDH absorbs via Hebbian update. No training loop.

Step 3 — Post-ingestion test:
  Same question asked again.
  BDH      : Answers correctly using ingested knowledge.
  GPT-2    : Still does not know. Cannot learn at inference time.

Step 4 — Cross-session retention:
  σ matrix saved. New session started.
  BDH      : Still retains the ingested guideline.
  GPT-2    : Forgotten.
```

**Metrics measured:**
- Answer accuracy before vs after ingestion (BDH)
- Answer accuracy before vs after ingestion (GPT-2 baseline)
- Retention rate across sessions (σ matrix persistence test)

---

### Experiment 3 — Interpretable Synapse Audit

**What we prove:**
BDH's monosemantic synapses allow exact tracing of why a clinical flag was raised. Every decision is auditable. No transformer can offer this natively.

**Protocol:**
```
Patient input : "Thakaan, sujan, sans lene mein takleef"
BDH output    : "Cardiac risk — HIGH"

Click "Kyun?" →

Dashboard shows:
┌──────────────────────────────────────────┐
│ TOP ACTIVATED SYNAPSES                   │
│                                          │
│ Synapse 234  ████████░░  "fatigue"       │
│ Synapse 891  ███████░░░  "swelling"      │
│ Synapse 445  █████████░  "breathless"    │
│ Synapse 112  ██████░░░░  "cardiac"       │
│                                          │
│ Token → Synapse map:                     │
│ "sujan"   → Synapse 891 (0.87)           │
│ "sans"    → Synapse 445 (0.92)           │
│ "thakaan" → Synapse 234 (0.79)           │
└──────────────────────────────────────────┘

Transformer baseline:
"Cardiac risk — probability: 0.73"
[No explanation possible without post-hoc methods]
```

**Metrics measured:**
- % of activated synapses per token (target: ~5% — paper claim)
- Synapse consistency: does Synapse 891 always activate for swelling-related tokens? (monosemanticity test)
- Comparison: BDH native interpretability vs SHAP on GPT-2 (complexity + accuracy)

---

### Experiment 4 — Memory Scaling Proof

**What we prove:**
BDH memory stays flat as context grows. GPT-2 crashes. Exact replication of glass-brain methodology — applied to medical domain.

**Protocol:**
```
Hardware : T4 GPU (Google Colab free tier)
Input    : Increasing token lengths — 1k, 5k, 10k, 20k, 50k

Measure  :
  - GPU memory usage (MB) at each length
  - Inference time (ms) at each length
  - Crash point for GPT-2

Expected result (based on glass-brain findings):
  BDH   : Flat memory at 50k+ tokens
  GPT-2 : Crash at ~12k tokens
```

**Metrics measured:**
- GPU memory (MB) vs token count — plotted graph
- Inference latency vs token count
- Exact crash point of transformer baseline

**Output:** A reproducible graph showing BDH's O(n×d) vs transformer's O(T²) — the paradigm shift made visual.

---

## 📊 Visualization Suite

These visuals make BDH's advantages **visceral, not theoretical.**

| Visual                   | What it shows                      | Why it matters                |
|--------------------------|------------------------------------|-------------------------------|
| Memory scaling graph     | BDH flat vs GPT-2 crash            | O(1) memory proof             |
| Sparsity heatmap         | 5% vs 100% activation              | Efficiency + interpretability |
| σ matrix evolution       | Synapses strengthening over visits | Hebbian learning made visible |
| Synapse audit dashboard  | Token → synapse → diagnosis        | Native interpretability       |
| Side-by-side comparison  | BDH remembers, GPT-2 forgets       | Paradigm shift in 10 seconds  |

---

## 🏗️ Architecture

```
smriti-bdh/
│
├── core/
│   ├── bdh_model.py           # pathwaycom/bdh — unchanged architecture
│   ├── session_memory.py      # σ matrix save/load (torch.save/load)
│   └── hebbian_ingest.py      # Inference-time literature absorption
│
├── experiments/
│   ├── exp1_cross_session.py  # Visit 1 → Visit 2 memory demo
│   ├── exp2_live_learning.py  # WHO guideline ingestion proof
│   ├── exp3_synapse_audit.py  # Monosemanticity dashboard
│   └── exp4_memory_scale.py   # BDH flat vs GPT-2 crash graph
│
├── baseline/
│   └── gpt2_compare.py        # GPT-2 / nanoGPT side-by-side
│
├── ui/
│   └── streamlit_app.py       # 3-visit demo + audit dashboard
│
└── data/
    ├── medical_qa/            # PubMedQA / MedQA samples
    └── who_guidelines/        # Inference-time learning corpus
```

---

## 🛠️ Tech Stack

| Component      | Tool                     | Why                         |
|----------------|--------------------------|-----------------------------|
| BDH Model      | pathwaycom/bdh (PyTorch) | Official implementation     |
| Session State  | torch.save / torch.load  | Persist σ matrix            |
| UI             | Streamlit                | Fast, HuggingFace-hostable  |
| Baseline       | nanoGPT / GPT-2          | Fair comparison             |
| Synapse Viz    | Plotly heatmap           | Interactive audit dashboard |
| Memory Graph   | Matplotlib               | Scaling proof               |
| Hosting        | HuggingFace Spaces       | Free, permanent demo        |
| Data           | PubMedQA, MedQA          | Open medical Q&A            |

---

## 🎬 3-Minute Demo Flow

```
0:00 — The Problem
       "Yeh patient January mein aayi thi. Aaj February hai.
        GPT-2 ko kuch yaad nahi. Dekhte hain BDH ko."

0:30 — Visit 1 (Live)
       Patient types January symptoms.
       BDH responds. σ matrix saved. File size shown on screen.

1:00 — Visit 2 (The Proof)
       New session. No context window. Only σ matrix loaded.
       BDH connects January + February symptoms.
       GPT-2 shown side-by-side — no memory. No connection.

1:30 — Live Learning (The Wow Moment)
       WHO guideline pasted.
       BDH asked same question — before and after.
       Before: doesn't know. After: answers correctly.
       GPT-2: still doesn't know.

2:00 — Synapse Audit
       "Kyun HIGH risk bola?" clicked.
       Dashboard shows exact synapses, exact tokens.
       Transformer: "probability 0.73 — no explanation."

2:30 — Memory Scaling Graph
       Graph shown: BDH flat line to 50k tokens.
       GPT-2 red X at ~12k — crashed.

2:50 — Impact
       "10 lakh ASHA workers. 1 billion patients.
        This is what post-transformer AI looks like."
```

---

## 📐 Experimental Rigor

Every claim in this project is:

- **Falsifiable** — exact metrics defined before experiments run
- **Reproducible** — all scripts open source, seed values fixed
- **Honest** — limitations documented clearly (see below)
- **Comparable** — GPT-2 baseline runs on identical hardware

### Limitations (Documented Honestly)

| Limitation                               | What it means                                                         |
|------------------------------------------|-----------------------------------------------------------------------|
| Small model scale                        | BDH trained on small medical corpus — not production-grade            |
| Synthetic patient data                   | Demo uses generated visits, not real patient records                  |
| Synapse monosemanticity is probabilistic | Not every synapse is perfectly monosemantic — paper acknowledges this |
| Inference-time learning has bounds       | BDH cannot learn from unlimited text at inference time                |
| No clinical validation                   | This is an architectural demonstration, not a certified medical tool  |

---

## 🔭 Future Scope

*The following features are planned extensions — not part of this submission — representing the full vision of Smriti as a deployable system.*

**User System**
- New user onboarding — profile, village, age, basic health info
- Returning user recognition via local σ matrix storage
- Privacy-first architecture — no central server, data stays on device

**Community Health Network**
- Permission-based ASHA connection — "Kya main aapki nearest ASHA worker ko share karoon?"
- Critical alert propagation to PHC doctors and district health officers
- Area-level health dashboard for epidemiological monitoring

**Pregnant Women Module**
- Dedicated trimester tracking
- WHO maternal guidelines pre-loaded at inference time
- High-risk flag → instant ASHA alert

**Doctor Integration**
- Time-range health summary generation for doctor visits
- Report photo upload with automatic parsing into σ matrix
- Doctor correction workflow — update AI suggestion post-diagnosis
- Doctor-verified diagnosis stored as ground truth

**Scale**
- Multilingual support (Hindi, Tamil, Telugu, Bengali)
- Offline mode — σ matrix on low-RAM devices
- District-level aggregate synapse analysis for outbreak detection

---

## 📋 Judging Criteria Alignment

### Path C — Open-Ended Frontier Exploration

| Criterion         | Weight | Our Answer                                                                                                                           |
|-------------------|--------|--------------------------------------------------------------------------------------------------------------------------------------|
| **Novelty**       | 30%    | No prior submission combines cross-session Hebbian memory + inference-time learning + interpretable synapse audit in one domain proof |
| **Rigor**         | 40%    | 4 experiments. Defined metrics. GPT-2 baseline. Reproducible scripts. Honest limitations section.                                    |
| **Communication** | 30%    | 3-minute demo. One story: "BDH remembers. GPT-2 forgets." Synapse dashboard makes interpretability visible to non-experts.           |

### BDH Capabilities Covered

| BDH Capability               | Paper Section   | Smriti Demonstration                    |
|------------------------------|-----------------|-----------------------------------------|
| Constant-size Hebbian memory | Section 3.2, 6  | Experiment 1 — cross-session σ matrix   |
| Inference-time learning      | Section 7.2     | Experiment 2 — WHO guideline ingestion  |
| Monosemantic synapses        | Section 6.3     | Experiment 3 — synapse audit dashboard  |
| ~5% sparse activations       | Section 6.4     | Experiment 3 + sparsity heatmap         |
| O(T) linear attention        | Section 3       | Experiment 4 — memory scaling graph     |

---

## 📚 Key References

| Resource                | URL                                   | Why                              |
|-------------------------|---------------------------------------|----------------------------------|
| BDH Repository          | github.com/pathwaycom/bdh             | Core model                       |
| Research Paper          | arxiv.org/abs/2509.26507              | Sections 3, 6, 7 critical        |
| glass-brain (IIT Madras)| github.com/sharmilcd/glass-brain      | Memory scaling methodology       |
| Lig17/Track2            | github.com/Lig17/Track2              | Inference-time Hebbian reference |
| PubMedQA                | pubmedqa.github.io                    | Training data                    |
| HuggingFace BDH         | huggingface.co/papers/2509.26507      | Community discussion             |

---

## 🚀 How to Run Locally

```bash
git clone https://github.com/binarybeasts/smriti-bdh
cd smriti-bdh
pip install -r requirements.txt
streamlit run ui/streamlit_app.py
```

**requirements.txt:**
```
torch
streamlit
plotly
transformers
numpy
matplotlib
datasets
```

---

## 🌐 Live Demo

HuggingFace Space: *[Coming soon — will be live at judging time]*

---

## 🎥 Demo Video

YouTube: *[Coming soon — 3-minute walkthrough of all 4 experiments]*

Covers:
- Cross-session memory proof (BDH vs GPT-2)
- Live WHO guideline ingestion
- Synapse audit dashboard
- Memory scaling graph

---

## 👥 Team — Binary Beasts

| Name        | Role                                                            |
|-------------|-----------------------------------------------------------------|
| **Harshit** | BDH model, experiments, σ matrix implementation, synapse audit  |
| **Karan**   | Streamlit UI, visualization, HuggingFace deployment, demo video |

---

## 🔑 One-Line Pitch

> *"Smriti is not a diagnosis app. It is proof that post-transformer AI can remember a patient across months without a database, learn new medical guidelines without retraining, and explain every decision through interpretable synapses — three things transformers fundamentally cannot do, demonstrated live, measured, and reproducible."*

---

*BDH Frontier Challenge | Post-Transformer Hackathon by Pathway | IIT Ropar | 2026*
