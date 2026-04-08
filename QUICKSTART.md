# Smriti — Quick Start

## Setup (2 minutes)
```bash
git clone https://github.com/binarybeasts/smriti-bdh
cd smriti-bdh
pip install -r requirements.txt
```

## Run Everything
```bash
python run_all.py          # all 4 experiments in terminal
```

## Run Individual Experiments
```bash
python experiments/exp1_cross_session.py   # cross-session memory
python experiments/exp2_live_learning.py   # inference-time learning
python experiments/exp3_synapse_audit.py   # synapse interpretability
python experiments/exp4_memory_scale.py    # memory scaling graph
python baseline/gpt2_compare.py            # full comparison table
```

## Launch UI (Demo)
```bash
streamlit run ui/streamlit_app.py
```
Then open: http://localhost:8501

## Demo Flow (3 minutes)
1. Sidebar → enter "ram_village_01" → Load Patient Memory
2. Tab 1 → type January symptoms → Analyze (BDH)
3. Tab 1 → type February symptoms → see cross-session connection
4. Tab 2 → Synapse Audit → click "Kyun?"
5. Tab 3 → Memory Scaling graph

## File Structure
```
smriti-bdh/
├── core/
│   ├── bdh_model.py        # HebbianMemory + SmritiBDH
│   ├── session_memory.py   # σ matrix save/load
│   └── hebbian_ingest.py   # inference-time ingestion
├── experiments/
│   ├── exp1_cross_session.py
│   ├── exp2_live_learning.py
│   ├── exp3_synapse_audit.py
│   └── exp4_memory_scale.py
├── baseline/
│   └── gpt2_compare.py
├── ui/
│   └── streamlit_app.py
├── run_all.py
└── requirements.txt
```

## HuggingFace Spaces Deployment
1. Create new Space → Streamlit
2. Upload all files
3. requirements.txt is already configured
4. Main file: ui/streamlit_app.py
