"""
run_all.py — Run all 4 Smriti experiments in sequence
Usage: python run_all.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

print("🧠 SMRITI — स्मृति")
print("BDH Frontier Challenge | Binary Beasts | IIT Ropar 2026")
print("=" * 60)
print("Running all 4 experiments...\n")

from experiments.exp1_cross_session import run_experiment as exp1
print("\n" + "━" * 60)
exp1()

from experiments.exp2_live_learning import run_experiment as exp2
print("\n" + "━" * 60)
exp2()

from experiments.exp3_synapse_audit import run_experiment as exp3
print("\n" + "━" * 60)
exp3()

from experiments.exp4_memory_scale import run_experiment as exp4
print("\n" + "━" * 60)
exp4()

from baseline.gpt2_compare import run_comparison
print("\n" + "━" * 60)
run_comparison()

print("\n\n" + "=" * 60)
print("✅ ALL EXPERIMENTS COMPLETE")
print()
print("Next steps:")
print("  → streamlit run ui/streamlit_app.py  (live demo)")
print("  → outputs/memory_scaling_graph.png   (Exp 4 graph)")
print("  → memory_store/                       (σ matrices)")
print("=" * 60)
