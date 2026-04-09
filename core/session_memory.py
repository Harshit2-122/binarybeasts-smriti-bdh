"""
session_memory.py — Smriti Core
σ matrix save/load for cross-session Hebbian memory
"""

import torch
import os
import json
from datetime import datetime

MEMORY_DIR = "memory_store"
os.makedirs(MEMORY_DIR, exist_ok=True)


def save_session(patient_id: str, sigma: torch.Tensor, meta: dict = None):
    """
    Save σ matrix to disk after a patient session.
    This IS the memory — no external database needed.
    """
    path = os.path.join(MEMORY_DIR, f"{patient_id}_sigma.pt")
    torch.save(sigma, path)

    # Save metadata (visit date, symptoms summary)
    meta_path = os.path.join(MEMORY_DIR, f"{patient_id}_meta.json")
    meta = meta or {}
    meta["last_updated"] = datetime.now().isoformat()

    existing = []
    if os.path.exists(meta_path):
        with open(meta_path, "r") as f:
            existing = json.load(f)

    existing.append(meta)
    with open(meta_path, "w") as f:
        json.dump(existing, f, indent=2)

    size_kb = os.path.getsize(path) / 1024
    print(f"[Smriti] σ matrix saved for '{patient_id}' — {size_kb:.1f} KB")
    return size_kb


def load_session(patient_id: str):
    """
    Load σ matrix from disk — restoring full patient memory.
    Returns None if no prior session exists (new patient).
    """
    path = os.path.join(MEMORY_DIR, f"{patient_id}_sigma.pt")
    if not os.path.exists(path):
        print(f"[Smriti] No prior memory found for '{patient_id}' — fresh start")
        return None, []

    sigma = torch.load(path)

    meta_path = os.path.join(MEMORY_DIR, f"{patient_id}_meta.json")
    history = []
    if os.path.exists(meta_path):
        with open(meta_path, "r") as f:
            history = json.load(f)

    print(f"[Smriti] Memory loaded for '{patient_id}' — {len(history)} prior visit(s)")
    return sigma, history


def get_memory_size(patient_id: str):
    """Return σ matrix file size in KB — proof of O(n×d) constant memory"""
    path = os.path.join(MEMORY_DIR, f"{patient_id}_sigma.pt")
    if os.path.exists(path):
        return os.path.getsize(path) / 1024
    return 0


def list_patients():
    """List all patients with saved memory"""
    files = [f for f in os.listdir(MEMORY_DIR) if f.endswith("_sigma.pt")]
    return [f.replace("_sigma.pt", "") for f in files]
