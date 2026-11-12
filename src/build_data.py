"""Build data/molecules.csv from the frozen Delaney ESOL csv (prereg spec)."""
from __future__ import annotations

import csv
import hashlib
import sys
from pathlib import Path

import numpy as np
from rdkit import Chem, RDLogger

sys.path.insert(0, str(Path(__file__).resolve().parent))
from chem import embeddable, explicit_formula, is_eligible, n_heavy, n_stereo_centers

RDLogger.DisableLog("rdApp.*")

RAW = Path("data/raw/delaney-processed.csv")
OUT = Path("data/molecules.csv")
SEED = 20261112
BINS = {"A": (5, 10), "B": (11, 20), "C": (21, 30)}
TARGET = 33


def main():
    rows = []
    for i, rec in enumerate(csv.DictReader(RAW.open())):
        smiles = rec["smiles"].strip()
        mol = Chem.MolFromSmiles(smiles)
        if not is_eligible(mol) or not embeddable(mol):
            continue
        nh = n_heavy(mol)
        for b, (lo, hi) in BINS.items():
            if lo <= nh <= hi:
                rows.append(
                    {
                        "mol_id": f"esol_{i:04d}",
                        "smiles": Chem.MolToSmiles(mol),
                        "formula": explicit_formula(mol),
                        "n_heavy": nh,
                        "n_atoms": mol.GetNumAtoms() + sum(a.GetTotalNumHs() for a in mol.GetAtoms()),
                        "n_stereo": n_stereo_centers(mol),
                        "bin": b,
                    }
                )
                break

    rng = np.random.RandomState(SEED)
    picked = []
    for b in BINS:
        cand = sorted([r["mol_id"] for r in rows if r["bin"] == b])
        take = min(TARGET, len(cand))
        if take < TARGET:
            print(f"SHORTFALL bin {b}: {take} < {TARGET}")
        idx = rng.choice(len(cand), take, replace=False) if cand else []
        sel = set(np.array(cand)[idx]) if cand else set()
        picked.extend(r for r in rows if r["mol_id"] in sel and r["bin"] == b)

    picked.sort(key=lambda r: r["mol_id"])
    with OUT.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["mol_id", "smiles", "formula", "n_heavy", "n_atoms", "n_stereo", "has_stereo", "bin"])
        w.writeheader()
        for r in picked:
            r["has_stereo"] = int(r["n_stereo"] > 0)
            w.writerow(r)
    print(f"wrote {len(picked)} molecules -> {OUT}")
    print("sha256", hashlib.sha256(OUT.read_bytes()).hexdigest())


if __name__ == "__main__":
    main()
