"""Controls: validate checker before the real run.

Positive: ETKDGv3 conformers round-tripped through XYZ -> expect >=95% pass.
Negative A: same + sigma=0.3A Gaussian noise -> expect ~0% pass.
Negative B: same with element labels permuted -> expect ~0% pass.
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np
from rdkit import Chem, RDLogger

sys.path.insert(0, str(Path(__file__).resolve().parent))
from chem import embed_mol, mol_from_smiles, mol_to_xyz
from checker import run_checks

RDLogger.DisableLog("rdApp.*")

MOLS = Path("data/molecules.csv")


def load_rows():
    return list(csv.DictReader(MOLS.open()))


def serialize(atoms, comment=""):
    lines = [str(len(atoms)), comment]
    lines += [f"{el} {x:.4f} {y:.4f} {z:.4f}" for el, x, y, z in atoms]
    return "\n".join(lines)


def run_controls():
    rows = load_rows()
    res = {"pos": [], "noise": [], "shuffle": []}
    rng = np.random.RandomState(7)
    for idx, r in enumerate(rows):
        target = mol_from_smiles(r["smiles"])
        mh = embed_mol(target, seed=0xC0FFEE + idx)
        atoms = []
        conf = mh.GetConformer()
        for i in range(mh.GetNumAtoms()):
            p = conf.GetAtomPosition(i)
            atoms.append([mh.GetAtomWithIdx(i).GetSymbol(), p.x, p.y, p.z])
        hs = bool(int(r["has_stereo"]))
        # positive
        out = run_checks(serialize(atoms), target, r["formula"], hs)
        res["pos"].append(out["p3"])
        # negative A: noise
        noisy = [[a[0], a[1] + rng.normal(0, 0.3), a[2] + rng.normal(0, 0.3), a[3] + rng.normal(0, 0.3)] for a in atoms]
        out = run_checks(serialize(noisy), target, r["formula"], hs)
        res["noise"].append(out["p3"])
        # negative B: permute element labels
        els = [a[0] for a in atoms]
        perm = rng.permutation(len(els))
        shuf = [[els[perm[i]], a[1], a[2], a[3]] for i, a in enumerate(atoms)]
        out = run_checks(serialize(shuf), target, r["formula"], hs)
        res["shuffle"].append(out["p3"])
    return {k: float(np.mean(v)) for k, v in res.items()}


if __name__ == "__main__":
    rates = run_controls()
    for k, v in rates.items():
        print(f"{k}: {v:.3f}")
    assert rates["pos"] >= 0.95, "positive control below 95% - checker is wrong"
    assert rates["noise"] <= 0.05, "noise control above 5% - checker too lax"
    assert rates["shuffle"] <= 0.05, "shuffle control above 5% - checker too lax"
    print("controls OK")
