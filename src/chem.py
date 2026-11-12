"""Shared chemistry helpers: molecule loading, embedding, formula, XYZ IO."""
from __future__ import annotations

import re

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, rdMolDescriptors

ALLOWED_ATOMIC_NUMS = {1, 5, 6, 7, 8, 9, 14, 15, 16, 17, 35, 53}  # H,B,C,N,O,F,Si,P,S,Cl,Br,I

PT = Chem.GetPeriodicTable()


def mol_from_smiles(smiles: str):
    mol = Chem.MolFromSmiles(smiles)
    return mol


def is_eligible(mol) -> bool:
    """Prereg filters: single fragment, no charged atoms, allowed elements."""
    if mol is None:
        return False
    if "." in Chem.MolToSmiles(mol):
        return False
    for atom in mol.GetAtoms():
        if atom.GetFormalCharge() != 0:
            return False
        if atom.GetAtomicNum() not in ALLOWED_ATOMIC_NUMS:
            return False
    return True


def embeddable(mol, seed: int = 0xF00D) -> bool:
    mh = Chem.AddHs(mol)
    params = AllChem.ETKDGv3()
    params.randomSeed = seed
    return AllChem.EmbedMolecule(mh, params) == 0


def explicit_formula(mol) -> str:
    return rdMolDescriptors.CalcMolFormula(Chem.AddHs(mol))


def formula_multiset(formula: str) -> dict[str, int]:
    """'C2H6O' -> {'C': 2, 'H': 6, 'O': 1}"""
    return {el: int(n or 1) for el, n in re.findall(r"([A-Z][a-z]?)(\d*)", formula)}


def n_heavy(mol) -> int:
    return sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() > 1)


def n_stereo_centers(mol) -> int:
    return len(Chem.FindMolChiralCenters(mol, includeUnassigned=False, useLegacyImplementation=False))


def embed_mol(mol, seed: int):
    """Return a copy of mol with explicit Hs and one ETKDGv3 conformer."""
    mh = Chem.AddHs(mol)
    params = AllChem.ETKDGv3()
    params.randomSeed = seed
    status = AllChem.EmbedMolecule(mh, params)
    if status != 0:
        raise RuntimeError(f"embed failed: {status}")
    AllChem.MMFFOptimizeMolecule(mh, maxIters=200)
    return mh


def mol_to_xyz(mol, comment: str = "") -> str:
    conf = mol.GetConformer()
    lines = [str(mol.GetNumAtoms()), comment]
    for i in range(mol.GetNumAtoms()):
        p = conf.GetAtomPosition(i)
        sym = mol.GetAtomWithIdx(i).GetSymbol()
        lines.append(f"{sym} {p.x:.4f} {p.y:.4f} {p.z:.4f}")
    return "\n".join(lines)


def inchikey_conn(mol) -> str:
    return Chem.MolToInchiKey(mol).split("-")[0]
