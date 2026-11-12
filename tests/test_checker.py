"""Checker validation tests. Must pass before the real run."""
import sys
from pathlib import Path

import numpy as np
import pytest
from rdkit import Chem, RDLogger

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from chem import embed_mol, mol_from_smiles, mol_to_xyz
from checker import (check_formula, check_geometry, check_graph, check_stereo,
                     determine_bonds_mol, parse_xyz, run_checks)

RDLogger.DisableLog("rdApp.*")

ETHANOL = "CCO"
ETHANOL_FORMULA = "C2H6O"


def good_atoms(smiles=ETHANOL, seed=1):
    target = mol_from_smiles(smiles)
    mh = embed_mol(target, seed=seed)
    conf = mh.GetConformer()
    atoms = []
    for i in range(mh.GetNumAtoms()):
        p = conf.GetAtomPosition(i)
        atoms.append((mh.GetAtomWithIdx(i).GetSymbol(), p.x, p.y, p.z))
    return target, atoms


def xyz_block(atoms):
    lines = [str(len(atoms)), "c"]
    lines += [f"{e} {x:.4f} {y:.4f} {z:.4f}" for e, x, y, z in atoms]
    return "\n".join(lines)


def test_parse_ok():
    _, atoms = good_atoms()
    parsed, err = parse_xyz(xyz_block(atoms))
    assert err is None and len(parsed) == len(atoms)


def test_parse_fenced():
    _, atoms = good_atoms()
    parsed, err = parse_xyz("```xyz\n" + xyz_block(atoms) + "\n```\n")
    assert err is None


def test_parse_count_mismatch():
    _, atoms = good_atoms()
    bad = xyz_block(atoms).split("\n")
    bad[0] = str(len(atoms) + 1)
    parsed, err = parse_xyz("\n".join(bad))
    assert parsed is None


def test_parse_prose_fails():
    _, atoms = good_atoms()
    parsed, err = parse_xyz("Here is your molecule:\n" + xyz_block(atoms))
    assert parsed is None


def test_parse_bad_element():
    parsed, err = parse_xyz("1\nc\nXx 0 0 0")
    assert parsed is None


def test_parse_extra_field():
    parsed, err = parse_xyz("1\nc\nC 0 0 0 9")
    assert parsed is None


def test_formula():
    assert check_formula([("C", 0, 0, 0), ("O", 0, 0, 1), ("H", 0, 0, 0)], "CHO")
    assert not check_formula([("C", 0, 0, 0), ("N", 0, 0, 1)], "CO")


def test_graph_match():
    target, atoms = good_atoms()
    mol = determine_bonds_mol(atoms)
    assert mol is not None
    assert check_graph(mol, target)


def test_graph_mismatch():
    target = mol_from_smiles("CCN")  # ethylamine vs ethanol coords
    _, atoms = good_atoms()
    mol = determine_bonds_mol(atoms)
    assert not check_graph(mol, target)


def test_geometry_good():
    target, atoms = good_atoms("c1ccccc1", seed=3)
    mol = determine_bonds_mol(atoms)
    ok, stats = check_geometry(mol)
    assert ok, stats


def test_geometry_stretched_bond_fails():
    target, atoms = good_atoms("CCO", seed=5)
    # displace a methyl carbon 3A: stretches its C-C bond and 1-3 distance to O
    atoms = [(e, x, y, z) for e, x, y, z in atoms]
    c_idx = next(i for i, a in enumerate(atoms) if a[0] == "C")
    e, x, y, z = atoms[c_idx]
    atoms[c_idx] = (e, x + 3.0, y, z)
    mol = determine_bonds_mol(atoms)
    if mol is not None:
        ok, stats = check_geometry(mol)
        assert not ok


def test_run_checks_end_to_end():
    target, atoms = good_atoms("c1ccncc1", seed=9)  # pyridine
    out = run_checks(xyz_block(atoms), target, "C5H5N", False)
    assert out["p0"] and out["p1"] and out["p2"] and out["p3"], out


def test_stereo_match():
    target, atoms = good_atoms("C[C@H](N)C(=O)O", seed=11)  # L-alanine
    mol = determine_bonds_mol(atoms)
    assert mol is not None
    assert check_stereo(mol, target) == "match"


def test_controls_subset():
    from controls import serialize
    from rdkit import RDLogger as _  # noqa
    import csv
    rows = list(csv.DictReader(open("data/molecules.csv")))[:12]
    rng = np.random.RandomState(7)
    pos = []
    for idx, r in enumerate(rows):
        target = mol_from_smiles(r["smiles"])
        mh = embed_mol(target, seed=0xC0FFEE + idx)
        conf = mh.GetConformer()
        atoms = [[mh.GetAtomWithIdx(i).GetSymbol(), *conf.GetAtomPosition(i)] for i in range(mh.GetNumAtoms())]
        out = run_checks(serialize(atoms), target, r["formula"], bool(int(r["has_stereo"])))
        pos.append(out["p3"])
    assert np.mean(pos) >= 0.95
