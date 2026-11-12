"""Checker: P0 parse, P1 formula, P2 connectivity, P3 geometry, stereo.

P3 mirrors posebusters.modules.distance_geometry.check_geometry at tolerance
0.25 (bonds/angles within 0.75x-1.25x of DG bounds, noncovalent pairs
>= 0.75x lower bound), heavy atoms only. See docs/prereg.md.
"""
from __future__ import annotations

import math
import re

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.rdDistGeom import GetMoleculeBoundsMatrix
from rdkit.Chem import rdDetermineBonds

from chem import PT, formula_multiset, inchikey_conn

TOL = 0.25
CLASH_FRAC = 1.0 - TOL  # nonbonded pair must be >= 0.75 * lower bound
BOUNDS_PARAMS = dict(set15bounds=True, scaleVDW=True, doTriangleSmoothing=True, useMacrocycle14config=False)
CONJ_SMARTS = Chem.MolFromSmarts("[O,N;D1;$([O,N;D1]-[*]=[O,N;D1]),$([O,N;D1]=[*]-[O,N;D1])]~[*]")

ATOM_RE = re.compile(r"^\s*([A-Za-z]{1,3})\s+(\S+)\s+(\S+)\s+(\S+)\s*$")


def parse_xyz(text: str):
    """P0. Returns (atoms, err) where atoms = [(el, x, y, z), ...] or None."""
    t = text.strip()
    if t.startswith("```"):
        lines = t.split("\n")
        if len(lines) >= 2 and lines[-1].strip() == "```":
            t = "\n".join(lines[1:-1]).strip()
    lines = t.split("\n")
    while lines and not lines[-1].strip():
        lines.pop()
    if len(lines) < 3:
        return None, "too few lines"
    try:
        n = int(lines[0].strip())
    except ValueError:
        return None, "line 1 not an integer"
    if n <= 0 or len(lines) != n + 2:
        return None, f"count line {n} != {len(lines) - 2} atom lines"
    atoms = []
    for i, ln in enumerate(lines[2:]):
        m = ATOM_RE.match(ln)
        if not m:
            return None, f"atom line {i + 3} malformed"
        el, xs, ys, zs = m.groups()
        try:
            Z = PT.GetAtomicNumber(el.capitalize())
        except Exception:
            return None, f"atom line {i + 3}: bad element {el!r}"
        try:
            x, y, z = float(xs), float(ys), float(zs)
        except ValueError:
            return None, f"atom line {i + 3}: non-float coord"
        if not all(math.isfinite(v) for v in (x, y, z)):
            return None, f"atom line {i + 3}: non-finite coord"
        atoms.append((el.capitalize(), x, y, z))
    return atoms, None


def check_formula(atoms, target_formula: str):
    """P1: element multiset equals target formula."""
    got: dict[str, int] = {}
    for el, *_ in atoms:
        got[el] = got.get(el, 0) + 1
    return got == formula_multiset(target_formula)


def atoms_to_xyz_block(atoms) -> str:
    lines = [str(len(atoms)), ""]
    lines += [f"{el} {x:.6f} {y:.6f} {z:.6f}" for el, x, y, z in atoms]
    return "\n".join(lines)


def determine_bonds_mol(atoms):
    """RDKit mol with bonds determined from coordinates. None on failure."""
    mol = Chem.MolFromXYZBlock(atoms_to_xyz_block(atoms))
    if mol is None:
        return None
    try:
        rdDetermineBonds.DetermineBonds(mol, charge=0)
    except Exception:
        return None
    if mol.GetNumBonds() == 0:
        return None
    frags = Chem.GetMolFrags(mol)
    if len(frags) != 1:
        return None
    return mol


def check_graph(pred_mol, target_mol):
    """P2: connectivity via first InChIKey block."""
    try:
        return inchikey_conn(pred_mol) == inchikey_conn(target_mol)
    except Exception:
        return False


def _bpe(d, lb, ub):
    if d < lb:
        return (d - lb) / lb
    if d > ub:
        return (d - ub) / ub
    return 0.0


def check_geometry(pred_mol, tol: float = TOL):
    """P3: PoseBusters-style DG-bounds check. Returns (pass, stats dict)."""
    mol = pred_mol
    try:
        bounds = GetMoleculeBoundsMatrix(mol, **BOUNDS_PARAMS)
    except Exception as e:
        return False, {"error": f"bounds: {e}"}

    bonds = {tuple(sorted((b.GetBeginAtomIdx(), b.GetEndAtomIdx()))) for b in mol.GetBonds()}
    adj = {a.GetIdx(): {n.GetIdx() for n in a.GetNeighbors()} for a in mol.GetAtoms()}
    angles = set()
    for j, nbrs in adj.items():
        for a in nbrs:
            for c in nbrs:
                if a < c:
                    angles.add((a, c))

    n = mol.GetNumAtoms()
    lb = np.zeros((n, n))
    ub = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            lb[i, j] = bounds[j, i]
            ub[i, j] = bounds[i, j]

    # symmetrize conjugated terminal groups by element pair
    conj = [tuple(sorted(p)) for p in mol.GetSubstructMatches(CONJ_SMARTS)]
    groups: dict[tuple, list] = {}
    for p in conj:
        key = tuple(sorted(mol.GetAtomWithIdx(a).GetSymbol() for a in p))
        groups.setdefault(key, []).append(p)
    for key, plist in groups.items():
        lo = min(lb[i, j] for i, j in plist)
        hi = max(ub[i, j] for i, j in plist)
        for i, j in plist:
            lb[i, j], ub[i, j] = lo, hi

    conf = mol.GetConformer()
    pos = conf.GetPositions()
    heavy = [a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() > 1]
    dist = np.linalg.norm(pos[:, None, :] - pos[None, :, :], axis=-1)

    stats = {"bad_bonds": 0, "bad_angles": 0, "clashes": 0}
    for ii, i in enumerate(heavy):
        for j in heavy[ii + 1 :]:
            d = dist[i, j]
            if (i, j) in bonds:
                if abs(_bpe(d, lb[i, j], ub[i, j])) > tol:
                    stats["bad_bonds"] += 1
            elif (i, j) in angles:
                if abs(_bpe(d, lb[i, j], ub[i, j])) > tol:
                    stats["bad_angles"] += 1
            else:
                if d < CLASH_FRAC * lb[i, j]:
                    stats["clashes"] += 1
    ok = stats["bad_bonds"] == 0 and stats["bad_angles"] == 0 and stats["clashes"] == 0
    return ok, stats


def check_stereo(pred_mol, target_mol):
    """Compare CIP labels at target-specified centers. match/mismatch/unevaluable."""
    try:
        Chem.AssignStereochemistry(target_mol, force=True, cleanIt=True)
        Chem.AssignStereochemistryFrom3D(pred_mol)
        # map target atom idx -> pred atom idx; pred mol carries explicit Hs
        match = pred_mol.GetSubstructMatch(target_mol)
        if not match:
            t2 = Chem.RWMol(target_mol)
            p2 = Chem.RWMol(pred_mol)
            for m in (t2, p2):
                for b in m.GetBonds():
                    b.SetBondType(Chem.BondType.SINGLE)
                    b.SetIsAromatic(False)
                for a in m.GetAtoms():
                    a.SetIsAromatic(False)
            match = p2.GetSubstructMatch(t2)
            if not match:
                return "unevaluable"
        # match[i] = pred atom idx matching target atom i
        n_checked, n_bad = 0, 0
        for a in target_mol.GetAtoms():
            tag = a.GetProp("_CIPCode") if a.HasProp("_CIPCode") else None
            if tag is None:
                continue
            pa = pred_mol.GetAtomWithIdx(int(match[a.GetIdx()]))
            ptag = pa.GetProp("_CIPCode") if pa.HasProp("_CIPCode") else None
            n_checked += 1
            if ptag != tag:
                n_bad += 1
        if n_checked == 0:
            return "unevaluable"
        return "match" if n_bad == 0 else "mismatch"
    except Exception:
        return "unevaluable"


def run_checks(atoms_text: str, target_mol, target_formula: str, has_stereo: bool):
    """Full funnel on a raw response string. Returns dict of outcomes."""
    out = {"p0": False, "p1": False, "p2": False, "p3": False, "stereo": "na", "fail_reason": "", "p3_stats": ""}
    atoms, err = parse_xyz(atoms_text)
    if atoms is None:
        out["fail_reason"] = f"P0:{err}"
        return out
    out["p0"] = True
    if not check_formula(atoms, target_formula):
        out["fail_reason"] = "P1:formula mismatch"
        return out
    out["p1"] = True
    mol = determine_bonds_mol(atoms)
    if mol is None or not check_graph(mol, target_mol):
        out["fail_reason"] = "P2:connectivity mismatch"
        return out
    out["p2"] = True
    ok, stats = check_geometry(mol)
    out["p3_stats"] = str(stats)
    if not ok:
        out["fail_reason"] = f"P3:{stats}"
        return out
    out["p3"] = True
    if has_stereo:
        out["stereo"] = check_stereo(mol, target_mol)
    return out
