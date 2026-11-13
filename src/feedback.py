"""Deterministic verifier-feedback builder (prereg v2, arm c).

Feedback contains only info about the model's own output plus the SMILES and
formula already in the prompt. No reference geometry ever appears.
"""
from __future__ import annotations

from rdkit import Chem

from checker import (check_formula, check_graph, determine_bonds_mol,
                     geometry_violations, parse_xyz_detail)
from chem import formula_multiset

BINARY_MESSAGE = "The structure failed validation. Please output a corrected XYZ block only."
CLOSER = "Output a corrected XYZ block only, same format as before."


def formula_diff(atoms, target_formula: str):
    """Per-element {el: (expected, got)} for elements whose counts differ."""
    exp = formula_multiset(target_formula)
    got: dict[str, int] = {}
    for el, *_ in atoms:
        got[el] = got.get(el, 0) + 1
    diff = {}
    for el in sorted(set(exp) | set(got)):
        if exp.get(el, 0) != got.get(el, 0):
            diff[el] = (exp.get(el, 0), got.get(el, 0))
    return diff, exp, got


def formula_str(ms: dict[str, int]) -> str:
    order = {"C": 0, "H": 1}
    return "".join(f"{el}{n}" for el, n in sorted(ms.items(), key=lambda kv: (order.get(kv[0], 2), kv[0])))


def diagnose(content: str, target_mol, target_formula: str):
    """First-failed stage plus details for feedback. stage in P0..P3, 'pass'."""
    atoms, det = parse_xyz_detail(content)
    if atoms is None:
        return "P0", det
    if not check_formula(atoms, target_formula):
        diff, exp, got = formula_diff(atoms, target_formula)
        return "P1", {"diff": diff, "expected": exp, "got": got}
    mol = determine_bonds_mol(atoms)
    if mol is None:
        return "P2", {"s_pred": None}
    if not check_graph(mol, target_mol):
        try:
            s_pred = Chem.MolToSmiles(Chem.RemoveHs(mol))
        except Exception:
            s_pred = Chem.MolToSmiles(mol)
        return "P2", {"s_pred": s_pred, "nb": mol.GetNumBonds()}
    ok, stats = check_geometry_local(mol)
    if not ok:
        return "P3", stats
    return "pass", {}


def check_geometry_local(mol):
    stats, viol = geometry_violations(mol)
    ok = stats.get("bad_bonds", 1) == 0 and stats.get("bad_angles", 1) == 0 and stats.get("clashes", 1) == 0
    stats = dict(stats)
    stats["violations"] = viol
    return ok, stats


def build_feedback(stage: str, det: dict, target_mol=None) -> str:
    """Structured feedback message for a stage. Never includes geometry ref."""
    if stage == "P0":
        code = det.get("code")
        if code == "count_mismatch":
            reason = (f"the count line says {det['n']} atoms but the body contains "
                      f"{det['m']} atom lines")
        elif code == "bad_line":
            reason = (f"line {det['line_no']} is malformed: '{det['line']}'. "
                      f"Each atom line must be 'Element x y z'")
        elif code == "bad_element":
            reason = f"line {det['line_no']}: '{det['el']}' is not a valid element symbol"
        elif code == "bad_count":
            reason = "line 1 is not a positive integer"
        else:
            reason = "the response must contain only the XYZ block; remove all prose"
        return f"Your previous response failed format validation: {reason}. {CLOSER}"
    if stage == "P1":
        diffs = "; ".join(f"{el}: expected {e}, got {g}" for el, (e, g) in det["diff"].items())
        return (f"Your previous response failed formula validation. Target formula: "
                f"{formula_str(det['expected'])}. Your formula: {formula_str(det['got'])}. "
                f"Differences: {diffs}. {CLOSER}")
    if stage == "P2":
        if det.get("s_pred"):
            nb_t = target_mol.GetNumBonds() if target_mol is not None else "?"
            s_t = Chem.MolToSmiles(target_mol) if target_mol is not None else "?"
            return (f"Your previous response fails connectivity validation. SMILES inferred "
                    f"from your coordinates: {det['s_pred']} ({det['nb']} bonds). "
                    f"Target SMILES: {s_t} ({nb_t} bonds). {CLOSER}")
        return (f"Your previous response fails connectivity validation: no consistent bond "
                f"structure could be inferred from your coordinates. {CLOSER}")
    if stage == "P3":
        lines = ["Your previous response fails geometry validation. Worst violations:"]
        for v in det.get("violations", [])[:5]:
            if v["kind"] == "clash":
                lines.append(f"atom pair ({v['i']},{v['j']}) clash: {v['d']:.2f} A below "
                             f"{v['lo']:.2f} A minimum")
            else:
                lines.append(f"atom pair ({v['i']},{v['j']}) {v['kind']}: {v['d']:.2f} A "
                             f"outside [{v['lo']:.2f},{v['hi']:.2f}]")
        lines.append(CLOSER)
        return "\n".join(lines)
    return BINARY_MESSAGE
