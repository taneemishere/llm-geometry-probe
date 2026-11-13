"""v2 tests: feedback builder, leakage scan, loop semantics, cache keys."""
import json
import sys
from pathlib import Path

import numpy as np
import pytest
from rdkit import Chem, RDLogger

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from chem import embed_mol, mol_from_smiles
from feedback import (BINARY_MESSAGE, CLOSER, build_feedback, diagnose)
import repair

RDLogger.DisableLog("rdApp.*")

ETH = mol_from_smiles("CCO")
ETH_F = "C2H6O"


def atoms_of(smiles, seed=1):
    mh = embed_mol(mol_from_smiles(smiles), seed=seed)
    conf = mh.GetConformer()
    return [(mh.GetAtomWithIdx(i).GetSymbol(), *conf.GetAtomPosition(i)) for i in range(mh.GetNumAtoms())]


def xyz(atoms, n=None, comment=""):
    lines = [str(len(atoms) if n is None else n), comment]
    lines += [f"{e} {x:.4f} {y:.4f} {z:.4f}" for e, x, y, z in atoms]
    return "\n".join(lines)


def test_feedback_p0_count_mismatch():
    atoms = atoms_of("CCO")
    stage, det = diagnose(xyz(atoms, n=len(atoms) + 1), ETH, ETH_F)
    assert stage == "P0" and det["code"] == "count_mismatch"
    fb = build_feedback(stage, det)
    assert f"count line says {len(atoms) + 1} atoms but the body contains {len(atoms)} atom lines" in fb
    assert fb.endswith(CLOSER)


def test_feedback_p0_bad_element():
    bad = "3\nc\nC 0 0 0\nXx 1 0 0\nO 2 0 0"
    stage, det = diagnose(bad, ETH, ETH_F)
    assert stage == "P0" and det["code"] == "bad_element"
    assert "'Xx' is not a valid element symbol" in build_feedback(stage, det)


def test_feedback_p0_malformed():
    bad = "3\nc\nC 0 0 0\nH 1 0\nO 2 0 0"
    stage, det = diagnose(bad, ETH, ETH_F)
    assert stage == "P0" and det["code"] == "bad_line"
    assert "line 4 is malformed" in build_feedback(stage, det)


def test_feedback_p1_diff():
    atoms = atoms_of("CCO") + [("H", 0.0, 0.0, 0.0)]  # extra H -> C2H7O
    stage, det = diagnose(xyz(atoms), ETH, ETH_F)
    assert stage == "P1"
    fb = build_feedback(stage, det)
    assert "Target formula: C2H6O" in fb and "Your formula: C2H7O" in fb
    assert "H: expected 6, got 7" in fb


def test_feedback_p2_connectivity():
    # dimethyl ether atoms vs ethanol target: same formula, different graph
    atoms = atoms_of("COC", seed=4)
    stage, det = diagnose(xyz(atoms), ETH, ETH_F)
    assert stage == "P2" and det.get("s_pred")
    fb = build_feedback(stage, det, ETH)
    assert "SMILES inferred from your coordinates" in fb
    assert "Target SMILES: CCO" in fb
    assert det["s_pred"] == "COC"


def test_feedback_p3_real_response():
    """Real v1 P3 failure (esol_0122 sample 0): diagnose P3 and format message."""
    resp_path = Path("results/responses.jsonl")
    if not resp_path.exists():
        pytest.skip("v1 results not present")
    mols = {r["mol_id"]: r for r in
            __import__("csv").DictReader(open("data/molecules.csv"))}
    for ln in resp_path.open():
        r = json.loads(ln)
        if r["mol_id"] == "esol_0122" and r["sample_idx"] == 0:
            row = mols["esol_0122"]
            t = mol_from_smiles(row["smiles"])
            stage, det = diagnose(r["content"], t, row["formula"])
            assert stage == "P3"
            fb = build_feedback(stage, det, t)
            assert "fails geometry validation" in fb
            assert "atom pair" in fb and fb.endswith(CLOSER)
            return
    pytest.skip("response not found")


def test_leakage_scan():
    """Feedback must never contain reference conformer coordinates."""
    atoms = atoms_of("CCO", seed=99)
    ref = np.array([[x, y, z] for _, x, y, z in atoms])
    cases = [
        xyz(atoms, n=len(atoms) + 1),                    # P0
        xyz(atoms + [("H", 9.9, 9.9, 9.9)]),             # P1
        xyz(atoms_of("COC", seed=4)),                    # P2
    ]
    for content in cases:
        stage, det = diagnose(content, ETH, ETH_F)
        fb = build_feedback(stage, det, ETH)
        for v in ref.flatten():
            s = f"{v:.4f}"
            if len(s) > 5 and v != 0:
                assert s not in fb, f"leak: {s} in {fb!r}"


class FakeClient:
    def __init__(self, script):
        self.script = script  # list of response contents or 'pass' marker
        self.calls = []

    def chat_messages(self, messages, key_parts):
        self.calls.append(key_parts)
        n = len(self.calls)
        resp = self.script[min(n - 1, len(self.script) - 1)]
        if resp == "__error__":
            raise RuntimeError("HTTP Error 429")
        return {"content": resp, "usage": {}, "cached": False}


class FakeFile:
    def __init__(self):
        self.buf = []

    def write(self, s):
        self.buf.append(s)

    def flush(self):
        pass


def _row():
    return {"mol_id": "esol_x", "smiles": "CCO", "formula": ETH_F,
            "n_atoms": "9", "has_stereo": "0", "bin": "A"}


def test_loop_stops_at_first_pass():
    good = xyz(atoms_of("CCO", seed=7))
    bad = "9\nc\nC 0 0 0\n" + "\n".join(f"H {i} 0 0" for i in range(8))
    client = FakeClient([bad, good, good])
    v1_r0 = {"esol_x": bad}
    recs, finished = repair.run_chain(_row(), "struct", client, v1_r0, FakeFile())
    assert len(client.calls) == 2          # round0(bad) -> r1 bad -> r2 good
    assert finished and recs[-1]["pass"] is True
    assert all(r["round"] == i for i, r in enumerate(recs))


def test_loop_max_four_calls():
    bad = "9\nc\nC 0 0 0\n" + "\n".join(f"H {i} 0 0" for i in range(8))
    client = FakeClient([bad])
    v1_r0 = {"esol_x": bad}
    recs, finished = repair.run_chain(_row(), "binary", client, v1_r0, FakeFile())
    assert len(client.calls) == 3          # rounds 1,2,3
    assert finished and not recs[-1]["pass"]


def test_call_error_not_a_result():
    bad = "9\nc\nC 0 0 0\n" + "\n".join(f"H {i} 0 0" for i in range(8))
    client = FakeClient(["__error__"])
    recs, finished = repair.run_chain(_row(), "struct", client, {"esol_x": bad}, FakeFile())
    assert not finished                      # unfinished chain gets retried
    assert all("pass" in r for r in recs)    # only completed rounds recorded


def test_cache_key_distinguishes():
    msgs = [{"role": "user", "content": "p"}]
    import hashlib
    k1 = hashlib.sha256(f"m|{json.dumps(msgs, sort_keys=True)}|struct|round1".encode()).hexdigest()
    k2 = hashlib.sha256(f"m|{json.dumps(msgs, sort_keys=True)}|binary|round1".encode()).hexdigest()
    k3 = hashlib.sha256(f"m|{json.dumps(msgs, sort_keys=True)}|struct|round2".encode()).hexdigest()
    assert len({k1, k2, k3}) == 3
