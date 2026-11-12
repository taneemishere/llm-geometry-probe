"""Cross-check: run posebusters (mol config, 0.25 tolerances) on every
P2-passing response's bond-determined mol; report agreement with our P3."""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from checker import determine_bonds_mol, parse_xyz

PB_CFG = {
    "modules": [{
        "name": "Geometry",
        "function": "distance_geometry",
        "parameters": {
            "threshold_bad_bond_length": 0.25,
            "threshold_bad_angle": 0.25,
            "threshold_clash": 0.25,
            "ignore_hydrogens": True,
            "sanitize": True,
        },
        "chosen_binary_test_output": [
            "bond_lengths_within_bounds", "bond_angles_within_bounds", "no_internal_clash"],
    }],
    "top_n": None,
    "loading": {"mol_pred": {"sanitize": True, "add_hs": False, "assign_stereo": False}},
}

OUT = Path("results/pb_crosscheck.csv")


def main():
    from posebusters import PoseBusters
    pb = PoseBusters(config=PB_CFG)
    resp = {tuple(map(str.strip, k)): json.loads(l)["content"]
            for l in open("results/responses.jsonl")
            for k in [(json.loads(l)["mol_id"], json.loads(l)["sample_idx"], json.loads(l)["model"])]}
    rows = list(csv.DictReader(open("results/checks.csv")))
    n_eval = agree = disagree = 0
    with OUT.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["mol_id", "sample_idx", "our_p3", "pb_bonds", "pb_angles", "pb_clash", "pb_pass", "agree"])
        for r in rows:
            if r["p2"] != "True":
                continue
            key = (r["mol_id"], r["sample_idx"], r["model"])
            content = resp.get(key)
            if content is None:
                continue
            atoms, _ = parse_xyz(content)
            mol = determine_bonds_mol(atoms)
            if mol is None:
                continue
            try:
                res = pb.bust([mol]).iloc[0]
                pb_pass = bool(res.get("bond_lengths_within_bounds", True)
                               and res.get("bond_angles_within_bounds", True)
                               and res.get("no_internal_clash", True))
            except Exception:
                continue
            our = r["p3"] == "True"
            n_eval += 1
            agree += int(our == pb_pass)
            disagree += int(our != pb_pass)
            w.writerow([r["mol_id"], r["sample_idx"], our,
                        res.get("bond_lengths_within_bounds"),
                        res.get("bond_angles_within_bounds"),
                        res.get("no_internal_clash"), pb_pass, our == pb_pass])
    print(f"evaluated {n_eval}; agreement {agree}/{n_eval} = {agree / max(1, n_eval):.3f}; disagreements {disagree}")


if __name__ == "__main__":
    main()
