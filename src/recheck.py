"""Recompute results/checks.csv from results/responses.jsonl (deduped)."""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from chem import mol_from_smiles
from checker import run_checks

RESP = Path("results/responses.jsonl")
CHECKS = Path("results/checks.csv")


def main():
    mols = {r["mol_id"]: r for r in csv.DictReader(open("data/molecules.csv"))}
    targets = {mid: (mol_from_smiles(r["smiles"]), r["formula"], bool(int(r["has_stereo"])))
               for mid, r in mols.items()}
    seen = set()
    with CHECKS.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["mol_id", "bin", "sample_idx", "model", "prompt_hash", "cached",
                    "p0", "p1", "p2", "p3", "pass", "stereo", "fail_reason", "p3_stats"])
        for ln in RESP.read_text().splitlines():
            rec = json.loads(ln)
            key = (rec["mol_id"], rec["sample_idx"], rec["model"])
            if key in seen or rec["mol_id"] not in targets:
                continue
            seen.add(key)
            out = run_checks(rec["content"], *targets[rec["mol_id"]])
            w.writerow([rec["mol_id"], rec["bin"], rec["sample_idx"], rec["model"],
                        rec["prompt_hash"], rec["cached"], out["p0"], out["p1"], out["p2"],
                        out["p3"], out["p0"] and out["p1"] and out["p2"] and out["p3"],
                        out["stereo"], out["fail_reason"], out["p3_stats"]])
    print(f"checks.csv rewritten ({len(seen)} rows)")


if __name__ == "__main__":
    main()
