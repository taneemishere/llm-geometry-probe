"""Run the experiment: k samples per molecule through NIM, then check."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from chem import mol_from_smiles
from checker import run_checks
from client import NIMClient, estimated_cost, BUDGET_USD
from prompts import build_prompt

RESP = Path("results/responses.jsonl")
CHECKS = Path("results/checks.csv")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="nvidia/nemotron-3-ultra-550b-a55b")
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--max-mols", type=int, default=0, help="debug: limit molecules")
    ap.add_argument("--max-samples", type=int, default=0, help="debug: limit k")
    ap.add_argument("--extra-json", default="{}", help="extra request params")
    args = ap.parse_args()

    api_key = Path("nvapi_key").read_text().strip()
    client = NIMClient(api_key, args.model, extra=json.loads(args.extra_json))

    rows = list(csv.DictReader(open("data/molecules.csv")))
    if args.max_mols:
        rows = rows[: args.max_mols]
    k = args.max_samples or args.k

    tasks = []
    for r in rows:
        prompt = build_prompt(r["smiles"], r["formula"], int(r["n_atoms"]))
        ph = hashlib.sha256(prompt.encode()).hexdigest()[:16]
        for s in range(k):
            tasks.append((r, prompt, ph, s))
    print(f"{len(tasks)} calls planned; est cost so far ${estimated_cost():.2f}")

    meta = {"model": args.model, "k": k, "temperature": 1.0, "top_p": 1.0,
            "max_tokens": 8192, "extra": json.loads(args.extra_json),
            "price_in_per_mtok": 3.0, "price_out_per_mtok": 9.0,
            "started": time.strftime("%Y-%m-%dT%H:%M:%S")}
    Path("results/run_meta.json").write_text(json.dumps(meta, indent=2))

    done = 0
    WAVE = 50
    with RESP.open("a") as rf:
        for wstart in range(0, len(tasks), WAVE):
            if estimated_cost() >= BUDGET_USD:
                print(f"BUDGET STOP at {done} calls")
                break
            wave = tasks[wstart : wstart + WAVE]
            with ThreadPoolExecutor(max_workers=args.workers) as ex:
                futs = {ex.submit(client.chat, prompt, s): (r, ph, s) for r, prompt, ph, s in wave}
                for fut in as_completed(futs):
                    r, ph, s = futs[fut]
                    try:
                        out = fut.result()
                    except Exception as e:
                        out = {"content": "", "usage": {}, "error": str(e), "cached": False}
                    rec = {"mol_id": r["mol_id"], "bin": r["bin"], "sample_idx": s,
                           "model": args.model, "prompt_hash": ph,
                           "content": out["content"], "cached": out.get("cached", False),
                           "usage": out.get("usage", {}), "error": out.get("error")}
                    rf.write(json.dumps(rec) + "\n")
                    rf.flush()
                    done += 1
            print(f"{done}/{len(tasks)} done, est cost ${estimated_cost():.2f}")

    # check every response
    targets = {r["mol_id"]: (mol_from_smiles(r["smiles"]), r["formula"], bool(int(r["has_stereo"]))) for r in rows}
    with CHECKS.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["mol_id", "bin", "sample_idx", "model", "prompt_hash", "cached",
                    "p0", "p1", "p2", "p3", "pass", "stereo", "fail_reason", "p3_stats"])
        seen = set()
        for ln in RESP.read_text().splitlines():
            rec = json.loads(ln)
            key = (rec["mol_id"], rec["sample_idx"], rec["model"])
            if key in seen:
                continue
            seen.add(key)
            t = targets[rec["mol_id"]]
            out = run_checks(rec["content"], *t)
            w.writerow([rec["mol_id"], rec["bin"], rec["sample_idx"], rec["model"],
                        rec["prompt_hash"], rec["cached"], out["p0"], out["p1"], out["p2"],
                        out["p3"], out["p0"] and out["p1"] and out["p2"] and out["p3"],
                        out["stereo"], out["fail_reason"], out["p3_stats"]])
    print(f"done. est cost ${estimated_cost():.2f}")


if __name__ == "__main__":
    main()
