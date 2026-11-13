"""v2 repair loop: binary and structured feedback arms (prereg v2).

Round 0 reuses the cached v1 sample idx 0. Rounds 1-3 send feedback and retry.
Stop at first pass, max 4 calls per molecule incl. round 0. Call errors
(429, network) are never counted as results: the chain is retried on the next
pass; if it still fails the molecule is reported unfinished. Errors remain in
the raw conversations log.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from chem import mol_from_smiles
from checker import run_checks
from client import NIMClient, estimated_cost
from feedback import BINARY_MESSAGE, build_feedback, diagnose
from prompts import build_prompt

V2 = Path("results/v2")
USAGE_V2 = V2 / "usage_v2.jsonl"
CACHE_V2 = V2 / "cache"
BUDGET_USD = 6.0
ARMS = ["struct", "binary"]  # prereg order: (c) then (b)
MAX_ROUNDS = 4  # incl. round 0

_lock = threading.Lock()


def stage_of(out: dict) -> str:
    for s in ("p0", "p1", "p2", "p3"):
        if not out[s]:
            return s.upper()
    return "pass"


def load_v1_round0():
    resp = {}
    for ln in open("results/responses.jsonl"):
        r = json.loads(ln)
        if int(r["sample_idx"]) == 0:
            resp[r["mol_id"]] = r["content"]
    return resp


def run_chain(row, arm, client, v1_r0, conv_f):
    """One molecule's chain in one arm. Returns list of round records."""
    target = mol_from_smiles(row["smiles"])
    prompt = build_prompt(row["smiles"], row["formula"], int(row["n_atoms"]))
    recs = []
    prev = v1_r0[row["mol_id"]]
    conv = [{"role": "user", "content": prompt}]
    for rnd in range(MAX_ROUNDS):
        if rnd > 0:
            stage, det = diagnose(prev, target, row["formula"])
            fb = BINARY_MESSAGE if arm == "binary" else build_feedback(stage, det, target)
            conv = conv + [{"role": "assistant", "content": prev}, {"role": "user", "content": fb}]
            try:
                out_api = client.chat_messages(conv, f"{arm}|round{rnd}")
                prev = out_api["content"]
                api_err = None
            except Exception as e:
                api_err = str(e)
            with _lock:
                conv_f.write(json.dumps({"mol_id": row["mol_id"], "arm": arm, "round": rnd,
                                         "messages": conv, "response": prev if api_err is None else None,
                                         "error": api_err}) + "\n")
                conv_f.flush()
            if api_err is not None:
                return recs, False
        out = run_checks(prev, target, row["formula"], bool(int(row["has_stereo"])))
        recs.append({"mol_id": row["mol_id"], "bin": row["bin"], "arm": arm, "round": rnd,
                     "stage": stage_of(out), "p0": out["p0"], "p1": out["p1"], "p2": out["p2"],
                     "p3": out["p3"], "pass": out["p0"] and out["p1"] and out["p2"] and out["p3"],
                     "fail_reason": out["fail_reason"]})
        if recs[-1]["pass"]:
            return recs, True
    return recs, True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="nvidia/nemotron-3-ultra-550b-a55b")
    ap.add_argument("--workers", type=int, default=5)
    ap.add_argument("--max-mols", type=int, default=0)
    ap.add_argument("--arm", default="", help="override: run one arm only")
    args = ap.parse_args()

    V2.mkdir(parents=True, exist_ok=True)
    api_key = Path("nvapi_key").read_text().strip()
    client = NIMClient(api_key, args.model,
                       extra={"chat_template_kwargs": {"enable_thinking": False}},
                       cache_dir=CACHE_V2, usage_log=USAGE_V2)
    rows = [r for r in csv.DictReader(open("data/molecules.csv")) if r["bin"] in ("A", "B")]
    if args.max_mols:
        rows = rows[: args.max_mols]
    v1_r0 = load_v1_round0()

    meta = {"model": args.model, "arms": ARMS, "max_rounds": MAX_ROUNDS,
            "temperature": 1.0, "top_p": 1.0, "max_tokens": 8192,
            "extra": {"chat_template_kwargs": {"enable_thinking": False}},
            "price_in_per_mtok": 3.0, "price_out_per_mtok": 9.0, "budget_usd": BUDGET_USD,
            "started": time.strftime("%Y-%m-%dT%H:%M:%S")}
    (V2 / "run_meta_v2.json").write_text(json.dumps(meta, indent=2))

    ck_path = V2 / "checks_v2.csv"
    write_header = not ck_path.exists()
    ck_f = ck_path.open("a", newline="")
    w = csv.writer(ck_f)
    if write_header:
        w.writerow(["mol_id", "bin", "arm", "round", "stage", "p0", "p1", "p2", "p3",
                    "pass", "fail_reason"])

    arms = [args.arm] if args.arm else ARMS
    for arm in arms:
        conv_f = (V2 / f"conversations_{arm}.jsonl").open("a")
        done_mols = set()
        if ck_path.exists():
            for r in csv.DictReader(ck_path.open()):
                if r["arm"] == arm:
                    done_mols.add(r["mol_id"])  # chains write all rounds atomically
        pending = [r for r in rows if r["mol_id"] not in done_mols]
        print(f"arm {arm}: {len(pending)} chains pending")
        for pass_no in range(1 + 3):  # initial pass + up to 3 retry passes
            if not pending:
                break
            with ThreadPoolExecutor(max_workers=args.workers) as ex:
                futs = {}
                for r in pending:
                    if estimated_cost(USAGE_V2) >= BUDGET_USD:
                        print(f"BUDGET STOP in arm {arm}")
                        break
                    futs[ex.submit(run_chain, r, arm, client, v1_r0, conv_f)] = r
                retry = []
                for fut in as_completed(futs):
                    r = futs[fut]
                    try:
                        recs, finished = fut.result()
                    except Exception as e:
                        print("chain exception:", r["mol_id"], e)
                        retry.append(r)
                        continue
                    if finished:
                        for rec in recs:
                            w.writerow([rec.get("mol_id"), rec.get("bin"), rec.get("arm"),
                                        rec.get("round"), rec.get("stage"), rec.get("p0"),
                                        rec.get("p1"), rec.get("p2"), rec.get("p3"),
                                        rec.get("pass"), rec.get("fail_reason", "")])
                            ck_f.flush()
                    else:
                        retry.append(r)
            started = {r["mol_id"] for r in futs.values()}
            not_started = [r for r in pending if r["mol_id"] not in started]
            pending = retry + not_started
            if pass_no > 0:
                print(f"retry pass {pass_no}: {len(pending)} chains")
        conv_f.close()
        if pending:
            print(f"arm {arm}: {len(pending)} molecules unfinished "
                  f"(budget or persistent call errors)")
    ck_f.close()
    print(f"done. est v2 cost ${estimated_cost(USAGE_V2):.2f}")


if __name__ == "__main__":
    main()
