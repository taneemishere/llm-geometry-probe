"""v2 analysis: cumulative success vs call budget, stage transitions, paired bootstrap."""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from analyze import wilson  # reuse v1 interval

STAGES = ["P0", "P1", "P2", "P3"]
RNG = np.random.default_rng(20250201)


def load_checks():
    df = pd.read_csv("results/v2/checks_v2.csv")
    # per (mol, arm, round): pass = stage P3 with pass col; stage col records fail stage or 'pass'
    df["passed"] = df["stage"].eq("pass")
    df["fail_stage"] = df["stage"].where(df["stage"].isin(STAGES))
    return df


def independent_pass_at_k(v1_checks, k):
    """Unbiased pass@k per molecule from cached v1 samples (n=8, c correct)."""
    v1 = pd.read_csv(v1_checks)
    v1 = v1[v1["mol_id"].isin(pd.read_csv("results/v2/checks_v2.csv")["mol_id"].unique())]
    g = v1.groupby("mol_id")["pass"].agg(["sum", "count"])
    out = {}
    for m, r in g.iterrows():
        n, c = int(r["count"]), int(r["sum"])
        out[m] = 1.0 - np.prod([1 - c / (n - i) for i in range(min(k, n))])
    return out


def arm_success(df, arm, budget):
    """Cumulative success at call budget: molecule passed in any round < budget."""
    per = {}
    for m, grp in df[df.arm == arm].groupby("mol_id"):
        grp = grp.sort_values("round")
        per[m] = bool(grp.head(budget)["passed"].any())
    return per


def cluster_boot_ci(vals_by_mol):
    """Wilson-free molecule-level bootstrap 95% CI of the mean."""
    mols, vals = list(vals_by_mol.keys()), np.array(list(vals_by_mol.values()), float)
    n = len(mols)
    if n == 0:
        return np.nan, np.nan, np.nan
    idx = RNG.integers(0, n, (10000, n))
    means = vals[idx].mean(axis=1)
    return float(vals.mean()), *np.percentile(means, [2.5, 97.5]).tolist()


def paired_boot(diff_by_mol):
    vals = np.array(list(diff_by_mol.values()), float)
    n = len(vals)
    if n == 0:
        return np.nan, np.nan, np.nan
    idx = RNG.integers(0, n, (10000, n))
    means = vals[idx].mean(axis=1)
    return float(vals.mean()), *np.percentile(means, [2.5, 97.5]).tolist()


def transitions(df, arm):
    """Stage transitions round r -> r+1 among molecules that attempted r+1."""
    rows = []
    d = df[(df.arm == arm) & df.stage.isin(STAGES + ["pass"])].copy()
    for m, grp in d.groupby("mol_id"):
        grp = grp.sort_values("round").reset_index(drop=True)
        for i in range(len(grp) - 1):
            a, b = grp.loc[i], grp.loc[i + 1]
            if a["passed"]:
                continue  # chain would have stopped; no next round expected
            sa, sb = a["fail_stage"], ("P4" if b["passed"] else b["fail_stage"])
            if b["stage"] == "call_error":
                kind = "call_error"
            elif b["passed"]:
                kind = "pass"
            elif STAGES.index(sb) > STAGES.index(sa):
                kind = "advance"
            elif sb == sa:
                kind = "same"
            else:
                kind = "regress"
            rows.append({"mol_id": m, "from": sa, "to": b["stage"], "round": int(a["round"]),
                         "kind": kind})
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--v1-checks", default="results/checks.csv")
    args = ap.parse_args()

    df = load_checks()
    df.to_csv("results/v2/checks_v2.csv", index=False)

    mols = pd.read_csv("data/molecules.csv").set_index("mol_id")
    df["bin"] = df["mol_id"].map(mols["bin"])

    arms = [a for a in ["struct", "binary"] if a in set(df.arm)]
    indep = {k: independent_pass_at_k(args.v1_checks, k) for k in range(1, 5)}

    # --- cumulative success table ---
    rows = []
    for b in ["A", "B"]:
        sub = df[df["bin"] == b]
        mlist = sorted(sub["mol_id"].unique())
        for k in range(1, 5):
            iv = {m: indep[k][m] for m in mlist}
            mean, lo, hi = cluster_boot_ci(iv)
            rows.append(dict(bin=b, arm="independent", calls=k, n_mol=len(mlist),
                             success=round(mean, 4), lo=round(lo, 4), hi=round(hi, 4)))
        for arm in arms:
            for budget in range(1, 5):
                per = {m: arm_success(sub, arm, budget)[m] for m in mlist
                       if m in arm_success(sub, arm, budget)}
                mean, lo, hi = cluster_boot_ci(per)
                n_done = sum(1 for v in per.values() if v)
                rows.append(dict(bin=b, arm=arm, calls=budget, n_mol=len(per),
                                 success=round(mean, 4), lo=round(lo, 4), hi=round(hi, 4),
                                 n_pass=n_done))
    succ = pd.DataFrame(rows)
    succ.to_csv("results/v2/success_by_budget.csv", index=False)
    print(succ.to_string(index=False))

    # --- stage transitions ---
    all_tr = []
    for arm in arms:
        tr = transitions(df, arm)
        tr["arm"] = arm
        all_tr.append(tr)
        if len(tr):
            tab = tr.groupby(["from", "kind"]).size().unstack(fill_value=0)
            print(f"\n=== transitions {arm} ===\n{tab}")
    if all_tr:
        pd.concat(all_tr).to_csv("results/v2/transitions.csv", index=False)

    # --- H1/H2 paired tests at budget 4 ---
    verdicts = {}
    for b in ["A", "B"]:
        sub = df[df["bin"] == b]
        mlist = sorted(sub["mol_id"].unique())
        if "struct" in arms:
            s4 = arm_success(sub, "struct", 4)
            d1 = {m: float(s4[m]) - indep[4][m] for m in mlist if m in s4}
            mean, lo, hi = paired_boot(d1)
            verdicts[f"H1_bin{b}"] = dict(diff=round(mean, 4), lo=round(lo, 4), hi=round(hi, 4),
                                         passes=(mean >= 0.10 and lo > 0))
        if "struct" in arms and "binary" in arms:
            s4, b4 = arm_success(sub, "struct", 4), arm_success(sub, "binary", 4)
            d2 = {m: float(s4.get(m, np.nan)) - float(b4.get(m, np.nan))
                  for m in mlist if m in s4 and m in b4}
            mean, lo, hi = paired_boot(d2)
            verdicts[f"H2_bin{b}"] = dict(diff=round(mean, 4), lo=round(lo, 4), hi=round(hi, 4),
                                         passes=(mean >= 0.10 and lo > 0))
    meta = {
        "n_mol": int(df["mol_id"].nunique()),
        "rounds_per_arm": df.groupby("arm")["round"].max().to_dict() if len(df) else {},
        "call_errors": int((df.stage == "call_error").sum()),
        "verdicts": verdicts,
    }
    Path("results/v2/summary_v2.json").write_text(json.dumps(meta, indent=2))
    print("\n" + json.dumps(verdicts, indent=2))


if __name__ == "__main__":
    main()
