"""Analysis: funnel stats, Wilson + cluster-adjusted CIs, pass@k, verdicts."""
from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.special import comb

CHECKS = Path("results/checks.csv")
SUMMARY = Path("results/summary.json")
TABLE = Path("results/summary_table.csv")

K = 8
VERDICT_T = 0.80
BOOT = 10000
RNG = np.random.RandomState(20261112)


def wilson(p: float, n: float, z: float = 1.959964):
    if n <= 0:
        return (0.0, 1.0)
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = (z / denom) * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return max(0.0, center - half), min(1.0, center + half)


def icc_anova(groups: list[list[int]]) -> float:
    """One-way ANOVA intracluster correlation for binary data."""
    vals = [x for g in groups for x in g]
    if not vals:
        return 0.0
    grand = float(np.mean(vals))
    m = float(np.mean([len(g) for g in groups]))
    G = len(groups)
    if G < 2:
        return 0.0
    msb = sum(len(g) * (np.mean(g) - grand) ** 2 for g in groups) / (G - 1)
    msw = sum(sum((x - np.mean(g)) ** 2 for x in g) for g in groups) / max(1, len(vals) - G)
    if msw == 0 and msb == 0:
        return 0.0
    icc = (msb - msw) / (msb + (m - 1) * msw) if (msb + (m - 1) * msw) else 0.0
    return max(0.0, min(1.0, icc))


def pass_at_k(c: int, n: int, k: int) -> float:
    if n - c < k:
        return 1.0
    return 1.0 - comb(n - c, k) / comb(n, k)


def cluster_boot(vals: np.ndarray) -> tuple[float, float]:
    n = len(vals)
    if n == 0:
        return (0.0, 0.0)
    idx = RNG.randint(0, n, size=(BOOT, n))
    means = vals[idx].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def n_iid_for_bound(p: float, side: str, thresh: float = VERDICT_T) -> int:
    """Smallest n where Wilson(p,n) bound on `side` clears thresh."""
    for n in range(1, 20000):
        lo, hi = wilson(p, n)
        if side == "lower" and lo >= thresh:
            return n
        if side == "upper" and hi < thresh:
            return n
    return 20000


def main():
    rows = list(csv.DictReader(CHECKS.open()))
    for r in rows:
        for c in ("p0", "p1", "p2", "p3", "pass"):
            r[c] = r[c] == "True"
    bins = ["A", "B", "C"]
    out = {"n_rows": len(rows), "bins": {}, "overall": {}}

    def analyze(sub):
        by_mol = defaultdict(list)
        for r in sub:
            by_mol[r["mol_id"]].append(r)
        mols = list(by_mol.values())
        M = len(mols)
        n = len(sub)
        res = {"M": M, "n_samples": n}
        # funnel (cumulative survival over all samples)
        funnel = {}
        prev = None
        for stage in ("p0", "p1", "p2", "p3"):
            surv = [r[stage] for r in sub if prev is None or all(r[p] for p in prev)]
            funnel[stage] = {
                "frac_of_all": float(np.mean([r[stage] and all(r[p] for p in (prev or [])) for r in sub])),
                "cond": float(np.mean(surv)) if surv else 0.0,
            }
            prev = (prev or []) + [stage]
        res["funnel"] = funnel
        # pass rate + CIs
        passv = np.array([r["pass"] for r in sub], dtype=float)
        p_hat = float(passv.mean())
        lo_n, hi_n = wilson(p_hat, n)
        groups = [[int(x["pass"]) for x in g] for g in mols]
        icc = icc_anova(groups)
        deff = 1 + (K - 1) * icc
        n_eff = n / deff
        lo_c, hi_c = wilson(p_hat, n_eff)
        res["pass_rate"] = p_hat
        res["wilson_naive"] = [lo_n, hi_n]
        res["icc"] = icc
        res["deff"] = deff
        res["n_eff"] = n_eff
        res["wilson_cluster"] = [lo_c, hi_c]
        # verdict
        if lo_c >= VERDICT_T:
            verdict = "RELIABLE"
            res["verdict"] = verdict
        elif hi_c < VERDICT_T:
            verdict = "UNRELIABLE"
            res["verdict"] = verdict
        else:
            res["verdict"] = "ABSTAIN"
            side = "lower" if p_hat >= VERDICT_T else "upper"
            n_iid = n_iid_for_bound(p_hat, side)
            res["abstain_extra_mols"] = int(math.ceil(deff * n_iid / K) - M)
        # pass@k
        pak = {}
        for k in range(1, K + 1):
            per_mol = np.array([pass_at_k(sum(x["pass"] for x in g), len(g), k) for g in mols])
            lo, hi = cluster_boot(per_mol)
            pak[str(k)] = {"mean": float(per_mol.mean()), "lo": lo, "hi": hi}
        res["pass_at_k"] = pak
        # stereo summary
        st = [r["stereo"] for r in sub if r["stereo"] != "na"]
        res["stereo"] = {"n": len(st), "match": st.count("match"),
                         "mismatch": st.count("mismatch"), "unevaluable": st.count("unevaluable")}
        # fail reasons
        fr = defaultdict(int)
        for r in sub:
            if r["fail_reason"]:
                fr[r["fail_reason"].split(":")[0]] += 1
        res["fail_stages"] = dict(fr)
        return res

    for b in bins:
        out["bins"][b] = analyze([r for r in rows if r["bin"] == b])
    out["overall"] = analyze(rows)
    SUMMARY.write_text(json.dumps(out, indent=2))

    with TABLE.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["bin", "M", "n", "f_P0", "f_P1", "f_P2", "f_P3", "pass@1",
                    "wlo_naive", "whi_naive", "icc", "wlo_cl", "whi_cl",
                    "pass@8", "p8_lo", "p8_hi", "verdict", "extra_mols"])
        for b in bins + ["overall"]:
            d = out["bins"].get(b) if b in bins else out["overall"]
            w.writerow([b, d["M"], d["n_samples"],
                        *[round(d["funnel"][s]["frac_of_all"], 4) for s in ("p0", "p1", "p2", "p3")],
                        round(d["pass_rate"], 4), round(d["wilson_naive"][0], 4),
                        round(d["wilson_naive"][1], 4), round(d["icc"], 3),
                        round(d["wilson_cluster"][0], 4), round(d["wilson_cluster"][1], 4),
                        round(d["pass_at_k"]["8"]["mean"], 4), round(d["pass_at_k"]["8"]["lo"], 4),
                        round(d["pass_at_k"]["8"]["hi"], 4),
                        d.get("verdict", ""), d.get("abstain_extra_mols", "")])
    print(TABLE.read_text())


if __name__ == "__main__":
    main()
