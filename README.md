# LLM 3D-geometry probe

Can a prompted LLM write valid 3D coordinates for a small molecule, and where
does it break along the way, in syntax, in chemistry, or in geometry? A small
preregistered probe, not a general answer.

## What was done

99 molecules from the ESOL set, 33 per heavy-atom bin (5-10, 11-20, 21-30),
seeded selection. One fixed prompt asks each molecule for an explicit-H XYZ
block. `nvidia/nemotron-3-ultra-550b-a55b` on NVIDIA NIM, k=8 samples at
temperature 1.0 (792 samples, about 0.84M output tokens). Each sample runs a
funnel of checks. P0 is a strict XYZ parse where the count line must match the
body, P1 checks the formula multiset, P2 infers connectivity from the
coordinates (rdDetermineBonds, first InChIKey block), and P3 applies
PoseBusters-style distance-geometry bounds at 25% tolerance. Pass means P0
through P3. Preregistration commit `89b3338`, made before any model call, with
one dated deviation (an early data-source swap).

## Headline result

The model fails at enumerating the right atoms for a SMILES string. Zero of
the 590 count-mismatch failures carry the correct element multiset. The count
on line 1 is right and the atoms listed under it are wrong. Fraction of
samples surviving each stage (A / B / C) goes P0 0.54 / 0.16 / 0.04,
P1 0.39 / 0.08 / 0.01, P2 0.06 / 0.02 / 0.00, P3 0.06 / 0.01 / 0.00. 75% of
failures are P0. Given a parse, the formula is usually right, and connectivity
is the next cliff. Conditional on P2, geometry passes 17 of 19 (small n).
pass@8 with the checker as verifier sits at 0.30 / 0.06 / 0.00. Every bin is
UNRELIABLE at the 0.80 threshold.

Two caveats on method. 147 rate-limited (HTTP 429) responses were deleted and
re-fetched, so all 792 rows are real model outputs. The run used non-reasoning
generation, because thinking mode spent all 8192 tokens on reasoning and
returned empty content. Reasoning mode and other models are untested.

## Checker validation (before the run)

- ETKDGv3 conformers round-tripped through XYZ pass at 1.00.
- The same conformers plus 0.3 A noise pass at 0.00; permuted element labels
  pass at 0.00.
- `posebusters` (mol config) agreed with our P3 on 19 of 19 evaluable
  responses.

## Limitations

- Small molecules from SMILES only; nothing here is about proteins or general
  structure prediction.
- One prompt, one model, temperature 1.0. Nothing is claimed about other
  prompts or models.
- Pass rate is a lower bound on capability. Strict format compliance is
  required, and the largest bin had too few parses to measure geometry.

## Reproduce

```
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python src/build_data.py        # needs data/raw/delaney-processed.csv
.venv/bin/python -m pytest tests/ -q
.venv/bin/python src/controls.py
.venv/bin/python src/run.py --model nvidia/nemotron-3-ultra-550b-a55b \
    --extra-json '{"chat_template_kwargs":{"enable_thinking":false}}'
.venv/bin/python src/recheck.py && .venv/bin/python src/analyze.py
.venv/bin/python src/figures.py && .venv/bin/python src/pb_crosscheck.py
```

The API key is read from `nvapi_key` (not committed). Responses are cached on
disk. Every number above comes from `results/`.

## Follow-up: verifier feedback

Does feeding checker diagnostics back to the model beat plain resampling, and
does the content of the feedback matter? Preregistered in `docs/prereg_v2.md`,
commit `b9741d80d2fabcb95bf30c6f2c768556ab242680`, before any v2 call. The 66
molecules of bins A and B ran one repair chain per arm, 4 calls each counting
the reused v1 round-0 response. Three arms. Independent resampling from the 8
cached v1 samples is the free baseline, a fixed binary message asks for a
correction, and structured feedback carries the checker's stage-specific
diagnostics. Same model and settings as v1. 375 calls, about 0.66M input and
0.25M output tokens, about $4.25.

At the 4-call budget (bin A / bin B, cumulative molecules passed) the numbers
are independent 0.19 / 0.03, binary 0.15 / 0.03, structured 0.21 / 0.00. Both
preregistered tests fail in both bins, H1 (structured minus independent at
least 0.10 with the paired cluster-bootstrap interval excluding 0) and H2
(structured minus binary under the same criterion). Checker feedback shows no
detectable gain over extra attempts at this budget. Power matters here. With
33 molecules per bin the intervals on arm differences run about +-0.12, so a
true +0.10 effect could hide inside the noise. The null is real but
underpowered.

The transition table shows what feedback did and did not change. In the
structured arm a P0 failure advanced to a later stage on the next round 23
times in 114 chances and passed once; binary advanced 13 times in 119.
Feedback moves molecules between stages, but regressions nearly cancel the
advances (16 in each arm). Fixing one stage often breaks an earlier one. P2
was nearly immovable in both arms (39/48 and 35/47 stayed put). In bin B
nothing passed under structured feedback. One hypothesis consistent with the
table is that the model can act on shallow enumeration feedback but not on the
deeper chemistry. That is an interpretation, not something this design tested.

Limits. One chain per molecule per arm, so chain-to-chain variance is not
measured. One model, one prompt, non-reasoning generation, small molecules
only. Bin C was excluded per the prereg (v1 pass rate 0.00).

```
.venv/bin/python src/repair.py --arm struct   # then --arm binary
.venv/bin/python src/analyze_v2.py && .venv/bin/python src/figures_v2.py
```

All v2 numbers come from `results/v2/`.
