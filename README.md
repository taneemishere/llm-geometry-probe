# LLM 3D-geometry probe

Can a prompted LLM write valid 3D coordinates for a small molecule, and where
does it break: syntax, chemistry, or geometry? Small preregistered probe, not
a general answer.

## What was done

99 molecules from the ESOL set, 33 per heavy-atom bin (5-10, 11-20, 21-30),
seeded selection. One fixed prompt asked each molecule for an explicit-H XYZ
block. `nvidia/nemotron-3-ultra-550b-a55b` via NVIDIA NIM, k=8 samples at
temperature 1.0 (792 samples, about 0.84M output tokens). Each sample runs a
funnel: P0 strict XYZ parse (count line must match the body), P1 formula
multiset, P2 connectivity inferred from coordinates (rdDetermineBonds, first
InChIKey block), P3 PoseBusters-style distance-geometry bounds at 25%
tolerance. Pass = P0-P3. Preregistration commit: `89b3338` (before any model
call; one dated deviation, an early data-source swap).

## Headline result

It fails at enumerating the right atoms for a SMILES string. Zero of the 590
count-mismatch failures have the correct element multiset: the model writes
the correct count on line 1 and then lists the wrong atoms. Fraction of
samples surviving each stage (A / B / C): P0 0.54 / 0.16 / 0.04,
P1 0.39 / 0.08 / 0.01, P2 0.06 / 0.02 / 0.00, P3 0.06 / 0.01 / 0.00. 75% of
failures are P0. Given a parse, the formula is usually right; connectivity is
the next cliff. Conditional on P2, geometry passes 17/19 (small n). pass@8
with the checker as verifier: 0.30 / 0.06 / 0.00. Every bin is UNRELIABLE at
the 0.80 threshold.

Two caveats on method: 147 rate-limited (HTTP 429) responses were deleted and
re-fetched, so all 792 rows are real model outputs; and the run used
non-reasoning generation because thinking mode spent all 8192 tokens on
reasoning and returned empty content. Reasoning mode and other models are
untested.

## Checker validation (before the run)

- Positive control (ETKDGv3 conformers round-tripped through XYZ): 1.00 pass.
- Negative controls: +0.3 A noise 0.00; permuted element labels 0.00.
- `posebusters` (mol config) agreed with our P3 on 19/19 evaluable responses.

## Limitations

- Small molecules from SMILES only; nothing about proteins or general
  structure prediction.
- One prompt, one model, temperature 1.0; nothing claimed about other
  prompts or models.
- Pass rate is a lower bound on capability: strict format compliance is
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

API key read from `nvapi_key` (not committed). Responses are cached on disk.
All numbers above come from `results/`.
