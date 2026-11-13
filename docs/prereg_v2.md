# Preregistration v2: verifier-feedback repair loop

Date: 2026-11-12. Committed before any new model call for this follow-up.
The text below must not be edited; later changes go in the dated Deviations
appendix. v1 artifacts (prereg, prompt template, results) are frozen.

## Question

Does feeding the checker's diagnostics back to the model fix the failures
seen in v1, or does independent resampling do just as well? Does the content
of the feedback matter, or only the extra attempts?

The v1 finding to repair. `nvidia/nemotron-3-ultra-550b-a55b` (thinking
disabled, temperature 1.0, k=8) rarely produced valid XYZ. 0 of 590
count-mismatch failures carried the correct element multiset; per-bin pass@1
was A 0.06, B 0.01, C 0.00.

## Molecule set

The 66 molecules in bins A (5-10 heavy atoms) and B (11-20) of
`data/molecules.csv`. Bin C is excluded because its v1 pass rate was 0.00, so
repair cannot be measured there at this budget.

## Design

- Round 0 reuses the cached v1 response with sample index 0. No new call. A
  molecule that passes at round 0 stops.
- Rounds 1-3 run a multi-turn conversation, the v1 user prompt, the previous
  model response as the assistant turn, then a feedback user message. On each
  new round the previous exchange is appended in order. Same model and
  settings as v1, temperature 1.0, top_p 1.0, max_tokens 8192, stream false,
  `chat_template_kwargs={"enable_thinking": false}`.
- A molecule's chain stops at its first pass. Maximum 4 calls per molecule
  including round 0. One chain per molecule per arm.
- The checks are identical to v1 (P0 parse, P1 formula, P2 connectivity, P3
  geometry at 25% tolerance). The same checker code decides pass/fail in both
  the loop and the analysis.

## Arms

- (a) Independent resampling, the control. pass@k for k = 1..4 from the 8
  cached v1 samples, unbiased estimator per molecule. Costs no calls.
- (b) Binary feedback. After each failure send exactly:

  ```
  The structure failed validation. Please output a corrected XYZ block only.
  ```

- (c) Structured feedback. Deterministic diagnostics from the checker, keyed
  by first-failed stage. Verbatim formats:

  - P0: `Your previous response failed format validation: {reason}.` where
    {reason} is one of:
    - `the count line says {n} atoms but the body contains {m} atom lines`
    - `line {i} is malformed: '{line}'. Each atom line must be 'Element x y z'`
    - `line {i}: '{el}' is not a valid element symbol`
    - `line 1 is not a positive integer`
    - `the response must contain only the XYZ block; remove all prose`
  - P1: `Your previous response failed formula validation. Target formula: {F}.
    Your formula: {Fgot}. Differences: {el}: expected {e}, got {g}; ...`
    listing every element whose count differs, sorted by symbol.
  - P2: `Your previous response fails connectivity validation. SMILES inferred
    from your coordinates: {s_pred} ({nb} bonds). Target SMILES: {s_t}
    ({nb_t} bonds).` If bond determination fails entirely: `no consistent bond
    structure could be inferred from your coordinates.`
  - P3: `Your previous response fails geometry validation. Worst violations:`
    then up to 5 lines `atom pair ({i},{j}) {kind}: {d} A outside [{lo},{hi}]`
    with kind in {bond, angle, clash}, worst = largest relative bound error.
  - Every structured message ends with the line
    `Output a corrected XYZ block only, same format as before.`

- Leakage rule. Feedback may contain only information about the model's own
  output plus the SMILES and formula already in the prompt. No reference
  coordinates, ETKDG conformers, or target geometry ever appears. A test
  scans feedback strings for this.

## Order, budget, cache

- Arm (c) runs first, then (b). Tokens are logged per call; hard cap $6
  estimated at the run-metadata list prices ($3/M in, $9/M out). If the cap
  is reached mid-arm, that arm is reported as unfinished.
- Cache key is sha256(model | serialized conversation | arm | round). Cached
  responses are never re-requested. HTTP 429 and network errors are retried
  and never counted as model results; a call that still fails after retries
  ends that molecule's chain as `call_error` (unfinished), not a failure.

## Statistics and hypotheses

- Cumulative success is the fraction of molecules passing within the first b
  calls, b = 1..4, per bin, per arm, with 95% cluster-bootstrap intervals
  (10,000 resamples of whole molecules).
- H1. At 4 calls, structured success minus independent pass@4 >= 0.10 in bin
  A or bin B, and the paired molecule-level bootstrap 95% interval of the
  difference excludes 0.
- H2. The same criterion for structured minus binary at 4 calls.
- If H1 fails, report that checker feedback adds nothing beyond extra
  attempts at this budget. If H1 holds and H2 fails, report that attempts
  matter and content does not. Report whichever happens.
- Stage-transition table per arm. For each molecule failing at stage S in
  round r, the share that in round r+1 passes, advances to a later stage,
  stays at S, or regresses. Counts of molecules, not just rates. One chain
  per molecule per arm; variance across chains is not measured.

## What counts as a deviation

- Any change to arm definitions, feedback strings, stopping rules, sampling
  settings, thresholds, or the molecule set after this commit.
- Adding arms, models, bins, or metrics.
- 429 and network errors are never deviations; they are retried or recorded
  as call_error.

## Deviations

(None at preregistration time.)

- 2026-11-13 (during run). Transient DNS and 429 errors interrupted 9 chains
  mid-way. Instead of ending those chains as `call_error`, unfinished chains
  were retried whole; completed rounds replay from the on-disk cache at zero
  API cost, so only the missing rounds consume calls. All 66 chains per arm
  finished. No error call was ever counted as a result; errors remain in the
  raw conversations logs. This follows the "retried" clause above and changes
  no arm, threshold, or metric.
