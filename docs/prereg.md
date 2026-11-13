# Preregistration: LLM 3D-geometry probe

Date: 2026-11-12. This file is committed before any model inference call. The
text below this line must not be edited. Later changes go in the dated
"Deviations" appendix at the bottom.

## Question

When a frontier LLM is prompted to write 3D coordinates (XYZ format) for a
small molecule, where does it break along the way, in syntax, in chemistry,
or in geometry? And how much can a deterministic checker rescue by
resampling?

Scope limits:

- Small molecules from SMILES, not protein-ligand placement.
- A prompted frontier model, no fine-tuning, no GPU.
- The probe measures where the text format breaks. It does not show that LLMs
  cannot do structure prediction.

## Data

- The source is the FreeSolv database, `database.txt`, release tag `v0.52` of
  https://github.com/MobleyLab/FreeSolv, license CC-BY-4.0. The raw file, its
  URL, and its sha256 are recorded in `data/SOURCE.md`.
- Selection parses each SMILES with RDKit and keeps molecules meeting all of
  the following.
  - Exactly one fragment (no `.` in SMILES).
  - Formal charge 0 on every atom.
  - Elements only in {B, C, N, O, F, Si, P, S, Cl, Br, I} plus implicit
    hydrogens, which excludes metals.
  - Embeds successfully under RDKit ETKDGv3 with a fixed seed.
  - 5-30 heavy atoms.
- Size strata by heavy-atom count are bin A = 5-10, bin B = 11-20,
  bin C = 21-30.
- The target is 33 molecules per bin, 99 total. If a bin holds fewer than 33
  eligible molecules, take all of them and record the shortfall.
- Selection is deterministic. Eligible molecule ids are sorted ascending,
  then sampled without replacement with numpy RandomState seeded 20261112.
  Selected ids are sorted in the output.
- `data/molecules.csv` columns: `mol_id` (FreeSolv id), `smiles` (canonical),
  `formula` (all hydrogens explicit, e.g. C2H6O), `n_heavy`, `n_atoms`
  (incl. H), `n_stereo` (specified tetrahedral centers), `has_stereo`, `bin`.

## Prompt template (fixed; one template for every molecule)

Exact template below. `{smiles}`, `{formula}`, `{n_atoms}`, and
`{n_atoms_plus_2}` are substituted per molecule. A single user message, no
system message.

```
You are given a small molecule.

SMILES: {smiles}
Molecular formula (all hydrogens explicit): {formula}

Task: write a 3D structure of this molecule in XYZ format with every hydrogen atom explicit.

Format requirements:
- Line 1: the total number of atoms ({n_atoms})
- Line 2: a comment line (may be empty)
- Lines 3 to {n_atoms_plus_2}: one line per atom formatted as `Element x y z`
- Coordinates in Angstroms as decimal numbers
- The geometry must be one chemically reasonable conformation consistent with the SMILES
- Output the XYZ block only. No prose, no explanation, no markdown.
```

## Model and sampling

- Provider is NVIDIA NIM on the OpenAI-compatible endpoint
  `https://integrate.api.nvidia.com/v1/chat/completions`.
- Primary model is `nvidia/nemotron-3-ultra-550b-a55b`.
- The model is a CLI flag. Another identifier may be substituted before the
  first call of a run only if the primary is unavailable, and that is
  recorded as a deviation.
- Sampling is k = 8 samples per molecule, temperature = 1.0, top_p = 1.0,
  max_tokens = 8192, stream = false, seed unset. If the endpoint offers a
  server-side reasoning disable switch it is used and logged in run metadata.
- Responses are cached on disk, keyed by sha256(model | prompt |
  sample_index). A cached response is never re-requested.
- Budget. Prompt and completion tokens are logged per call. The run stops
  when the estimated cost (tokens at the provider list price recorded in run
  metadata) reaches $10 or the planned call set is exhausted, whichever comes
  first.

## Checks (a funnel; each stage requires the previous)

### P0 parse

Parser spec, applied to raw `message.content`:

1. Strip leading and trailing whitespace. If the whole content is wrapped in
   a single markdown code fence (```` ``` ```` or ```` ```xyz ````), strip the
   two fence lines. Any other markdown or prose remains and fails below.
2. Drop trailing empty lines. The remaining text must be exactly n+2 lines,
   where n is the integer on line 1.
3. Line 1 is a positive integer n equal to the number of subsequent atom
   lines.
4. Line 2 is a comment line (any content, may be empty).
5. Lines 3..n+2 carry exactly 4 whitespace-separated fields, an element
   symbol then x, y, z. The element must be a valid periodic-table symbol.
   The coordinates must parse as finite floats, scientific notation allowed.
   No extra fields.

P0 passes iff all of the above hold.

### P1 formula

The element multiset of the parsed atoms must equal the target `formula`
multiset (hydrogens explicit). Pass/fail only.

### P2 graph (connectivity)

- Build an RDKit mol from the parsed coordinates (`MolFromXYZBlock`), then
  `rdDetermineBonds.DetermineBonds(mol, charge=0)`. Any exception, None, or a
  disconnected result fails the stage.
- Compare connectivity. `MolToInChIKey(pred_mol).split("-")[0]` must equal
  the first InChIKey block of the sanitized target molecule. First block
  only, the connectivity layer, so stereochemistry and bond-order nuance are
  ignored.

### P3 geometry

PoseBusters-style distance-geometry check (Buttenschoen et al., PoseBusters,
Chem. Sci. 2024, doi:10.1039/D3SC04185A; reference implementation the
`posebusters` package, `check_geometry`), run on the P2 molecule, tolerance
0.25 (the paper's Table 4 tolerances, bonds and angles within 0.75x-1.25x of
bounds):

- Bounds come from `GetMoleculeBoundsMatrix(mol, set15bounds=True,
  scaleVDW=True, doTriangleSmoothing=True, useMacrocycle14config=False)`.
- Conjugated terminal group symmetrization, SMARTS
  `[O,N;D1;$([O,N;D1]-[*]=[O,N;D1]),$([O,N;D1]=[*]-[O,N;D1])]~[*]`. For matched
  pairs grouped by element pair, lower bound = group min, upper = group max.
- Heavy-atom pairs only. Any pair containing H is skipped.
- Bond check. For every bonded pair, `(d-lb)/lb` (if d<lb) or `(d-ub)/ub`
  (if d>ub) must lie within +/-0.25; else fail.
- Angle check. The same bounds test on every 1-3 pair (two bonded atoms
  sharing a common bonded neighbor), tolerance 0.25.
- Clash check. For every pair that is neither bonded nor a 1-3 pair,
  `d >= 0.75 * lb`; any violation fails.
- P3 passes iff bond, angle, and clash checks all pass.
- If the `posebusters` package can run on a standalone molecule, its results
  are recorded as a secondary column and agreement with our implementation is
  reported. P3 still comes from our implementation.

### Stereo (separate column, not part of Pass)

Only for `has_stereo` molecules. On the P2 molecule, assign 3D
stereochemistry, map atoms to the target by substructure match, and compare
CIP labels at every target-specified stereocenter. Values are match,
mismatch, or unevaluable (no P2, or no atom map).

### Pass

Pass = P0 AND P1 AND P2 AND P3. Stereo is reported, never required.

## Controls (must pass before the real run)

Run on every dataset molecule through the identical parser and checker:

- Positive control. An ETKDGv3 conformer (seed = 0xC0FFEE + molecule index)
  is serialized to XYZ by our serializer, re-parsed, and checked. Expect at
  least 95% pass overall. Below that the checker is buggy and must be fixed
  before the run.
- Negative control A. The same conformers plus iid Gaussian noise sigma =
  0.3 A on every coordinate. Expect near-zero pass; reported.
- Negative control B. The same conformers with element labels permuted across
  rows (seeded). Expect near-zero pass; reported.

## Analysis and statistics

- Each (molecule, sample) row records every check outcome plus a failure
  reason.
- Funnel. Per bin, the fraction of all samples surviving each cumulative
  stage P0, P1, P2, P3. Stage-conditional rates are also reported.
- Pass@k for k = 1..8 uses the unbiased estimator
  `1 - C(8-c, k) / C(8, k)` per molecule with c passing samples; bin pass@k
  is the mean over molecules; 95% intervals come from a cluster bootstrap
  (10,000 resamples of whole molecules).
- Per-bin k=1 pass rate p_hat gets two 95% intervals:
  - a naive Wilson score interval on pooled n = 8*M samples;
  - a cluster-adjusted Wilson. Intracluster correlation ICC by one-way ANOVA
    on the pass indicator grouped by molecule, DEFF = 1 + 7*max(ICC,0),
    n_eff = n_pooled/DEFF, Wilson interval on (p_hat*n_eff, n_eff). Samples
    of one molecule are not independent; this is the reported interval.
- Verdict per bin. RELIABLE if the adjusted-Wilson lower bound >= 0.80,
  UNRELIABLE if the upper bound < 0.80, else ABSTAIN (insufficient evidence).
  For each ABSTAIN bin report the additional molecules needed,
  M_extra = ceil(DEFF * n_iid / 8) - M, where n_iid is the smallest n for
  which the plain Wilson(p_hat, n) interval clears 0.80 on the side p_hat
  falls.
- Negative and null results are reported as they are.

## Figures

1. The funnel, fraction surviving P0..P3 per size bin (headline).
2. pass@k curves per size bin with intervals.

Clean, labeled axes, no decoration.

## What counts as a deviation

- Any change to the prompt template, checks, tolerances, thresholds, sampling
  settings, or molecule selection after the preregistration commit.
- Adding models, molecule sets, or metrics.
- A changed prompt is a new run. v1 results stay in the report and the change
  is logged here with a date.
- Fixing a checker bug found by the controls before the first model call is
  logged but is not a deviation; after the first model call it is a
  deviation.

## Deviations

- 2026-11-12 (before any model call). Data source changed from FreeSolv
  `database.txt` v0.52 to the Delaney ESOL dataset
  (`delaney-processed.csv`, as distributed by DeepChem/MoleculeNet; original
  data Delaney 2004, J. Chem. Inf. Model. 44(3):1000-1005). Reason: FreeSolv
  v0.52 yields only 4 eligible molecules in bin C (21-30 heavy atoms), which
  makes the size-stratified analysis uninformative; ESOL has at least 100
  eligible molecules per bin. All filters, strata, seed, and the selection
  procedure are unchanged. `mol_id` values are `esol_<row index>` of the
  frozen file.
