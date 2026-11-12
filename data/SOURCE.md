# Data provenance

- Dataset: Delaney ESOL aqueous solubility set, as distributed by
  DeepChem/MoleculeNet (`delaney-processed.csv`). Original data: Delaney,
  J. Chem. Inf. Model. 2004, 44(3), 1000-1005, doi:10.1021/ci034243x.
- URL: https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/delaney-processed.csv
- Retrieved: 2026-11-12. Raw file: `data/raw/delaney-processed.csv`
  (not tracked by git; re-download to reproduce).
- sha256 of raw file: `8c06a76f0c6487d29ab0f903e6a7a7139f189ab3c1178f159c8be8964602f189`
- License: publicly distributed benchmark file (MoleculeNet, MIT-licensed
  distribution; underlying measurements from the cited publication).
- Selection: `src/build_data.py`, seed 20261112, filters per
  `docs/prereg.md`. Output `data/molecules.csv` (99 rows, 33 per size bin).
- sha256 of molecules.csv: `6b98fb29b845b78b52a519f25ef40f0527df69d2582f17aab83e2e2e951c3d07`

Note: preregistered source was FreeSolv v0.52; switched before any model call
because FreeSolv has only 4 eligible molecules in the 21-30 heavy-atom bin.
See the Deviations appendix in `docs/prereg.md`.
