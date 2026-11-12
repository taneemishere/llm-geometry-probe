"""Prompt builder: the single preregistered template."""

TEMPLATE = """You are given a small molecule.

SMILES: {smiles}
Molecular formula (all hydrogens explicit): {formula}

Task: write a 3D structure of this molecule in XYZ format with every hydrogen atom explicit.

Format requirements:
- Line 1: the total number of atoms ({n_atoms})
- Line 2: a comment line (may be empty)
- Lines 3 to {n_atoms_plus_2}: one line per atom formatted as `Element x y z`
- Coordinates in Angstroms as decimal numbers
- The geometry must be one chemically reasonable conformation consistent with the SMILES
- Output the XYZ block only. No prose, no explanation, no markdown."""


def build_prompt(smiles: str, formula: str, n_atoms: int) -> str:
    return TEMPLATE.format(smiles=smiles, formula=formula, n_atoms=n_atoms, n_atoms_plus_2=n_atoms + 2)
