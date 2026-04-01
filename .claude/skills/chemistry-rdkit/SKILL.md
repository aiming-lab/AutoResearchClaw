---
name: chemistry-rdkit
description: Computational chemistry with RDKit for molecular analysis, descriptors, fingerprints, and substructure search. Use when working with SMILES, drug discovery, or cheminformatics tasks.
metadata:
  category: domain
  trigger-keywords: "molecule,SMILES,chemical,drug,rdkit,fingerprint,molecular,compound,reaction,cheminformatics"
  applicable-stages: "9,10,12"
  priority: "4"
  version: "1.0"
  author: researchclaw
  references: "adapted from K-Dense-AI/claude-scientific-skills"
---

## RDKit Cheminformatics Best Practice

### Molecular I/O
1. Create molecules from SMILES: `mol = Chem.MolFromSmiles('CCO')`
2. Always check for None: `MolFromSmiles` returns None on invalid input
3. Convert to canonical SMILES: `Chem.MolToSmiles(mol)`
4. Read SDF files: `suppl = Chem.SDMolSupplier('file.sdf')`
5. Read SMILES files: `suppl = Chem.SmilesMolSupplier('file.smi')`
6. Write molecules: `writer = Chem.SDWriter('output.sdf')`

### Molecular Descriptors
1. Molecular weight: `Descriptors.MolWt(mol)`
2. LogP (lipophilicity): `Descriptors.MolLogP(mol)`
3. TPSA (polar surface area): `Descriptors.TPSA(mol)`
4. H-bond donors/acceptors: `Descriptors.NumHDonors(mol)`, `Descriptors.NumHAcceptors(mol)`
5. Rotatable bonds: `Descriptors.NumRotatableBonds(mol)`
6. Lipinski Rule of 5: MW <= 500, LogP <= 5, HBD <= 5, HBA <= 10

### Fingerprints and Similarity
1. Morgan (circular) fingerprints: `AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)`
2. RDKit fingerprints: `Chem.RDKFingerprint(mol)`
3. MACCS keys: `MACCSkeys.GenMACCSKeys(mol)`
4. Tanimoto similarity: `DataStructs.TanimotoSimilarity(fp1, fp2)`
5. Use radius=2 (ECFP4 equivalent) as default for most applications
6. For virtual screening, Tanimoto > 0.7 suggests structural similarity

### Substructure Search
1. SMARTS patterns: `pattern = Chem.MolFromSmarts('[OH]')`
2. Check match: `mol.HasSubstructMatch(pattern)`
3. Get all matches: `mol.GetSubstructMatches(pattern)`
4. Common SMARTS: `[#6](=O)[OH]` (carboxylic acid), `[NH2]` (primary amine)
5. Filter compound libraries by functional group presence

### Property Calculation Patterns
1. Batch processing: iterate over SDMolSupplier, skip None entries
2. Use `Chem.Descriptors.descList` for all available descriptors
3. For ADMET filtering, calculate Lipinski, Veber, and PAINS filters
4. Generate 3D coordinates: `AllChem.EmbedMolecule(mol, AllChem.ETKDG())`
5. Minimize energy: `AllChem.MMFFOptimizeMolecule(mol)`

### Common Pitfalls
1. Always sanitize molecules (default behavior) — disable only when needed
2. Add hydrogens explicitly for 3D work: `Chem.AddHs(mol)`
3. Handle stereochemistry: use `Chem.AssignStereochemistry(mol)`
4. Large SDF files: use `ForwardSDMolSupplier` for memory efficiency
5. Kekulization errors usually indicate invalid SMILES input
