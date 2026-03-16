#!/usr/bin/env python
# coding: utf-8
"""
Example demonstrating the new chemistry toolkit functionality.

This script shows how to use both the BaseChemistryToolkit and
ChemicalSimilarityToolkit for molecular analysis and similarity calculations.
"""

import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from cs_copilot.tools.chemistry import BaseChemistryToolkit, ChemicalSimilarityToolkit


def main():
    """Demonstrate chemistry toolkit functionality."""

    # Example SMILES strings
    aspirin = "CC(=O)OC1=CC=CC=C1C(=O)O"
    ibuprofen = "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O"
    caffeine = "CN1C=NC2=C1C(=O)N(C(=O)N2C)C"

    print("=== Base Chemistry Toolkit Demo ===\n")

    # Validate SMILES
    print("1. SMILES Validation:")
    for smiles in [aspirin, ibuprofen, caffeine, "invalid_smiles"]:
        is_valid = BaseChemistryToolkit.validate_smiles(smiles)
        print(f"   {smiles[:30]}... -> Valid: {is_valid}")

    # Get molecular properties
    print("\n2. Molecular Properties:")
    for name, smiles in [("Aspirin", aspirin), ("Ibuprofen", ibuprofen), ("Caffeine", caffeine)]:
        try:
            mw = BaseChemistryToolkit.get_molecular_weight(smiles)
            formula = BaseChemistryToolkit.get_molecular_formula(smiles)
            print(f"   {name}: MW={mw:.1f}, Formula={formula}")
        except Exception as e:
            print(f"   {name}: Error - {e}")

    # Get Lipinski descriptors
    print("\n3. Lipinski Descriptors (Aspirin):")
    try:
        lipinski = BaseChemistryToolkit.get_lipinski_descriptors(aspirin)
        for key, value in lipinski.items():
            print(f"   {key}: {value}")
    except Exception as e:
        print(f"   Error: {e}")

    print("\n=== Chemical Similarity Toolkit Demo ===\n")

    # Calculate various similarity metrics
    print("4. Similarity Metrics (Aspirin vs Ibuprofen):")
    try:
        similarities = ChemicalSimilarityToolkit.calculate_all_similarities(aspirin, ibuprofen)
        for metric, score in similarities.items():
            print(f"   {metric}: {score:.4f}")
    except Exception as e:
        print(f"   Error: {e}")

    # Find most similar molecule
    print("\n5. Most Similar Molecules (query: Aspirin):")
    reference_molecules = [ibuprofen, caffeine]
    try:
        most_similar = ChemicalSimilarityToolkit.find_most_similar(
            aspirin, reference_molecules, metric="tanimoto", top_k=2
        )
        for smiles, score, idx in most_similar:
            print(f"   Similarity: {score:.4f} -> {smiles[:30]}...")
    except Exception as e:
        print(f"   Error: {e}")

    # Calculate similarity matrix
    print("\n6. Similarity Matrix:")
    molecules = [aspirin, ibuprofen, caffeine]
    try:
        matrix = ChemicalSimilarityToolkit.calculate_similarity_matrix(molecules)
        print("   ", end="")
        for i, mol in enumerate(molecules):
            print(f"  Mol{i+1}", end="")
        print()
        for i, row in enumerate(matrix):
            print(f"   Mol{i+1}", end="")
            for val in row:
                print(f"  {val:.3f}", end="")
            print()
    except Exception as e:
        print(f"   Error: {e}")

    print("\n=== Demo Complete ===")


if __name__ == "__main__":
    main()
