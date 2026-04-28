from __future__ import annotations

import pandas as pd

from cs_copilot.tools.features.molecular_feature_toolkit import MolecularFeatureToolkit


def _sample_feature_input(tmp_path):
    input_csv = tmp_path / "molecules.csv"
    pd.DataFrame(
        {
            "SMILES": ["CCO", "CCN"],
            "Y": [1.0, 2.0],
        }
    ).to_csv(input_csv, index=False)
    return input_csv


def test_morgan_output_keeps_normalized_smiles_for_future_joins(tmp_path):
    toolkit = MolecularFeatureToolkit()
    input_csv = _sample_feature_input(tmp_path)
    output_csv = tmp_path / "morgan.csv"

    toolkit.smiles_to_morgan_fingerprints(
        input_csv=str(input_csv),
        smiles_column="SMILES",
        output_csv=str(output_csv),
        input_columns_to_keep=["Y"],
    )

    output_df = pd.read_csv(output_csv)
    assert "smiles" in output_df.columns
    assert "Y" in output_df.columns


def test_rdkit_output_keeps_normalized_smiles_for_future_joins(tmp_path):
    toolkit = MolecularFeatureToolkit()
    input_csv = _sample_feature_input(tmp_path)
    output_csv = tmp_path / "rdkit.csv"

    toolkit.smiles_to_rdkit_descriptors(
        input_csv=str(input_csv),
        smiles_column="SMILES",
        output_csv=str(output_csv),
        descriptor_set="basic",
        input_columns_to_keep=["Y"],
    )

    output_df = pd.read_csv(output_csv)
    assert "smiles" in output_df.columns
    assert "Y" in output_df.columns
