#!/usr/bin/env python
# coding: utf-8
"""
Data formatting utilities for cs_copilot.

This module contains functions for formatting DataFrames, lists, and other data structures
for display and processing. Migrated from fns/helper_functions.py.
"""

import base64
import io
import math
import re
from pathlib import Path
from typing import Any, List, Optional, Tuple

import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import Draw

# Disable RDKit logging to suppress SMILES parsing errors
RDLogger.DisableLog("rdApp.*")


def has_integer_sqrt(n: int) -> bool:
    """
    Check if a number has an integer square root.

    Args:
        n: Number to check

    Returns:
        bool: True if n has an integer square root
    """
    if n < 0:
        return False
    root = math.sqrt(n)
    return root.is_integer()


def gtm_grid_size(gtm) -> int:
    """
    Calculate the grid size (side length) of a GTM model.

    Args:
        gtm: GTM model with num_nodes attribute

    Returns:
        int: Side length of the grid

    Raises:
        AssertionError: If num_nodes doesn't have an integer square root
    """
    size = gtm.num_nodes
    assert has_integer_sqrt(
        size
    ), f"The resps array (len {size}) doesn't have an integer square root"
    side_size = int(math.sqrt(size))
    return side_size


def list_to_list_of_str(items: List[Any]) -> List[str]:
    """
    Convert a list of mixed types to a list of formatted strings.

    Args:
        items: List of values (floats, ints, or other types)

    Returns:
        List[str]: Formatted string representations
    """
    return [
        (
            f"{item:.1f}"
            if isinstance(item, float)
            else f"{item:d}" if isinstance(item, int) else f"{item}"
        )
        for item in items
    ]


def df_as_str(df: pd.DataFrame) -> str:
    """
    Convert a DataFrame to a space-separated string representation.

    Args:
        df: DataFrame to convert

    Returns:
        str: String representation with columns on first line and data rows below
    """
    lines = [" ".join(df.columns)]
    for i in range(len(df)):
        lines.append(" ".join(list_to_list_of_str(df.iloc[i].to_list())))
    return "\n".join(lines)


def value_counts_df(df_in: pd.DataFrame, col_in: str) -> pd.DataFrame:
    """
    Compute value counts for a given column of a DataFrame and return the result as a new DataFrame.

    Args:
        df_in (pd.DataFrame): Input DataFrame.
        col_in (str): Column name to compute value counts.

    Returns:
        pd.DataFrame: A DataFrame with the unique values and their counts.
    """
    df_out = pd.DataFrame(df_in[col_in].value_counts())
    df_out.index.name = col_in
    df_out.columns = ["count"]
    return df_out.reset_index()


def get_density_in_node(density_table: pd.DataFrame, node_id: int) -> float:
    """
    Get the density values for a specific node from the density table.

    Args:
        density_table (pd.DataFrame): DataFrame containing density values.
        node_id (int): ID of the node to get density values for (integer, starting from 1).

    Returns:
        float: Density value for the specified node.
    """
    return float(density_table[density_table["nodes"] == node_id].density.iloc[0])


def sort_df_by_density(df: pd.DataFrame) -> pd.DataFrame:
    """
    Sort a dataframe by density in a specific node.

    Args:
        df (pd.DataFrame): DataFrame to sort.

    Returns:
        pd.DataFrame: Sorted dataframe.
    """
    return df.sort_values(by="density", ascending=False)


def smiles_to_markdown(
    text: str,
    md_path: Optional[Path] = None,
    img_dir: Optional[Path] = None,
    inline_base64: bool = False,
    return_base64: bool = False,
) -> str:
    """
    Convert SMILES strings in text to markdown with molecule images.

    Args:
        text: Input text containing SMILES strings
        md_path: Optional path to save markdown file
        img_dir: Optional directory to save images
        inline_base64: If True, embed images as base64 data URLs
        return_base64: If True, return base64-encoded images

    Returns:
        str: Markdown formatted text with molecule images
    """
    # Pattern to match SMILES strings (simplified)
    # Look for strings in backticks that look like SMILES
    pattern = r"`([A-Za-z0-9@+\-\[\]\(\)=#$]+)`"

    def replace_smiles(match):
        smiles = match.group(1)
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return match.group(0)  # Return original if not valid SMILES

            # Generate image
            img = Draw.MolToImage(mol, size=(200, 200))

            if inline_base64 or return_base64:
                # Convert to base64
                buf = io.BytesIO()
                img.save(buf, format="PNG")
                b64 = base64.b64encode(buf.getvalue()).decode()
                return f"![{smiles}](data:image/png;base64,{b64})"
            elif img_dir:
                # Save to file
                img_dir.mkdir(parents=True, exist_ok=True)
                # Use hash of SMILES as filename
                img_name = f"mol_{hash(smiles) % 10000:04d}.png"
                img_path = img_dir / img_name
                img.save(img_path)
                return f"![{smiles}]({img_path})"
            else:
                return match.group(0)
        except Exception:
            return match.group(0)

    result = re.sub(pattern, replace_smiles, text)

    if md_path:
        md_path.parent.mkdir(parents=True, exist_ok=True)
        with open(md_path, "w") as f:
            f.write(result)

    return result


def smiles_to_png_bytes(smiles: str, size: Tuple[int, int] = (200, 200)) -> bytes:
    """
    Convert a SMILES string to PNG image bytes.

    Args:
        smiles: SMILES string
        size: Image size as (width, height) tuple

    Returns:
        bytes: PNG image data

    Raises:
        ValueError: If SMILES is invalid
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES string: {smiles}")

    img = Draw.MolToImage(mol, size=size)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()
