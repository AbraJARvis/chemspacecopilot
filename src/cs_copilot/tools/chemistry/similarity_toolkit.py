#!/usr/bin/env python
# coding: utf-8
"""
Chemical similarity toolkit providing various similarity metrics and calculations.

This module extends the base chemistry toolkit with specialized similarity
calculations including Tanimoto, Dice, Tversky, and other similarity metrics.
"""

import logging
from typing import Dict, List, Tuple

from rdkit import DataStructs

from .base_chemistry import BaseChemistryToolkit, ChemistryError

logger = logging.getLogger(__name__)


class SimilarityError(ChemistryError):
    """Exception raised for similarity calculation errors."""

    pass


class ChemicalSimilarityToolkit(BaseChemistryToolkit):
    """
    Chemical similarity toolkit providing various similarity metrics.

    This class extends the base chemistry toolkit with specialized similarity
    calculations for molecular comparison and analysis.
    """

    def __init__(self):
        """Initialize the ChemicalSimilarityToolkit."""
        super().__init__("chemical_similarity")
        # Register all similarity tools
        self.register(self.calculate_tanimoto_similarity)
        self.register(self.calculate_dice_similarity)
        self.register(self.calculate_tversky_similarity)
        self.register(self.calculate_cosine_similarity)
        self.register(self.calculate_euclidean_distance)
        self.register(self.calculate_all_similarities)
        self.register(self.find_most_similar)

    def calculate_tanimoto_similarity(
        self, smiles1: str, smiles2: str, fp_type: str = "rdkit"
    ) -> float:
        """
        Calculate Tanimoto similarity between two molecules.

        Args:
            smiles1: SMILES string for first molecule
            smiles2: SMILES string for second molecule
            fp_type: Type of fingerprint ('rdkit', 'morgan', 'maccs')

        Returns:
            Tanimoto similarity score (0.0 to 1.0)

        Raises:
            SimilarityError: If calculation fails
        """
        try:
            # Generate fingerprints
            fp1 = self.generate_fingerprint(smiles1, fp_type)
            fp2 = self.generate_fingerprint(smiles2, fp_type)

            # Calculate Tanimoto similarity
            similarity = DataStructs.FingerprintSimilarity(
                fp1, fp2
            )  # TODO: change to in-house similarity

            logger.debug(f"Tanimoto similarity ({fp_type}): {similarity:.4f}")
            return similarity

        except Exception as e:
            logger.error(f"Error calculating Tanimoto similarity: {e}")
            raise SimilarityError(f"Failed to calculate Tanimoto similarity: {e}") from e

    def calculate_dice_similarity(
        self, smiles1: str, smiles2: str, fp_type: str = "rdkit"
    ) -> float:
        """
        Calculate Dice similarity between two molecules.

        Args:
            smiles1: SMILES string for first molecule
            smiles2: SMILES string for second molecule
            fp_type: Type of fingerprint ('rdkit', 'morgan', 'maccs')

        Returns:
            Dice similarity score (0.0 to 1.0)
        """
        try:
            fp1 = self.generate_fingerprint(smiles1, fp_type)
            fp2 = self.generate_fingerprint(smiles2, fp_type)

            similarity = DataStructs.DiceSimilarity(fp1, fp2)

            logger.debug(f"Dice similarity ({fp_type}): {similarity:.4f}")
            return similarity

        except Exception as e:
            logger.error(f"Error calculating Dice similarity: {e}")
            raise SimilarityError(f"Failed to calculate Dice similarity: {e}") from e

    def calculate_tversky_similarity(
        self,
        smiles1: str,
        smiles2: str,
        alpha: float = 0.5,
        beta: float = 0.5,
        fp_type: str = "rdkit",
    ) -> float:
        """
        Calculate Tversky similarity between two molecules.

        Args:
            smiles1: SMILES string for first molecule
            smiles2: SMILES string for second molecule
            alpha: Weight for first molecule
            beta: Weight for second molecule
            fp_type: Type of fingerprint ('rdkit', 'morgan', 'maccs')

        Returns:
            Tversky similarity score (0.0 to 1.0)
        """
        try:
            fp1 = self.generate_fingerprint(smiles1, fp_type)
            fp2 = self.generate_fingerprint(smiles2, fp_type)

            similarity = DataStructs.TverskySimilarity(fp1, fp2, alpha, beta)

            logger.debug(f"Tversky similarity ({fp_type}, α={alpha}, β={beta}): {similarity:.4f}")
            return similarity

        except Exception as e:
            logger.error(f"Error calculating Tversky similarity: {e}")
            raise SimilarityError(f"Failed to calculate Tversky similarity: {e}") from e

    def calculate_cosine_similarity(
        self, smiles1: str, smiles2: str, fp_type: str = "rdkit"
    ) -> float:
        """
        Calculate cosine similarity between two molecules.

        Args:
            smiles1: SMILES string for first molecule
            smiles2: SMILES string for second molecule
            fp_type: Type of fingerprint ('rdkit', 'morgan', 'maccs')

        Returns:
            Cosine similarity score (0.0 to 1.0)
        """
        try:
            fp1 = self.generate_fingerprint(smiles1, fp_type)
            fp2 = self.generate_fingerprint(smiles2, fp_type)

            similarity = DataStructs.CosineSimilarity(fp1, fp2)

            logger.debug(f"Cosine similarity ({fp_type}): {similarity:.4f}")
            return similarity

        except Exception as e:
            logger.error(f"Error calculating cosine similarity: {e}")
            raise SimilarityError(f"Failed to calculate cosine similarity: {e}") from e

    def calculate_euclidean_distance(
        self, smiles1: str, smiles2: str, fp_type: str = "rdkit", normalize: bool = True
    ) -> float:
        """
        Calculate Euclidean distance between two molecules.

        Note: Euclidean distance is a dissimilarity metric - smaller values
        indicate more similar molecules. Unlike similarity metrics, the range
        is not bounded to [0, 1] unless normalized.

        Args:
            smiles1: SMILES string for first molecule
            smiles2: SMILES string for second molecule
            fp_type: Type of fingerprint ('rdkit', 'morgan', 'maccs')
            normalize: If True, normalize by fingerprint length (default: True)

        Returns:
            Euclidean distance (lower values = more similar molecules)
        """
        try:
            fp1 = self.generate_fingerprint(smiles1, fp_type)
            fp2 = self.generate_fingerprint(smiles2, fp_type)

            # Calculate Euclidean distance
            distance = DataStructs.FingerprintSimilarity(
                fp1, fp2, metric=DataStructs.EuclideanSimilarity
            )

            # Note: RDKit's EuclideanSimilarity actually returns 1 - normalized_distance
            # So we need to convert it back to actual distance
            if normalize:
                # RDKit returns similarity, convert to normalized distance
                distance = 1.0 - distance
            else:
                # For unnormalized distance, we need to calculate manually
                import numpy as np

                # Convert fingerprints to numpy arrays
                arr1 = np.zeros(len(fp1))
                arr2 = np.zeros(len(fp2))
                DataStructs.ConvertToNumpyArray(fp1, arr1)
                DataStructs.ConvertToNumpyArray(fp2, arr2)
                distance = np.linalg.norm(arr1 - arr2)

            logger.debug(f"Euclidean distance ({fp_type}, normalized={normalize}): {distance:.4f}")
            return distance

        except Exception as e:
            logger.error(f"Error calculating Euclidean distance: {e}")
            raise SimilarityError(f"Failed to calculate Euclidean distance: {e}") from e

    def calculate_all_similarities(
        self, smiles1: str, smiles2: str, fp_type: str = "rdkit"
    ) -> Dict[str, float]:
        """
        Calculate all available similarity metrics between two molecules.

        Args:
            smiles1: SMILES string for first molecule
            smiles2: SMILES string for second molecule
            fp_type: Type of fingerprint ('rdkit', 'morgan', 'maccs')

        Returns:
            Dictionary containing all similarity scores and distance metrics
        """
        try:
            similarities = {
                "tanimoto": self.calculate_tanimoto_similarity(smiles1, smiles2, fp_type),
                "dice": self.calculate_dice_similarity(smiles1, smiles2, fp_type),
                "cosine": self.calculate_cosine_similarity(smiles1, smiles2, fp_type),
                "euclidean_distance": self.calculate_euclidean_distance(
                    smiles1, smiles2, fp_type, normalize=True
                ),
                "tversky_0.5_0.5": self.calculate_tversky_similarity(
                    smiles1, smiles2, 0.5, 0.5, fp_type
                ),
                "tversky_1.0_1.0": self.calculate_tversky_similarity(
                    smiles1, smiles2, 1.0, 1.0, fp_type
                ),
            }

            logger.debug(f"All similarities calculated for {fp_type} fingerprints")
            return similarities

        except Exception as e:
            logger.error(f"Error calculating all similarities: {e}")
            raise SimilarityError(f"Failed to calculate all similarities: {e}") from e

    def find_most_similar(
        self,
        query_smiles: str,
        reference_smiles: List[str],
        metric: str = "tanimoto",
        fp_type: str = "rdkit",
        top_k: int = 5,
    ) -> List[Tuple[str, float, int]]:
        """
        Find the most similar molecules to a query molecule.

        Args:
            query_smiles: Query molecule SMILES string
            reference_smiles: List of reference molecule SMILES strings
            metric: Similarity metric to use ('tanimoto', 'dice', 'cosine', 'euclidean')
            fp_type: Type of fingerprint ('rdkit', 'morgan', 'maccs')
            top_k: Number of top similar molecules to return

        Returns:
            List of tuples (smiles, similarity_score, index) sorted by similarity
        """
        try:
            similarities = []
            is_distance_metric = metric.lower() == "euclidean"

            for i, ref_smiles in enumerate(reference_smiles):
                try:
                    if metric.lower() == "tanimoto":
                        sim = self.calculate_tanimoto_similarity(query_smiles, ref_smiles, fp_type)
                    elif metric.lower() == "dice":
                        sim = self.calculate_dice_similarity(query_smiles, ref_smiles, fp_type)
                    elif metric.lower() == "cosine":
                        sim = self.calculate_cosine_similarity(query_smiles, ref_smiles, fp_type)
                    elif metric.lower() == "euclidean":
                        sim = self.calculate_euclidean_distance(
                            query_smiles, ref_smiles, fp_type, normalize=True
                        )
                    else:
                        raise ValueError(f"Unsupported similarity metric: {metric}")

                    similarities.append((ref_smiles, sim, i))

                except Exception as e:
                    logger.warning(f"Failed to calculate similarity for molecule {i}: {e}")
                    continue

            # Sort by similarity (descending for similarity, ascending for distance)
            similarities.sort(key=lambda x: x[1], reverse=not is_distance_metric)

            logger.debug(f"Found {len(similarities)} similar molecules, returning top {top_k}")
            return similarities[:top_k]

        except Exception as e:
            logger.error(f"Error finding most similar molecules: {e}")
            raise SimilarityError(f"Failed to find most similar molecules: {e}") from e

    # def calculate_similarity_matrix
