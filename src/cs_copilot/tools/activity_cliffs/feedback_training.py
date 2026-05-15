#!/usr/bin/env python
# coding: utf-8
"""Shared activity-cliff feedback-loop training summaries.

The training itself remains backend-specific.  This module owns the common
contract once a backend has trained baseline/filtered variants on fixed
holdouts.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional


def compact_split_result(result: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "strategy_label": result.get("strategy_label"),
        "strategy_family": result.get("strategy_family"),
        "backend_split_type": result.get("backend_split_type"),
        "seed": result.get("seed"),
        "metrics": result.get("metrics", {}).get("test") or {},
        "model_path": result.get("model_path") or result.get("best_model_path"),
        "best_model_path": result.get("best_model_path") or result.get("model_path"),
        "test_predictions_path": result.get("test_predictions_path"),
        "splits_path": result.get("splits_path"),
        "fixed_split_training_csv": result.get("fixed_split_training_csv"),
        "output_dir": result.get("output_dir"),
        "source_train_count": result.get("source_train_count"),
        "effective_train_count": result.get("effective_train_count"),
        "validation_count": result.get("validation_count"),
        "test_count": result.get("test_count"),
        "removed_from_train_count": result.get("removed_from_train_count", 0),
        "requested_exclusion_count": result.get("requested_exclusion_count", 0),
        "activity_cliff_variant_id": result.get("activity_cliff_variant_id"),
        "duration_seconds": result.get("duration_seconds"),
    }


def variant_summary(
    *,
    variant: Dict[str, Any],
    split_results: List[Dict[str, Any]],
) -> Dict[str, Any]:
    from cs_copilot.tools.prediction.qsar_training_policy import assess_protocol_results

    assessment = assess_protocol_results(split_results)
    return {
        "variant_id": variant.get("variant_id"),
        "loop_index": variant.get("loop_index"),
        "removed_tiers": variant.get("removed_tiers") or [],
        "removed_count": variant.get("removed_count", 0),
        "remaining_rows": variant.get("remaining_rows"),
        "filtered_training_csv": variant.get("filtered_training_csv"),
        "training_completed": bool(split_results),
        "split_results": [compact_split_result(item) for item in split_results],
        "validation_assessment": assessment,
    }


def hardest_split_r2(assessment: Dict[str, Any]) -> Optional[float]:
    metrics = assessment.get("governance", {}).get("hardest_split_metrics") or {}
    value = metrics.get("r2")
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def variant_comparison_rows(variant_summaries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for variant in variant_summaries:
        for split_result in variant.get("split_results") or []:
            metrics = split_result.get("metrics") or {}
            rows.append(
                {
                    "variant_id": variant.get("variant_id"),
                    "loop_index": variant.get("loop_index"),
                    "split": split_result.get("strategy_label"),
                    "removed_tiers": variant.get("removed_tiers") or [],
                    "removed_count_dataset": variant.get("removed_count", 0),
                    "requested_exclusion_count": split_result.get("requested_exclusion_count", 0),
                    "removed_from_train_count": split_result.get("removed_from_train_count", 0),
                    "source_train_count": split_result.get("source_train_count"),
                    "effective_train_count": split_result.get("effective_train_count"),
                    "validation_count": split_result.get("validation_count"),
                    "test_count": split_result.get("test_count"),
                    "r2": metrics.get("r2"),
                    "rmse": metrics.get("rmse"),
                    "mae": metrics.get("mae"),
                    "mse": metrics.get("mse"),
                    "n": metrics.get("n"),
                }
            )
    return rows


def format_activity_cliff_value(value: Any) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, float):
        return f"{value:.4g}"
    if isinstance(value, list):
        return ", ".join(str(item) for item in value) if value else "-"
    return str(value)


def activity_cliff_canonical_section_markdown(
    *,
    activity_cliffs: Dict[str, Any],
    variant_comparison_rows_: List[Dict[str, Any]],
    recommended_variant_text: Optional[str],
    neighborhood_policy_text: str,
    priority_counts_text: str,
) -> str:
    params = activity_cliffs.get("index_parameters") or {}
    mode = str(activity_cliffs.get("mode") or "standard")
    index_display = "SALI (Structure-Activity Landscape Index)"
    fingerprint = params.get("fingerprint")
    fingerprint_radius = format_activity_cliff_value(params.get("fingerprint_radius"))
    fingerprint_dimensions = params.get("fingerprint_dimensions")
    fingerprint_bits = params.get("fingerprint_bits")
    similarity_metric = params.get("similarity_metric")
    if fingerprint == "morgan_count":
        fingerprint_text = (
            "Morgan count fingerprints, "
            f"rayon {fingerprint_radius}, "
            f"{format_activity_cliff_value(fingerprint_dimensions)} dimensions"
        )
        similarity_text = "Tanimoto ponderee par les comptes"
    else:
        fingerprint_size_text = (
            f"{format_activity_cliff_value(fingerprint_bits)} bits"
            if fingerprint_bits is not None
            else f"{format_activity_cliff_value(fingerprint_dimensions)} dimensions"
        )
        fingerprint_text = (
            f"{format_activity_cliff_value(fingerprint)}, "
            f"rayon {fingerprint_radius}, {fingerprint_size_text}"
        )
        similarity_text = format_activity_cliff_value(similarity_metric)
    lines = [
        "Activity cliffs",
        "",
        f"L'analyse des falaises d'activite a ete realisee avec l'indice {index_display}.",
        "",
        "Parametres de voisinage :",
        f"- Empreinte : {fingerprint_text}",
        f"- Similarite : {similarity_text}",
        f"- Seuil de similarite : {format_activity_cliff_value(params.get('similarity_threshold'))}",
        f"- Nombre maximal de voisins : {format_activity_cliff_value(params.get('top_k_neighbors'))}",
        f"- Seuil de signalement : {format_activity_cliff_value(params.get('flag_threshold'))}",
        f"- Normalisation : {format_activity_cliff_value(params.get('normalization'))}",
        "",
        "Resultats d'annotation :",
        f"- Composes analyses : {format_activity_cliff_value(activity_cliffs.get('ranked_molecule_count'))}",
        f"- Composes signales : {format_activity_cliff_value(activity_cliffs.get('flagged_count'))}",
        f"- Repartition des priorites : {priority_counts_text}",
        f"- Mode : {mode}",
    ]
    if mode != "with_feedback_loops":
        lines.extend(
            [
                "",
                "Aucune boucle de retroaction n'a ete demandee. Aucun compose n'a ete retire ; "
                "l'annotation Activity Cliffs enrichit uniquement le rapport et les artefacts.",
            ]
        )
        return "\n".join(lines)

    lines.extend(
        [
            "",
            "Les boucles de retroaction ont ete evaluees avec des holdouts de validation et de test fixes "
            "et non filtres. Seul l'ensemble d'entrainement est filtre selon les paliers SALI.",
            "",
            "Variantes comparees :",
            "| Variante | Split | Paliers retires | Retires dataset | Retires train | Train effectif | Validation | Test | RMSE | MAE | R2 | MSE |",
            "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in variant_comparison_rows_:
        lines.append(
            "| "
            f"{format_activity_cliff_value(row.get('variant_id'))} | "
            f"{format_activity_cliff_value(row.get('split'))} | "
            f"{format_activity_cliff_value(row.get('removed_tiers'))} | "
            f"{format_activity_cliff_value(row.get('removed_count_dataset'))} | "
            f"{format_activity_cliff_value(row.get('removed_from_train_count'))} | "
            f"{format_activity_cliff_value(row.get('effective_train_count'))} | "
            f"{format_activity_cliff_value(row.get('validation_count'))} | "
            f"{format_activity_cliff_value(row.get('test_count'))} | "
            f"{format_activity_cliff_value(row.get('rmse'))} | "
            f"{format_activity_cliff_value(row.get('mae'))} | "
            f"{format_activity_cliff_value(row.get('r2'))} | "
            f"{format_activity_cliff_value(row.get('mse'))} |"
        )
    if recommended_variant_text:
        lines.extend(["", recommended_variant_text])
    lines.extend(["", f"Politique de voisinage canonique : {neighborhood_policy_text}"])
    return "\n".join(lines)


def activity_cliff_validation_metrics_markdown(
    *,
    variant_comparison_rows_: List[Dict[str, Any]],
    recommended_variant: Optional[str],
) -> Optional[str]:
    if not variant_comparison_rows_:
        return None
    lines = [
        "Metriques de validation par variante Activity Cliffs",
        "",
        "Toutes les variantes sont reportees avec les memes holdouts de validation et de test non filtres.",
        "",
        "| Variante | Split | Retires train | Train effectif | R2 | RMSE | MAE | MSE | n |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in variant_comparison_rows_:
        marker = " (recommandee)" if recommended_variant and row.get("variant_id") == recommended_variant else ""
        lines.append(
            "| "
            f"{format_activity_cliff_value(row.get('variant_id'))}{marker} | "
            f"{format_activity_cliff_value(row.get('split'))} | "
            f"{format_activity_cliff_value(row.get('removed_from_train_count'))} | "
            f"{format_activity_cliff_value(row.get('effective_train_count'))} | "
            f"{format_activity_cliff_value(row.get('r2'))} | "
            f"{format_activity_cliff_value(row.get('rmse'))} | "
            f"{format_activity_cliff_value(row.get('mae'))} | "
            f"{format_activity_cliff_value(row.get('mse'))} | "
            f"{format_activity_cliff_value(row.get('n'))} |"
        )
    if recommended_variant:
        lines.extend(["", f"Variante recommandee pour le statut final : {recommended_variant}."])
    return "\n".join(lines)


def activity_cliff_reporting_handoff(
    *,
    activity_cliffs: Dict[str, Any],
    variant_comparison_rows_: List[Dict[str, Any]],
    recommended_variant: Optional[str],
    recommendation_reason: Optional[str],
) -> Dict[str, Any]:
    params = activity_cliffs.get("index_parameters") or {}
    priority_counts = activity_cliffs.get("priority_counts") or {}
    fingerprint = params.get("fingerprint")
    if fingerprint == "morgan_count":
        fingerprint_policy = (
            "Morgan count fingerprints, "
            f"radius={params.get('fingerprint_radius')}, "
            f"dimensions={params.get('fingerprint_dimensions')}, "
            "similarity_metric=count_tanimoto"
        )
    else:
        fingerprint_policy = (
            f"Fingerprint={fingerprint}, radius={params.get('fingerprint_radius')}, "
            f"bits={params.get('fingerprint_bits')}, "
            f"similarity_metric={params.get('similarity_metric')}"
        )
    neighborhood_policy_text = (
        f"{fingerprint_policy}, "
        f"similarity_threshold={params.get('similarity_threshold')}, "
        f"top-k neighbors={params.get('top_k_neighbors')}, "
        f"flag_threshold={params.get('flag_threshold')}, "
        f"normalization={params.get('normalization')}."
    )
    priority_counts_text = (
        f"none={priority_counts.get('none', 0)}, low={priority_counts.get('low', 0)}, "
        f"medium={priority_counts.get('medium', 0)}, high={priority_counts.get('high', 0)}"
    )
    recommended_variant_text = (
        f"Variante recommandee : {recommended_variant}. {recommendation_reason}"
        if recommended_variant and recommendation_reason
        else None
    )
    canonical_section_markdown = activity_cliff_canonical_section_markdown(
        activity_cliffs=activity_cliffs,
        variant_comparison_rows_=variant_comparison_rows_,
        recommended_variant_text=recommended_variant_text,
        neighborhood_policy_text=neighborhood_policy_text,
        priority_counts_text=priority_counts_text,
    )
    validation_metrics_markdown = activity_cliff_validation_metrics_markdown(
        variant_comparison_rows_=variant_comparison_rows_,
        recommended_variant=recommended_variant,
    )
    return {
        "mode": activity_cliffs.get("mode"),
        "index_name": activity_cliffs.get("index_name"),
        "index_display_name": "SALI (Structure-Activity Landscape Index)",
        "neighborhood_policy_text": neighborhood_policy_text,
        "flagged_count": activity_cliffs.get("flagged_count"),
        "priority_counts_text": priority_counts_text,
        "variant_comparison_rows": variant_comparison_rows_,
        "recommended_variant": recommended_variant,
        "recommended_variant_text": recommended_variant_text,
        "holdout_policy_text": (
            "Validation and test holdouts are fixed and non-filtered; activity-cliff tiers are removed "
            "from the training split only."
        ),
        "canonical_section_markdown": canonical_section_markdown,
        "validation_metrics_markdown": validation_metrics_markdown,
    }


def attach_activity_cliff_variant_training(
    *,
    activity_cliffs: Dict[str, Any],
    baseline_split_results: List[Dict[str, Any]],
    variant_split_results: Dict[str, List[Dict[str, Any]]],
    backend_name: str,
) -> Dict[str, Any]:
    if not activity_cliffs.get("enabled"):
        return activity_cliffs
    if int(activity_cliffs.get("feedback_loops_requested") or 0) <= 0:
        activity_cliffs["feedback_evaluation_status"] = "not_requested"
        activity_cliffs["reporting_handoff"] = activity_cliff_reporting_handoff(
            activity_cliffs=activity_cliffs,
            variant_comparison_rows_=[],
            recommended_variant=None,
            recommendation_reason=None,
        )
        return activity_cliffs

    variants = list(activity_cliffs.get("variants") or [])
    by_variant = {"baseline_loop_0": baseline_split_results, **variant_split_results}
    variant_summaries: List[Dict[str, Any]] = []
    for variant in variants:
        variant_id = str(variant.get("variant_id"))
        split_results = by_variant.get(variant_id, [])
        summary = variant_summary(variant=variant, split_results=split_results)
        variant["training_result"] = summary
        variant_summaries.append(summary)

    comparable = [
        item
        for item in variant_summaries
        if item.get("training_completed")
        and hardest_split_r2(item.get("validation_assessment") or {}) is not None
    ]
    recommended_variant = None
    recommendation_reason = None
    if comparable:
        comparable.sort(
            key=lambda item: (
                hardest_split_r2(item.get("validation_assessment") or {}) or float("-inf"),
                -int(item.get("loop_index") or 0),
            ),
            reverse=True,
        )
        recommended_variant = comparable[0].get("variant_id")
        best_r2 = hardest_split_r2(comparable[0].get("validation_assessment") or {})
        recommendation_reason = (
            "Selection par meilleur R2 sur le split le plus difficile, avec holdouts fixes "
            f"(R2={best_r2:.3f})."
            if best_r2 is not None
            else None
        )

    comparison_rows = variant_comparison_rows(variant_summaries)
    activity_cliffs["variant_training"] = variant_summaries
    activity_cliffs["variant_comparison_table"] = comparison_rows
    activity_cliffs["recommended_variant"] = recommended_variant
    activity_cliffs["recommendation_reason"] = recommendation_reason
    activity_cliffs["feedback_evaluation_status"] = "evaluated" if comparable else "failed_no_comparable_variant"
    activity_cliffs["reporting_handoff"] = activity_cliff_reporting_handoff(
        activity_cliffs=activity_cliffs,
        variant_comparison_rows_=comparison_rows,
        recommended_variant=recommended_variant,
        recommendation_reason=recommendation_reason,
    )
    activity_cliffs["loop_training_policy"] = {
        "backend_name": backend_name,
        "baseline_trained": True,
        "train_filtering": "remove selected activity-cliff tiers from train split only",
        "holdout_policy": "validation and test indices remain fixed and non-filtered",
        "comparison_metric": "hardest_split_r2",
    }
    return activity_cliffs
