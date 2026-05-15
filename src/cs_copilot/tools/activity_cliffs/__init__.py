#!/usr/bin/env python
# coding: utf-8
"""Activity-cliff framework exposed to QSAR training workflows."""

from .toolkit import ActivityCliffToolkit
from .service import (
    ACTIVITY_CLIFF_ANNOTATION_PREFIX,
    DEFAULT_ACTIVITY_CLIFF_INDEX,
    ActivityCliffConfig,
    ActivityCliffIndexRegistry,
    build_activity_cliff_loop_comparison_plots,
    prepare_activity_cliff_context,
    split_activity_cliff_args,
    strip_activity_cliff_columns,
)
from .feedback_training import (
    attach_activity_cliff_variant_training,
    compact_split_result,
    hardest_split_r2,
    variant_comparison_rows,
    variant_summary,
)

__all__ = [
    "ACTIVITY_CLIFF_ANNOTATION_PREFIX",
    "DEFAULT_ACTIVITY_CLIFF_INDEX",
    "ActivityCliffConfig",
    "ActivityCliffIndexRegistry",
    "ActivityCliffToolkit",
    "build_activity_cliff_loop_comparison_plots",
    "prepare_activity_cliff_context",
    "split_activity_cliff_args",
    "strip_activity_cliff_columns",
    "attach_activity_cliff_variant_training",
    "compact_split_result",
    "hardest_split_r2",
    "variant_comparison_rows",
    "variant_summary",
]
