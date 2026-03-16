"""Configuration management for MLflow tracking."""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import yaml


@dataclass
class MLflowConfig:
    """Configuration for MLflow tracking.

    Attributes:
        enabled: Whether MLflow tracking is enabled
        tracking_uri: MLflow tracking server URI
        experiment_name: Default experiment name
        track_costs: Whether to track LLM API costs
        cost_per_1k_prompt_tokens: Cost per 1K prompt tokens in USD
        cost_per_1k_completion_tokens: Cost per 1K completion tokens in USD
        offline_mode: Whether to run in offline mode (no network)
    """

    enabled: bool = True
    tracking_uri: str = "file:///tmp/mlflow"
    experiment_name: str = "production_sessions"
    track_costs: bool = True
    cost_per_1k_prompt_tokens: float = 0.00027  # DeepSeek default
    cost_per_1k_completion_tokens: float = 0.0011  # DeepSeek default
    offline_mode: bool = False

    @classmethod
    def from_env(cls, config_path: Optional[Path] = None) -> "MLflowConfig":
        """Load configuration from environment variables and optional YAML file.

        Args:
            config_path: Optional path to YAML configuration file

        Returns:
            MLflowConfig instance
        """
        # Load from YAML if provided
        yaml_config = {}
        if config_path and config_path.exists():
            with open(config_path) as f:
                yaml_config = yaml.safe_load(f) or {}

        # Environment variables take precedence over YAML
        enabled_str = os.getenv("MLFLOW_TRACKING_ENABLED", yaml_config.get("enabled", "true"))
        if isinstance(enabled_str, bool):
            enabled = enabled_str
        else:
            enabled = str(enabled_str).lower() in ("true", "1", "yes", "on")

        tracking_uri = os.getenv(
            "MLFLOW_TRACKING_URI", yaml_config.get("tracking_uri", "file:///tmp/mlflow")
        )

        experiment_name = os.getenv(
            "MLFLOW_EXPERIMENT_NAME", yaml_config.get("experiment_name", "production_sessions")
        )

        track_costs_str = os.getenv("MLFLOW_TRACK_COSTS", yaml_config.get("track_costs", "true"))
        if isinstance(track_costs_str, bool):
            track_costs = track_costs_str
        else:
            track_costs = str(track_costs_str).lower() in ("true", "1", "yes", "on")

        cost_per_1k_prompt = float(
            os.getenv(
                "COST_PER_1K_TOKENS_PROMPT",
                yaml_config.get("cost_per_1k_prompt_tokens", 0.00027),
            )
        )

        cost_per_1k_completion = float(
            os.getenv(
                "COST_PER_1K_TOKENS_COMPLETION",
                yaml_config.get("cost_per_1k_completion_tokens", 0.0011),
            )
        )

        offline_mode_str = os.getenv(
            "MLFLOW_OFFLINE_MODE", yaml_config.get("offline_mode", "false")
        )
        if isinstance(offline_mode_str, bool):
            offline_mode = offline_mode_str
        else:
            offline_mode = str(offline_mode_str).lower() in ("true", "1", "yes", "on")

        return cls(
            enabled=enabled,
            tracking_uri=tracking_uri,
            experiment_name=experiment_name,
            track_costs=track_costs,
            cost_per_1k_prompt_tokens=cost_per_1k_prompt,
            cost_per_1k_completion_tokens=cost_per_1k_completion,
            offline_mode=offline_mode,
        )

    def is_enabled(self) -> bool:
        """Check if tracking is enabled."""
        return self.enabled and not self.offline_mode
