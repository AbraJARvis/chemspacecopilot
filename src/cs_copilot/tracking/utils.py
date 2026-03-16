"""Utility functions for MLflow tracking."""

import hashlib
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass(frozen=True)
class PromptSignature:
    """Normalized prompt metadata for versioning."""

    text: str
    sha256: str
    version: str
    num_lines: int
    num_chars: int


def build_prompt_signature(prompt: Optional[Any]) -> Optional[PromptSignature]:
    """Build a stable prompt signature from a prompt template.

    Args:
        prompt: Prompt template string or list of instruction strings.

    Returns:
        PromptSignature if prompt exists, else None.
    """
    if not prompt:
        return None

    if isinstance(prompt, list):
        normalized_parts = [str(item).strip() for item in prompt if item is not None]
        prompt_text = "\n".join(normalized_parts).strip()
    else:
        prompt_text = str(prompt).strip()

    if not prompt_text:
        return None

    sha256 = hashlib.sha256(prompt_text.encode("utf-8")).hexdigest()
    version = f"sha256:{sha256[:12]}"
    num_lines = prompt_text.count("\n") + 1
    num_chars = len(prompt_text)
    return PromptSignature(
        text=prompt_text,
        sha256=sha256,
        version=version,
        num_lines=num_lines,
        num_chars=num_chars,
    )


def count_tokens(text: str) -> int:
    """Estimate token count for a text string.

    Uses a simple heuristic: ~4 characters per token for English text.
    For more accurate counting, consider using tiktoken or similar libraries.

    Args:
        text: Input text

    Returns:
        Estimated token count
    """
    if not text:
        return 0
    # Simple heuristic: average of 4 characters per token
    return len(text) // 4


def calculate_cost(
    prompt_tokens: int,
    completion_tokens: int,
    cost_per_1k_prompt: float,
    cost_per_1k_completion: float,
) -> float:
    """Calculate total cost for LLM API call.

    Args:
        prompt_tokens: Number of prompt tokens
        completion_tokens: Number of completion tokens
        cost_per_1k_prompt: Cost per 1K prompt tokens in USD
        cost_per_1k_completion: Cost per 1K completion tokens in USD

    Returns:
        Total cost in USD
    """
    prompt_cost = (prompt_tokens / 1000) * cost_per_1k_prompt
    completion_cost = (completion_tokens / 1000) * cost_per_1k_completion
    return prompt_cost + completion_cost


def safe_log_value(value: Any) -> Optional[float]:
    """Safely convert a value to a loggable numeric type.

    Args:
        value: Value to convert

    Returns:
        Float value or None if conversion fails
    """
    if value is None:
        return None
    try:
        return float(value)
    except (ValueError, TypeError):
        return None


def format_duration(seconds: float) -> str:
    """Format duration in seconds to human-readable string.

    Args:
        seconds: Duration in seconds

    Returns:
        Formatted string (e.g., "1m 23s", "45s")
    """
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes = int(seconds // 60)
    remaining_seconds = seconds % 60
    return f"{minutes}m {remaining_seconds:.0f}s"


class Timer:
    """Context manager for timing code execution."""

    def __init__(self):
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None
        self.duration: Optional[float] = None

    def __enter__(self):
        self.start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time.time()
        if self.start_time:
            self.duration = self.end_time - self.start_time

    def get_duration(self) -> float:
        """Get duration in seconds."""
        return self.duration or 0.0


def sanitize_run_name(name: str, max_length: int = 100) -> str:
    """Sanitize run name for MLflow.

    Args:
        name: Raw run name
        max_length: Maximum length for run name

    Returns:
        Sanitized run name
    """
    # Replace invalid characters
    sanitized = name.replace("/", "_").replace("\\", "_").replace(":", "_")
    # Truncate if too long
    if len(sanitized) > max_length:
        sanitized = sanitized[:max_length]
    return sanitized


def merge_metrics(existing: Dict[str, float], new: Dict[str, float]) -> Dict[str, float]:
    """Merge two metric dictionaries, summing overlapping keys.

    Args:
        existing: Existing metrics
        new: New metrics to merge

    Returns:
        Merged metrics dictionary
    """
    result = existing.copy()
    for key, value in new.items():
        if key in result:
            result[key] += value
        else:
            result[key] = value
    return result
