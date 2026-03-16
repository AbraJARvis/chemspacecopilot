"""Decorators for automatic MLflow tracking."""

import functools
import inspect
import logging
from typing import Any, Callable, Optional

from .core import get_tracker
from .utils import count_tokens

logger = logging.getLogger(__name__)


def track_agent_run(
    agent_name: Optional[str] = None,
    agent_type: Optional[str] = None,
):
    """Decorator to track agent execution.

    Args:
        agent_name: Name of the agent (if None, uses function name)
        agent_type: Type of agent (optional)

    Returns:
        Decorated function

    Example:
        @track_agent_run(agent_name="ChEMBL Downloader")
        def run_agent(prompt: str):
            # Agent logic here
            return result
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            tracker = get_tracker()
            if not tracker.is_enabled():
                return func(*args, **kwargs)

            # Extract prompt from args/kwargs
            prompt = _extract_prompt(args, kwargs)

            # Use provided agent_name or derive from function
            name = agent_name or func.__name__

            with tracker.track_agent_run(name, prompt, agent_type):
                result = func(*args, **kwargs)

                # Log result metrics if possible
                _log_result_metrics(tracker, result)

                return result

        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            tracker = get_tracker()
            if not tracker.is_enabled():
                return await func(*args, **kwargs)

            # Extract prompt from args/kwargs
            prompt = _extract_prompt(args, kwargs)

            # Use provided agent_name or derive from function
            name = agent_name or func.__name__

            with tracker.track_agent_run(name, prompt, agent_type):
                result = await func(*args, **kwargs)

                # Log result metrics if possible
                _log_result_metrics(tracker, result)

                return result

        # Return appropriate wrapper based on function type
        if inspect.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper

    return decorator


def track_tool_call(tool_name: Optional[str] = None):
    """Decorator to track tool execution.

    Args:
        tool_name: Name of the tool (if None, uses function name)

    Returns:
        Decorated function

    Example:
        @track_tool_call(tool_name="search_chembl")
        def search_database(query: str):
            # Tool logic here
            return results
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            tracker = get_tracker()
            if not tracker.is_enabled():
                return func(*args, **kwargs)

            # Use provided tool_name or derive from function
            name = tool_name or func.__name__

            # Extract args for logging (skip 'self' for methods)
            log_args = _extract_tool_args(func, args, kwargs)

            success = False
            error = None

            with tracker.track_tool_call(name, log_args):
                try:
                    result = func(*args, **kwargs)
                    success = True

                    # Log result metrics
                    _log_tool_result_metrics(tracker, result)

                    return result
                except Exception as e:
                    error = str(e)
                    raise
                finally:
                    # Log success/failure
                    tracker.log_metrics({"success": 1.0 if success else 0.0})
                    if error:
                        tracker.log_params({"error": error[:500]})

        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            tracker = get_tracker()
            if not tracker.is_enabled():
                return await func(*args, **kwargs)

            # Use provided tool_name or derive from function
            name = tool_name or func.__name__

            # Extract args for logging (skip 'self' for methods)
            log_args = _extract_tool_args(func, args, kwargs)

            success = False
            error = None

            with tracker.track_tool_call(name, log_args):
                try:
                    result = await func(*args, **kwargs)
                    success = True

                    # Log result metrics
                    _log_tool_result_metrics(tracker, result)

                    return result
                except Exception as e:
                    error = str(e)
                    raise
                finally:
                    # Log success/failure
                    tracker.log_metrics({"success": 1.0 if success else 0.0})
                    if error:
                        tracker.log_params({"error": error[:500]})

        # Return appropriate wrapper based on function type
        if inspect.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper

    return decorator


def _extract_prompt(args: tuple, kwargs: dict) -> str:
    """Extract prompt from function arguments.

    Args:
        args: Positional arguments
        kwargs: Keyword arguments

    Returns:
        Extracted prompt string
    """
    # Common parameter names for prompts
    prompt_keys = ["prompt", "query", "message", "text", "input"]

    # Check kwargs first
    for key in prompt_keys:
        if key in kwargs:
            return str(kwargs[key])

    # Check first positional argument (after self)
    if len(args) > 0:
        # Skip 'self' if it's a method
        first_arg = args[0]
        if not hasattr(first_arg, "__dict__") or not hasattr(first_arg, "__class__"):
            return str(first_arg)
        elif len(args) > 1:
            return str(args[1])

    return ""


def _extract_tool_args(func: Callable, args: tuple, kwargs: dict) -> dict:
    """Extract tool arguments for logging.

    Args:
        func: Function being called
        args: Positional arguments
        kwargs: Keyword arguments

    Returns:
        Dictionary of arguments
    """
    # Get function signature
    try:
        sig = inspect.signature(func)
        bound_args = sig.bind(*args, **kwargs)
        bound_args.apply_defaults()

        # Skip 'self' and 'cls' parameters
        log_args = {}
        for name, value in bound_args.arguments.items():
            if name not in ("self", "cls"):
                # Only log serializable types
                if isinstance(value, (str, int, float, bool, type(None))):
                    log_args[name] = value
                elif isinstance(value, (list, tuple)):
                    log_args[name] = f"<{type(value).__name__} len={len(value)}>"
                elif isinstance(value, dict):
                    log_args[name] = f"<dict len={len(value)}>"
                else:
                    log_args[name] = f"<{type(value).__name__}>"

        return log_args
    except Exception as e:
        logger.debug(f"Failed to extract tool args: {e}")
        return {}


def _log_result_metrics(tracker, result: Any):
    """Log metrics about agent execution result.

    Args:
        tracker: MLflow tracker instance
        result: Agent execution result
    """
    try:
        metrics = {}

        # If result is a string, log token count
        if isinstance(result, str):
            metrics["output_tokens"] = float(count_tokens(result))
            metrics["output_length"] = float(len(result))

        # If result is dict-like, try to extract metrics
        elif hasattr(result, "__dict__"):
            if hasattr(result, "usage"):
                usage = result.usage
                if hasattr(usage, "prompt_tokens"):
                    metrics["prompt_tokens"] = float(usage.prompt_tokens)
                if hasattr(usage, "completion_tokens"):
                    metrics["completion_tokens"] = float(usage.completion_tokens)
                if hasattr(usage, "total_tokens"):
                    metrics["total_tokens"] = float(usage.total_tokens)

        if metrics:
            tracker.log_metrics(metrics)

    except Exception as e:
        logger.debug(f"Failed to log result metrics: {e}")


def _log_tool_result_metrics(tracker, result: Any):
    """Log metrics about tool execution result.

    Args:
        tracker: MLflow tracker instance
        result: Tool execution result
    """
    try:
        metrics = {}

        # Log result type and size
        if isinstance(result, (list, tuple)):
            metrics["result_count"] = float(len(result))
        elif isinstance(result, dict):
            metrics["result_keys"] = float(len(result))
        elif isinstance(result, str):
            metrics["result_length"] = float(len(result))

        # Try to extract DataFrame metrics
        try:
            import pandas as pd

            if isinstance(result, pd.DataFrame):
                metrics["rows_returned"] = float(len(result))
                metrics["columns_returned"] = float(len(result.columns))
        except ImportError:
            pass

        if metrics:
            tracker.log_metrics(metrics)

    except Exception as e:
        logger.debug(f"Failed to log tool result metrics: {e}")
