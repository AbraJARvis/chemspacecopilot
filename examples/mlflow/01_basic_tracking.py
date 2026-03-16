#!/usr/bin/env python
"""
Basic MLflow Tracking Example for Cs_copilot

This example demonstrates how to use MLflow tracking with Cs_copilot agents.
It shows:
- Session tracking
- Agent execution tracking
- Tool call tracking
- Metrics and parameters logging
"""

import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

from dotenv import load_dotenv

load_dotenv()

# Configure MLflow (use temporary directory for this example)
os.environ["MLFLOW_TRACKING_ENABLED"] = "true"
os.environ["MLFLOW_TRACKING_URI"] = "file:///tmp/cs_copilot_mlflow_example"
os.environ["MLFLOW_EXPERIMENT_NAME"] = "basic_tracking_example"

from cs_copilot.tracking import get_tracker


def example_1_simple_tracking():
    """Example 1: Simple session and agent tracking."""
    print("\n" + "=" * 60)
    print("Example 1: Simple Session and Agent Tracking")
    print("=" * 60)

    tracker = get_tracker()

    if not tracker.is_enabled():
        print("❌ MLflow tracking is not enabled. Check your configuration.")
        return

    # Track a session
    with tracker.track_session(
        session_id="example_session_001", user_id="demo_user", interface="script"
    ):
        print("✅ Session started")

        # Log session-level parameters
        tracker.log_params({"example_type": "basic", "version": "1.0"})

        # Track an agent execution
        with tracker.track_agent_run(
            agent_name="ChEMBL Downloader", prompt="Download compounds for EGFR", agent_type="chembl"
        ):
            print("✅ Agent execution started")

            # Simulate agent work and log metrics
            tracker.log_metrics(
                {
                    "compounds_found": 1234.0,
                    "execution_time": 5.2,
                    "api_calls": 3.0,
                }
            )

            # Track tool calls
            with tracker.track_tool_call("search_chembl", {"target": "EGFR", "limit": 1000}):
                print("✅ Tool call tracked")
                tracker.log_metrics({"search_results": 1234.0, "query_time": 2.1})

    print("✅ Session completed")
    print(f"\nView results in MLflow UI:")
    print(f"  mlflow ui --backend-store-uri {os.environ['MLFLOW_TRACKING_URI']}")


def example_2_nested_tracking():
    """Example 2: Multiple nested agent and tool calls."""
    print("\n" + "=" * 60)
    print("Example 2: Nested Tracking (Multi-Agent Workflow)")
    print("=" * 60)

    tracker = get_tracker()

    with tracker.track_session(session_id="example_session_002", interface="script"):
        print("✅ Session started")

        # First agent: Download data
        with tracker.track_agent_run("ChEMBL Downloader", "Download EGFR compounds"):
            tracker.log_metrics({"compounds_downloaded": 5000.0})

            with tracker.track_tool_call("download_bioactivity"):
                tracker.log_metrics({"download_size_mb": 2.5})

        # Second agent: Build GTM map
        with tracker.track_agent_run("GTM Optimization", "Build chemical space map"):
            tracker.log_metrics({"map_nodes": 256.0, "training_time": 45.0})

            with tracker.track_tool_call("train_gtm"):
                tracker.log_metrics({"optimization_iterations": 100.0})

        # Third agent: Analyze results
        with tracker.track_agent_run("GTM Analysis", "Analyze compound distribution"):
            tracker.log_metrics({"clusters_identified": 12.0})

    print("✅ Multi-agent workflow completed")


def example_3_cost_tracking():
    """Example 3: Track LLM API costs."""
    print("\n" + "=" * 60)
    print("Example 3: Cost Tracking")
    print("=" * 60)

    from cs_copilot.tracking.utils import calculate_cost

    tracker = get_tracker()

    with tracker.track_session(session_id="example_session_003", interface="script"):
        with tracker.track_agent_run("Custom Agent", "Process user query"):
            # Simulate LLM API usage
            prompt_tokens = 500
            completion_tokens = 1500

            # Calculate cost (DeepSeek pricing)
            cost = calculate_cost(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                cost_per_1k_prompt=0.00027,
                cost_per_1k_completion=0.0011,
            )

            # Log tokens and cost
            tracker.log_metrics(
                {
                    "prompt_tokens": float(prompt_tokens),
                    "completion_tokens": float(completion_tokens),
                    "total_tokens": float(prompt_tokens + completion_tokens),
                    "cost_usd": cost,
                }
            )

            print(f"✅ Tracked LLM usage:")
            print(f"   Prompt tokens: {prompt_tokens}")
            print(f"   Completion tokens: {completion_tokens}")
            print(f"   Total cost: ${cost:.4f}")


def example_4_artifacts():
    """Example 4: Log artifacts (files, plots, etc.)."""
    print("\n" + "=" * 60)
    print("Example 4: Logging Artifacts")
    print("=" * 60)

    import json
    import tempfile

    tracker = get_tracker()

    with tracker.track_session(session_id="example_session_004", interface="script"):
        with tracker.track_agent_run("Data Analyzer", "Analyze dataset"):
            # Create temporary files to log
            with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
                data = {"results": [1, 2, 3], "summary": "Analysis complete"}
                json.dump(data, f)
                temp_file = f.name

            # Log file as artifact
            tracker.log_artifact(temp_file, artifact_path="analysis")
            print(f"✅ Logged artifact: {temp_file}")

            # Log text directly
            tracker.log_text("This is a sample analysis report", "report.txt")
            print("✅ Logged text artifact")

            # Log dict as JSON
            tracker.log_dict({"metric": 0.95, "threshold": 0.8}, "metrics.json")
            print("✅ Logged JSON artifact")

            # Cleanup
            os.unlink(temp_file)


def example_5_querying_runs():
    """Example 5: Query and analyze tracked runs."""
    print("\n" + "=" * 60)
    print("Example 5: Querying MLflow Runs")
    print("=" * 60)

    try:
        import mlflow
        from mlflow.tracking import MlflowClient

        client = MlflowClient()

        # Get experiment
        experiment_name = os.environ.get("MLFLOW_EXPERIMENT_NAME", "basic_tracking_example")
        experiment = client.get_experiment_by_name(experiment_name)

        if not experiment:
            print(f"❌ Experiment '{experiment_name}' not found")
            return

        # Search recent runs
        runs = client.search_runs(
            experiment_ids=[experiment.experiment_id],
            max_results=10,
            order_by=["start_time DESC"],
        )

        print(f"\n📊 Found {len(runs)} runs in experiment '{experiment_name}'")

        for i, run in enumerate(runs[:5], 1):
            print(f"\nRun {i}:")
            print(f"  ID: {run.info.run_id}")
            print(f"  Status: {run.info.status}")

            # Print key metrics
            if run.data.metrics:
                print("  Metrics:")
                for key, value in list(run.data.metrics.items())[:3]:
                    print(f"    {key}: {value:.2f}")

            # Print tags
            if "session_id" in run.data.tags:
                print(f"  Session ID: {run.data.tags['session_id']}")

    except ImportError:
        print("❌ MLflow not installed. Cannot query runs.")


def main():
    """Run all examples."""
    print("\n" + "=" * 60)
    print("Cs_copilot MLflow Tracking Examples")
    print("=" * 60)

    # Check if MLflow is available
    try:
        import mlflow

        print(f"✅ MLflow version: {mlflow.__version__}")
    except ImportError:
        print("❌ MLflow not installed. Please run: uv sync")
        return

    # Run examples
    example_1_simple_tracking()
    example_2_nested_tracking()
    example_3_cost_tracking()
    example_4_artifacts()
    example_5_querying_runs()

    print("\n" + "=" * 60)
    print("All examples completed!")
    print("=" * 60)
    print("\n📊 To view results in MLflow UI, run:")
    print(f"  mlflow ui --backend-store-uri {os.environ['MLFLOW_TRACKING_URI']}")
    print("\nThen open: http://localhost:5000")


if __name__ == "__main__":
    main()
