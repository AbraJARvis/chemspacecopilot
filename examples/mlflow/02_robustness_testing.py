#!/usr/bin/env python
"""
MLflow-Enhanced Robustness Testing Example

This example demonstrates how to run robustness tests with MLflow tracking.
It shows how to:
- Configure MLflow for robustness testing
- Run tests with automatic tracking
- Query and analyze results
- Generate trend reports
"""

import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / "src"))
sys.path.insert(0, str(project_root / "tests" / "robustness"))

from dotenv import load_dotenv

load_dotenv()

# Configure MLflow
os.environ["MLFLOW_TRACKING_ENABLED"] = "true"
os.environ["MLFLOW_TRACKING_URI"] = "file:///tmp/cs_copilot_robustness_mlflow"
os.environ["MLFLOW_EXPERIMENT_NAME"] = "robustness_testing_example"


def example_1_basic_robustness_test():
    """Example 1: Run a basic robustness test with MLflow tracking."""
    print("\n" + "=" * 60)
    print("Example 1: Basic Robustness Test with MLflow")
    print("=" * 60)

    from robustness_minimal_example import RobustnessConfig, TestConfig
    from mlflow_runner import MLflowRobustnessRunner

    # Create minimal test configuration
    config = RobustnessConfig(
        n_variations=3,  # Use fewer variations for quick demo
        debug_mode=True,
        save_artifacts=True,
        model_provider="deepseek",
        model_id="deepseek-chat",
    )

    # Add a simple test
    config.tests = {
        "example_test": TestConfig(
            name="example_test",
            enabled=True,
            prompt_key="chembl_download",
            description="Example robustness test",
        )
    }

    # Create MLflow-enhanced runner
    runner = MLflowRobustnessRunner(
        config, experiment_name="robustness_testing_example", enable_mlflow=True
    )

    print(f"✅ Created MLflow runner")
    print(f"   Experiment: {runner.experiment_name}")
    print(f"   Output directory: {runner.output_dir}")
    print("\n⚠️  Note: This example requires a valid API key to run actual tests")


def example_2_query_test_results():
    """Example 2: Query and analyze robustness test results from MLflow."""
    print("\n" + "=" * 60)
    print("Example 2: Query Test Results from MLflow")
    print("=" * 60)

    try:
        from mlflow_reporter import MLflowRobustnessReporter

        reporter = MLflowRobustnessReporter(experiment_name="robustness_testing")

        # Get recent test runs
        print("\n📊 Fetching recent test runs...")
        recent_runs = reporter.get_recent_runs(days=7, max_results=10)

        if recent_runs:
            print(f"Found {len(recent_runs)} recent runs:")
            for i, run in enumerate(recent_runs[:3], 1):
                print(f"\nRun {i}:")
                print(f"  Run ID: {run['run_id']}")
                print(f"  Start time: {run['start_time']}")
                if "robustness_score" in run["metrics"]:
                    print(f"  Robustness score: {run['metrics']['robustness_score']:.3f}")
        else:
            print("No test runs found. Run some robustness tests first!")

        # Get test history
        print("\n📈 Fetching test history...")
        history = reporter.get_test_history("chembl_download", limit=10)

        if not history.empty:
            print(f"\nTest history for 'chembl_download' ({len(history)} runs):")
            print(history[["start_time", "robustness_score", "pass_rate"]].head())
        else:
            print("No history available for 'chembl_download' test")

    except Exception as e:
        print(f"❌ Error querying results: {e}")
        print("Make sure you have run some robustness tests with MLflow enabled first")


def example_3_generate_trend_report():
    """Example 3: Generate a trend analysis report."""
    print("\n" + "=" * 60)
    print("Example 3: Generate Trend Analysis Report")
    print("=" * 60)

    try:
        from mlflow_reporter import MLflowRobustnessReporter

        reporter = MLflowRobustnessReporter(experiment_name="robustness_testing")

        # Generate trend report
        print("\n📊 Generating trend report for last 30 days...")
        trend_report = reporter.generate_trend_report(days=30)

        if "error" in trend_report:
            print(f"⚠️  {trend_report['error']}")
            print("Run some robustness tests with --mlflow flag to generate trends")
            return

        print(f"\nTrend Report:")
        print(f"  Period: {trend_report['period_days']} days")
        print(f"  Total runs: {trend_report['total_runs']}")

        if "metric_trends" in trend_report:
            print("\n📈 Metric Trends:")
            for metric, stats in trend_report["metric_trends"].items():
                print(f"\n  {metric}:")
                print(f"    Mean: {stats['mean']:.3f}")
                print(f"    Std: {stats['std']:.3f}")
                print(f"    Min: {stats['min']:.3f}")
                print(f"    Max: {stats['max']:.3f}")
                print(f"    Trend: {stats['trend']}")

    except Exception as e:
        print(f"❌ Error generating report: {e}")


def example_4_find_failing_tests():
    """Example 4: Identify failing or low-performing tests."""
    print("\n" + "=" * 60)
    print("Example 4: Find Failing Tests")
    print("=" * 60)

    try:
        from mlflow_reporter import MLflowRobustnessReporter

        reporter = MLflowRobustnessReporter(experiment_name="robustness_testing")

        # Find tests below threshold
        threshold = 0.75
        print(f"\n🔍 Finding tests with robustness score < {threshold}...")

        failing_tests = reporter.get_failing_tests(threshold=threshold)

        if failing_tests:
            print(f"\n⚠️  Found {len(failing_tests)} failing test runs:")
            for i, test in enumerate(failing_tests[:5], 1):
                print(f"\n  {i}. {test['test_name']}")
                print(f"     Score: {test['robustness_score']:.3f}")
                print(f"     Run ID: {test['run_id']}")
                print(f"     Time: {test['start_time']}")
        else:
            print("✅ No failing tests found!")

    except Exception as e:
        print(f"❌ Error finding failing tests: {e}")


def example_5_compare_runs():
    """Example 5: Compare two test runs."""
    print("\n" + "=" * 60)
    print("Example 5: Compare Test Runs")
    print("=" * 60)

    try:
        from mlflow_reporter import MLflowRobustnessReporter

        reporter = MLflowRobustnessReporter(experiment_name="robustness_testing")

        # Get recent runs to compare
        recent_runs = reporter.get_recent_runs(days=30, max_results=2)

        if len(recent_runs) < 2:
            print("⚠️  Need at least 2 runs to compare")
            print("Run more robustness tests with --mlflow flag")
            return

        run1_id = recent_runs[0]["run_id"]
        run2_id = recent_runs[1]["run_id"]

        print(f"\n📊 Comparing runs:")
        print(f"  Run 1: {run1_id}")
        print(f"  Run 2: {run2_id}")

        comparison = reporter.compare_runs(run1_id, run2_id)

        print("\n📈 Metric Differences:")
        for metric, diff_data in list(comparison["metric_diffs"].items())[:5]:
            print(f"\n  {metric}:")
            print(f"    Run 1: {diff_data['run1']:.3f}")
            print(f"    Run 2: {diff_data['run2']:.3f}")
            print(f"    Diff: {diff_data['diff']:+.3f} ({diff_data['percent_change']:+.1f}%)")

    except Exception as e:
        print(f"❌ Error comparing runs: {e}")


def print_instructions():
    """Print instructions for running real robustness tests."""
    print("\n" + "=" * 60)
    print("How to Run Real Robustness Tests with MLflow")
    print("=" * 60)

    instructions = """
To run actual robustness tests with MLflow tracking:

1. Basic robustness test with MLflow:
   uv run python tests/robustness/robustness_minimal_example.py --mlflow

2. Run specific test:
   uv run python tests/robustness/robustness_minimal_example.py \\
       --mlflow --test chembl_download --n-variations 5

3. Custom experiment name:
   uv run python tests/robustness/robustness_minimal_example.py \\
       --mlflow --mlflow-experiment my_experiment

4. View results in MLflow UI:
   mlflow ui --backend-store-uri file:///tmp/mlflow
   # Then open http://localhost:5000

5. Query results programmatically:
   python examples/mlflow/02_robustness_testing.py
"""
    print(instructions)


def main():
    """Run all examples."""
    print("\n" + "=" * 60)
    print("MLflow-Enhanced Robustness Testing Examples")
    print("=" * 60)

    # Check if MLflow is available
    try:
        import mlflow

        print(f"✅ MLflow version: {mlflow.__version__}")
    except ImportError:
        print("❌ MLflow not installed. Please run: uv sync")
        return

    # Run examples
    example_1_basic_robustness_test()
    example_2_query_test_results()
    example_3_generate_trend_report()
    example_4_find_failing_tests()
    example_5_compare_runs()

    # Print instructions
    print_instructions()

    print("\n" + "=" * 60)
    print("Examples completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
