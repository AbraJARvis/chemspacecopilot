# ChemSpace Copilot Tracking Module

MLflow GenAI integration for comprehensive agent and LLM tracking in ChemSpace Copilot.

## Overview

This module provides automatic tracking of:
- **User Sessions**: Complete interaction tracking via Chainlit
- **Agent Executions**: Performance, tool usage, and LLM calls
- **Prompt Registry**: Registers agent prompts and versions via MLflow GenAI prompt registry
- **Tool Calls**: Individual tool execution metrics
- **LLM Costs**: Token usage and cost estimation
- **Robustness Tests**: Test runs with variation metrics

## Quick Start

### Installation

MLflow is included in project dependencies:

```bash
uv sync
```

### Configuration

Set environment variables in `.env`:

```bash
MLFLOW_TRACKING_ENABLED=true
MLFLOW_TRACKING_URI=file:///tmp/mlflow
MLFLOW_EXPERIMENT_NAME=production_sessions
MLFLOW_TRACK_COSTS=true
```

### Basic Usage

#### Automatic Tracking (Recommended)

Agents created through factories are automatically tracked:

```python
from cs_copilot.agents.teams import get_cs_copilot_agent_team

team = get_cs_copilot_agent_team(model, enable_mlflow_tracking=True)
team.run("Download compounds for EGFR")  # Automatically tracked
```

#### Manual Tracking

For custom scripts:

```python
from cs_copilot.tracking import get_tracker

tracker = get_tracker()

with tracker.track_session("my_session"):
    with tracker.track_agent_run("my_agent", "user prompt"):
        tracker.log_metrics({"accuracy": 0.95})
```

#### Using Decorators

```python
from cs_copilot.tracking import track_agent_run, track_tool_call

@track_agent_run(agent_name="Custom Agent")
def my_agent(prompt: str):
    return process(prompt)

@track_tool_call(tool_name="custom_tool")
def my_tool(arg: str):
    return compute(arg)
```

## Module Structure

```
tracking/
├── __init__.py           # Public API exports
├── config.py            # Configuration management
├── core.py              # MLflowTracker class
├── decorators.py        # @track_agent_run, @track_tool_call
├── utils.py             # Helper functions
└── streaming_buffer.py  # Streaming event buffer
```

## Core Components

### MLflowTracker

Main tracking interface with context managers:

```python
tracker = get_tracker()

# Track a session
with tracker.track_session(session_id, user_id, interface):
    # Track agent execution
    with tracker.track_agent_run(agent_name, prompt, agent_type):
        # Track tool call
        with tracker.track_tool_call(tool_name, args):
            result = execute_tool()
```

**Methods:**
- `track_session()` - Track user session
- `track_agent_run()` - Track agent execution
- `track_tool_call()` - Track tool invocation
- `log_metrics()` - Log metrics to current run
- `log_params()` - Log parameters to current run
- `log_artifact()` - Log file artifact
- `log_text()` - Log text content
- `log_dict()` - Log dictionary as JSON

### Configuration

Loads from environment variables or YAML:

```python
from cs_copilot.tracking.config import MLflowConfig

# Load from environment
config = MLflowConfig.from_env()

# Custom configuration
config = MLflowConfig(
    enabled=True,
    tracking_uri="sqlite:///mlflow.db",
    experiment_name="my_experiment"
)
```

**Configuration Options:**
- `enabled` - Enable/disable tracking
- `tracking_uri` - MLflow server URI
- `experiment_name` - Default experiment name
- `track_costs` - Enable cost tracking
- `cost_per_1k_prompt_tokens` - Prompt token cost (USD)
- `cost_per_1k_completion_tokens` - Completion token cost (USD)
- `offline_mode` - Run without network

### Decorators

Function-level tracking decorators:

```python
from cs_copilot.tracking.decorators import track_agent_run, track_tool_call

@track_agent_run(agent_name="MyAgent", agent_type="custom")
def run_agent(prompt: str):
    return process(prompt)

@track_tool_call(tool_name="my_tool")
def my_tool(arg1: str, arg2: int):
    return result
```

**Features:**
- Automatic argument extraction
- Error handling and logging
- Both sync and async support
- Result metrics extraction

### Utilities

Helper functions for common operations:

```python
from cs_copilot.tracking.utils import (
    count_tokens,
    calculate_cost,
    format_duration,
    Timer,
)

# Estimate tokens
tokens = count_tokens("Hello world")

# Calculate LLM cost
cost = calculate_cost(
    prompt_tokens=500,
    completion_tokens=1500,
    cost_per_1k_prompt=0.00027,
    cost_per_1k_completion=0.0011
)

# Time execution
with Timer() as timer:
    do_work()
duration = timer.get_duration()
```

## Run Hierarchy

Runs are organized hierarchically:

```
Session (root run)
└── Agent Execution (nested)
    └── Tool Call (nested)
```

For robustness tests:

```
Test Suite (root)
└── Individual Test (nested)
    └── Prompt Variation (nested)
        └── Agent Execution (nested)
            └── Tool Call (nested)
```

## Metrics

### Session Metrics
- `total_agents_used`
- `total_tool_calls`
- `session_duration_seconds`
- `total_tokens`
- `total_cost_usd`

### Agent Metrics
- `execution_time_seconds`
- `llm_calls`
- `tool_calls`
- `tokens_prompt`
- `tokens_completion`
- `tokens_total`

### Tool Metrics
- `execution_time_seconds`
- `success` (1.0 or 0.0)
- `rows_returned`
- `api_calls`

### Robustness Metrics
- `robustness_score`
- `data_similarity`
- `semantic_similarity`
- `process_consistency`
- `visual_similarity`

## Disabling Tracking

### Globally

```bash
export MLFLOW_TRACKING_ENABLED=false
```

### Per Agent

```python
agent = factory.create_agent(model, enable_mlflow_tracking=False)
```

### Per Team

```python
team = get_cs_copilot_agent_team(model, enable_mlflow_tracking=False)
```

## Viewing Results

### MLflow UI

```bash
mlflow ui --backend-store-uri file:///tmp/mlflow
# Open http://localhost:5000
```

### Python API

```python
import mlflow
from mlflow.tracking import MlflowClient

client = MlflowClient()
experiment = client.get_experiment_by_name("production_sessions")
runs = client.search_runs([experiment.experiment_id])

for run in runs:
    print(f"Run: {run.info.run_id}")
    print(f"Metrics: {run.data.metrics}")
```

## Testing

Run tracking tests:

```bash
# Unit tests
uv run pytest tests/unit/tracking/ -v

# Integration tests
uv run pytest tests/integration/test_agent_tracking.py -v

# Run examples
uv run python examples/mlflow/01_basic_tracking.py
```

## Troubleshooting

### Tracking Not Working

Check configuration:
```python
from cs_copilot.tracking import get_tracker

tracker = get_tracker()
print(f"Enabled: {tracker.is_enabled()}")
print(f"URI: {tracker.config.tracking_uri}")
```

### Permission Errors

Ensure directory is writable:
```bash
mkdir -p /tmp/mlflow
chmod 755 /tmp/mlflow
```

### Import Errors

Install MLflow:
```bash
uv sync
```

## Performance

- **Agent creation overhead**: < 1ms
- **Run tracking overhead**: < 5ms per execution
- **Total impact**: < 5% on execution time

## Examples

See `examples/mlflow/` directory:
- `01_basic_tracking.py` - Basic tracking examples
- `02_robustness_testing.py` - Robustness test tracking

## API Reference

See inline docstrings in module files:
- `config.py` - Configuration classes
- `core.py` - MLflowTracker class
- `decorators.py` - Tracking decorators
- `utils.py` - Helper functions

## License

Part of the ChemSpace Copilot project - see main LICENSE file.
