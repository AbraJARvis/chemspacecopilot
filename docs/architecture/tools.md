# Tools System

Tools are organized as **Toolkit classes** that inherit from `Toolkit` (Agno framework).

**Location**: `src/cs_copilot/tools/`

## Directory Structure

```
tools/
├── databases/          Database integrations
│   ├── base.py        BaseDatabaseToolkit (abstract)
│   ├── chembl.py      ChemblToolkit (ChEMBL REST API)
│   └── types.py       Query types and configurations
│
├── chemography/       Dimensionality reduction
│   ├── gtm.py         GTMToolkit (high-level interface)
│   └── gtm_operations.py  Core GTM implementations
│
├── chemistry/         Molecular operations
│   ├── similarity_toolkit.py      Similarity calculations
│   ├── autoencoder_toolkit.py     LSTM autoencoder operations
│   └── descriptors.py             Molecular descriptors
│
├── prediction/        Predictive modeling backends and toolkits
│   ├── backend.py     Backend contract for pluggable predictors
│   ├── chemprop_backend.py  Chemprop CLI adapter
│   └── chemprop_toolkit.py  Model registration + prediction workflows
│
├── io/                I/O and formatting
│   ├── pointer_pandas_tools.py   DataFrame ops + S3 integration
│   └── formatting.py              SMILES → images, markdown
│
└── constants.py       Configuration constants
```

Each toolkit registers methods as tools via `self.register(method)`. Agents call these tools via the Agno tool-calling mechanism.

## Adding a New Tool

1. Create a toolkit in `src/cs_copilot/tools/`:

```python
from agno import Toolkit

class MyNewToolkit(Toolkit):
    def __init__(self):
        super().__init__(name="my_new_toolkit")
        self.register(self.my_tool_function)

    def my_tool_function(self, param: str) -> str:
        """Tool description for LLM."""
        return f"Result: {param}"
```

2. Import and pass to the agent factory's `tools` parameter
