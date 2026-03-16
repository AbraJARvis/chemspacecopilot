"""Unit tests for the SynPlanner integration toolkit."""

from __future__ import annotations

import sys
import types

import pytest

from cs_copilot.tools.chemistry.synplanner_toolkit import SynPlannerToolkit


@pytest.fixture(autouse=True)
def fake_synplanner(monkeypatch):
    """Provide a lightweight stub of the external SynPlanner package."""

    module = types.ModuleType("synplanner")

    class FakeSynPlanner:
        def __init__(self, prefer_gpu: bool = False):
            self.prefer_gpu = prefer_gpu
            self.loaded = False
            self.invocations = []

        def load(self):
            self.loaded = True

        def plan(self, smiles: str, top_k: int = 3):
            self.invocations.append((smiles, top_k))
            return [
                {
                    "score": 0.42,
                    "steps": [
                        {
                            "description": "Break ester to acid + alcohol",
                            "reactants": ["acid chloride", "salicylic acid"],
                            "products": ["aspirin"],
                            "reagents": ["pyridine"],
                        },
                        {
                            "summary": "Assemble salicylic acid",
                            "precursors": ["phenol"],
                            "targets": ["salicylic acid"],
                        },
                    ],
                }
            ]

    def name_to_smiles(name: str) -> str:
        lookup = {
            "aspirin": "CC(=O)OC1=CC=CC=C1C(=O)O",
        }
        key = name.lower()
        if key not in lookup:
            raise KeyError(name)
        return lookup[key]

    def canonicalize_smiles(smiles: str) -> str:
        return f"canonical({smiles})"

    module.SynPlanner = FakeSynPlanner
    module.name_to_smiles = name_to_smiles
    module.canonicalize_smiles = canonicalize_smiles

    monkeypatch.setitem(sys.modules, "synplanner", module)

    yield module

    sys.modules.pop("synplanner", None)


@pytest.fixture
def fake_pubchem(monkeypatch):
    module = types.ModuleType("pubchempy")

    class FakeCompound:
        def __init__(self, smiles: str):
            self.canonical_smiles = smiles
            self.isomeric_smiles = smiles
            self.connectivity_smiles = smiles

    queries = []

    def get_compounds(value: str, namespace: str = "name"):
        queries.append((namespace, value))
        if namespace == "smiles":
            return [FakeCompound("CCO")]  # Valid SMILES for ethanol
        if namespace == "name":
            return [FakeCompound("CC(=O)OC1=CC=CC=C1C(=O)O")]  # Aspirin SMILES
        return []

    module.get_compounds = get_compounds
    module.queries = queries

    monkeypatch.setitem(sys.modules, "pubchempy", module)

    yield module

    sys.modules.pop("pubchempy", None)


@pytest.fixture
def empty_pubchem(monkeypatch):
    module = types.ModuleType("pubchempy")
    module.queries = []

    def get_compounds(value: str, namespace: str = "name"):
        module.queries.append((namespace, value))
        return []

    module.get_compounds = get_compounds

    monkeypatch.setitem(sys.modules, "pubchempy", module)

    yield module

    sys.modules.pop("pubchempy", None)


def test_identify_input_accepts_smiles():
    toolkit = SynPlannerToolkit()

    result = toolkit.identify_input("CCO")

    assert result["smiles"] == "CCO"
    assert result["source"] == "smiles"


def test_identify_input_rejects_empty():
    toolkit = SynPlannerToolkit()

    from cs_copilot.tools.chemistry.synplanner_toolkit import SynPlannerError

    with pytest.raises(SynPlannerError):
        toolkit.identify_input("")

    with pytest.raises(SynPlannerError):
        toolkit.identify_input("   ")


def test_convert_name_to_smiles_prefers_pubchem(fake_synplanner, fake_pubchem):
    toolkit = SynPlannerToolkit()

    smiles = toolkit.convert_name_to_smiles("aspirin", llm_smiles_guess="CCO")

    # PubChem returns a SMILES that gets canonicalized by RDKit
    assert smiles is not None
    assert len(smiles) > 0
    # Verify PubChem was queried with the SMILES namespace
    assert any(ns == "smiles" for ns, _ in fake_pubchem.queries)


def test_convert_name_to_smiles_uses_pubchem_name_lookup(fake_synplanner, fake_pubchem):
    toolkit = SynPlannerToolkit()

    smiles = toolkit.convert_name_to_smiles("aspirin")

    assert smiles is not None
    assert len(smiles) > 0
    # Verify PubChem was queried with the name namespace
    assert any(ns == "name" for ns, _ in fake_pubchem.queries)


def test_convert_name_to_smiles_raises_when_no_resolution(fake_synplanner, empty_pubchem):
    """When PubChem returns nothing and no LLM guess, should raise."""
    toolkit = SynPlannerToolkit()

    from cs_copilot.tools.chemistry.synplanner_toolkit import SynPlannerError

    with pytest.raises(SynPlannerError, match="Could not resolve"):
        toolkit.convert_name_to_smiles("totally_unknown_molecule_xyz")


def test_convert_name_to_smiles_with_llm_guess_asks_confirmation(fake_synplanner, empty_pubchem):
    """When PubChem fails but LLM guess is provided, should raise UserConfirmationRequiredError."""
    toolkit = SynPlannerToolkit()

    from cs_copilot.tools.chemistry.synplanner_toolkit import UserConfirmationRequiredError

    with pytest.raises(UserConfirmationRequiredError) as exc_info:
        toolkit.convert_name_to_smiles("mystery", llm_smiles_guess="CCN")

    assert exc_info.value.smiles is not None
    assert exc_info.value.molecule_name == "mystery"


def test_toolkit_registration():
    """Verify all expected tools are registered."""
    toolkit = SynPlannerToolkit()

    # Check that the toolkit has the expected tool methods
    assert hasattr(toolkit, "identify_input")
    assert hasattr(toolkit, "convert_name_to_smiles")
    assert hasattr(toolkit, "plan_synthesis")
    assert hasattr(toolkit, "describe_plan")
    assert hasattr(toolkit, "get_route_visualizations")


def test_toolkit_name():
    toolkit = SynPlannerToolkit()
    assert toolkit.name == "synplanner"


def test_identify_input_with_valid_smiles():
    toolkit = SynPlannerToolkit()

    # Test with a valid SMILES string
    result = toolkit.identify_input("c1ccccc1")  # benzene
    assert result["source"] == "smiles"
    assert result["smiles"] is not None


def test_ensure_sequence_static_method():
    assert SynPlannerToolkit._ensure_sequence(None) == []
    assert SynPlannerToolkit._ensure_sequence("single") == ["single"]
    assert SynPlannerToolkit._ensure_sequence(["a", "b"]) == ["a", "b"]
    assert SynPlannerToolkit._ensure_sequence(("x", "y")) == ["x", "y"]


def test_normalise_steps():
    toolkit = SynPlannerToolkit()

    steps = [
        {
            "description": "Step A",
            "reactants": ["R1"],
            "products": ["P1"],
            "reagents": ["Rg1"],
        },
        {
            "summary": "Step B",
            "precursors": ["Pre1"],
            "targets": ["T1"],
            "conditions": None,
        },
    ]

    normalised = toolkit._normalise_steps(steps)
    assert len(normalised) == 2
    assert normalised[0].index == 1
    assert normalised[0].description == "Step A"
    assert normalised[0].reactants == ["R1"]
    assert normalised[0].products == ["P1"]
    assert normalised[0].reagents == ["Rg1"]
    assert normalised[1].index == 2
    assert normalised[1].description == "Step B"
    assert normalised[1].reactants == ["Pre1"]
    assert normalised[1].products == ["T1"]
    assert normalised[1].reagents == []


def test_normalise_routes_empty():
    toolkit = SynPlannerToolkit()

    assert toolkit._normalise_routes(None) == []
    assert toolkit._normalise_routes([]) == []
    assert toolkit._normalise_routes("not a list") == []
