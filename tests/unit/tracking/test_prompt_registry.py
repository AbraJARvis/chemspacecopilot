"""Unit tests for MLflow prompt registry integration."""

from cs_copilot.tracking.config import MLflowConfig
from cs_copilot.tracking.core import MLflowTracker


class DummyPrompt:
    def __init__(self, template: str, version: int = 1):
        self.template = template
        self.version = version


class DummyGenAIAllowMissing:
    def __init__(self, existing: DummyPrompt | None = None):
        self._existing = existing
        self.load_calls = []
        self.register_calls = []

    def load_prompt(self, name: str, allow_missing: bool = False):
        self.load_calls.append((name, allow_missing))
        return self._existing

    def register_prompt(self, name: str, template: str, commit_message=None, tags=None):
        self.register_calls.append(
            {"name": name, "template": template, "commit_message": commit_message, "tags": tags}
        )
        prompt = DummyPrompt(template=template, version=2)
        self._existing = prompt
        return prompt


class DummyGenAINoAllowMissing:
    def __init__(self, existing: DummyPrompt | None = None):
        self._existing = existing
        self.load_calls = []
        self.register_calls = []

    def load_prompt(self, name: str):
        self.load_calls.append(name)
        return self._existing

    def register_prompt(self, name: str, template: str, commit_message=None, tags=None):
        self.register_calls.append(
            {"name": name, "template": template, "commit_message": commit_message, "tags": tags}
        )
        prompt = DummyPrompt(template=template, version=3)
        self._existing = prompt
        return prompt


class DummyMlflow:
    def __init__(self, genai):
        self.genai = genai


def _make_tracker_with_mlflow(mlflow_obj) -> MLflowTracker:
    config = MLflowConfig(enabled=False)
    tracker = MLflowTracker(config)
    tracker.config.enabled = True
    tracker._initialized = True
    tracker._mlflow = mlflow_obj
    return tracker


def test_register_prompt_reuses_existing_template():
    """Do not create a new version when template is unchanged."""
    existing = DummyPrompt(template="Prompt A", version=7)
    genai = DummyGenAIAllowMissing(existing=existing)
    tracker = _make_tracker_with_mlflow(DummyMlflow(genai))

    prompt = tracker.register_prompt_version(
        name="cs_copilot.test",
        template="Prompt A",
        commit_message="update",
        tags={"component": "cs_copilot"},
    )

    assert prompt is existing
    assert genai.register_calls == []
    assert genai.load_calls == [("cs_copilot.test", True)]


def test_register_prompt_creates_new_version():
    """Create a new prompt version when template changes or is missing."""
    genai = DummyGenAIAllowMissing(existing=None)
    tracker = _make_tracker_with_mlflow(DummyMlflow(genai))

    prompt = tracker.register_prompt_version(
        name="cs_copilot.test",
        template="Prompt B",
        commit_message="update",
        tags={"component": "cs_copilot"},
    )

    assert prompt is not None
    assert len(genai.register_calls) == 1
    call = genai.register_calls[0]
    assert call["name"] == "cs_copilot.test"
    assert call["template"] == "Prompt B"
    assert call["commit_message"] == "update"
    assert call["tags"] == {"component": "cs_copilot"}


def test_register_prompt_without_allow_missing_support():
    """Fall back to load_prompt without allow_missing when unsupported."""
    genai = DummyGenAINoAllowMissing(existing=None)
    tracker = _make_tracker_with_mlflow(DummyMlflow(genai))

    prompt = tracker.register_prompt_version(
        name="cs_copilot.test",
        template="Prompt C",
        commit_message=None,
        tags=None,
    )

    assert prompt is not None
    assert genai.load_calls == ["cs_copilot.test"]
    assert len(genai.register_calls) == 1


def test_register_prompt_missing_genai():
    """Gracefully skip when MLflow genai registry is unavailable."""

    class NoGenAI:
        pass

    tracker = _make_tracker_with_mlflow(NoGenAI())
    prompt = tracker.register_prompt_version(
        name="cs_copilot.test",
        template="Prompt D",
        commit_message=None,
        tags=None,
    )
    assert prompt is None
