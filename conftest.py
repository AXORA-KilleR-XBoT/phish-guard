from pathlib import Path

import pytest

from phishguard import build_model, load, save_model

SAMPLE = Path(__file__).parent / "data" / "sample_sms.tsv"


@pytest.fixture(scope="session")
def sample_model():
    """A model fitted on the 20-message sample file (fast, and good enough for the tests)."""
    texts, labels = load(SAMPLE)
    return build_model().fit(texts, labels)


@pytest.fixture
def model_file(tmp_path, sample_model):
    """A saved model with its checksum file, in a fresh folder."""
    path = tmp_path / "model.joblib"
    save_model(sample_model, path)
    return path
