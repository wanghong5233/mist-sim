import copy
import tempfile
from pathlib import Path
import pytest
from mist_sim.config import load_config


@pytest.fixture
def tmp_path():
    with tempfile.TemporaryDirectory(prefix="mist-test-") as directory:
        yield Path(directory)


@pytest.fixture
def config():
    return load_config(Path(__file__).resolve().parents[1] / "configs" / "smoke.json")
