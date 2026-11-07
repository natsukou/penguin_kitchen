import json
from pathlib import Path

import pytest


@pytest.fixture
def config():
    return json.loads((Path(__file__).parents[1] / "configs/simulation.json").read_text())
