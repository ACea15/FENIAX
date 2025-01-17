# content of conftest.py
# https://docs.pytest.org/en/latest/example/simple.html#control-skipping-of-tests-according-to-command-line-option
import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--runslow", action="store_true", default=False, help="run slow tests"
    )
    parser.addoption(
        "--runprivate", action="store_true", default=False, help="run proprietary tests"
    )


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: mark test as slow to run")
    config.addinivalue_line("markers", "private: mark test as non-opensource")


def pytest_collection_modifyitems(config, items):
    if config.getoption("--runslow") and config.getoption("--runprivate"):
        # --runslow given in cli: do not skip slow tests
        return
    skip_slow = pytest.mark.skip(reason="need --runslow option to run")
    skip_private = pytest.mark.skip(reason="need --runprivate option to run")    
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip_slow)
        if "private" in item.keywords:
            item.add_marker(skip_private)
