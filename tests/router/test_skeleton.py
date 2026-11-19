"""Smoke tests for the router package layout and the import rule in pyproject.toml."""

import importlib

import pytest

SUBPACKAGES = [
    "router.data",
    "router.core",
    "router.baselines",
    "router.sim",
    "router.experiments",
    "router.live",
    "router.proxy",
    "router.dashboard",
]


@pytest.mark.parametrize("name", SUBPACKAGES)
def test_subpackages_import(name):
    importlib.import_module(name)


def test_settings_paths_point_into_repo():
    from router import settings

    assert (settings.REPO_ROOT / "router" / "settings.py").is_file()
    assert settings.PHASE5_CSV.is_file()
    assert settings.BENCHMARK_DIR.is_dir()


def test_router_data_is_not_shadowed_by_repo_data_dir():
    # The repo root has a data/ directory on the path; router code must get its own package.
    import router.data

    assert router.data.__file__.endswith("router/data/__init__.py")


def test_pipeline_modules_import_by_bare_name():
    from components.llm_gateway import LLMGateway
    from components.test_runner import TestRunner

    assert LLMGateway and TestRunner
