"""Regression tests for the plotting KDE version compatibility layer."""

import importlib.util
from pathlib import Path

import numpy as np


KDE_PATH = Path(__file__).parents[1] / "plotting" / "kde.py"


def load_kde_module():
    spec = importlib.util.spec_from_file_location("cosmosis_plotting_kde", KDE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_kde_import_and_evaluation_without_distutils():
    module = load_kde_module()
    rng = np.random.default_rng(1234)
    points = rng.normal(size=(2, 128))
    kde = module.KDE(points)

    axes, values = kde.grid_evaluate(12, [(-2.0, 2.0), (-2.0, 2.0)])
    evaluated = kde.normalize_and_evaluate([0.0, 0.0])

    assert len(axes) == 2
    assert values.shape == (12, 12)
    assert np.all(np.isfinite(values))
    assert np.isfinite(evaluated)
