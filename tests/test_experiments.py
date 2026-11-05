"""The committed experiment results and the scripts that plot them stay in sync.

Each plot script is run on a copy of experiments/ (so the committed PNGs are
never overwritten) and must exit cleanly and write its plot. The training
runners are only checked for syntax: running them takes minutes per seed.
"""
import ast
import os
import shutil
import subprocess
import sys

import pytest

EXPERIMENTS = os.path.join(os.path.dirname(__file__), "..", "experiments")

# (script, arguments, plot it writes)
PLOTS = [
    ("plot_exp1.py", [], "exp1_rewards.png"),
    ("plot_exp2.py", [], "exp2_norm.png"),
    ("plot_tuning.py", [], "tuning_rewards.png"),
    ("plot_sweep.py", [], "sweep.png"),
    ("plot_duration.py", [], "duration.png"),
    ("plot_exp8.py", [], "exp8_agents.png"),
    ("plot_scenarios.py", [], "scenarios.png"),
    ("plot_robustness.py", [], "robustness.png"),
    ("plot_robustness.py", ["robustness_100iters_results.json"], "robustness_100iters.png"),
]


@pytest.fixture
def experiments_copy(tmp_path):
    for name in os.listdir(EXPERIMENTS):
        if name.endswith((".py", ".json")):
            shutil.copy(os.path.join(EXPERIMENTS, name), tmp_path)
    return tmp_path


@pytest.mark.parametrize("script,args,plot", PLOTS, ids=[f"{s} {' '.join(a)}".strip() for s, a, _ in PLOTS])
def test_plot_script_runs_on_committed_results(experiments_copy, script, args, plot):
    pytest.importorskip("matplotlib")
    result = subprocess.run(
        [sys.executable, str(experiments_copy / script), *args],
        capture_output=True, text=True, env={**os.environ, "MPLBACKEND": "Agg"}, timeout=120,
    )
    assert result.returncode == 0, result.stderr
    assert (experiments_copy / plot).stat().st_size > 10_000


def test_every_plot_script_is_covered():
    scripts = {f for f in os.listdir(EXPERIMENTS) if f.startswith("plot_") and f.endswith(".py")}
    assert scripts == {s for s, _, _ in PLOTS}


@pytest.mark.parametrize("script", sorted(f for f in os.listdir(EXPERIMENTS) if f.startswith("run_") and f.endswith(".py")))
def test_runner_scripts_parse(script):
    with open(os.path.join(EXPERIMENTS, script), encoding="utf-8") as f:
        ast.parse(f.read(), filename=script)
