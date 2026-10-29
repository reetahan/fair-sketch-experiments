#!/usr/bin/env python3
"""Redraw every figure from the CSVs the drivers already saved, without re-running any experiment.
Only the presentation changes (trimming, labels, legends); the numbers come from the CSVs as they are. Tables (.tex) are not touched.
Usage: python replot_all.py [results/reg_s4_synth results/class_s5_real ...] [--latex]     (default: every folder under results/)"""
import argparse, glob, importlib, importlib.util, os, re, subprocess, sys
import pandas as pd
from fs_common import set_style, plot_size_direction, plot_methods_sd, plot_vanilla_sweep, plot_beta_gain, ALTER_LABEL, SFAMS
import lib_classification as cl

ROOT = os.path.dirname(os.path.abspath(__file__))
DEPS = os.path.join(ROOT, ".pydeps")      # SciencePlots is installed here: the conda overlay is mounted read-only
SCIENCEPLOTS = "SciencePlots==2.1.1"


def ensure_scienceplots():
    """Make SciencePlots importable for set_style, pip-installing it into .pydeps/ on first use (matplotlib is its only
    dependency and is already in the environment, hence --no-deps). On failure set_style falls back to the default style."""
    if DEPS not in sys.path:
        sys.path.insert(0, DEPS)
    try:
        import scienceplots  # noqa: F401
        return
    except ImportError:
        pass
    print(f"installing {SCIENCEPLOTS} into {DEPS} ...", flush=True)
    r = subprocess.run([sys.executable, "-m", "pip", "install", "--quiet", "--no-deps", "--target", DEPS, SCIENCEPLOTS])
    importlib.invalidate_caches()
    if r.returncode != 0:
        print("WARNING: could not install SciencePlots; using the matplotlib default style", flush=True)

# Regression figures: CSV stem -> (plot kind, x label, log x). The x label and scale are not stored in the CSV, so they are
# listed here; keep them in line with synthetic/s{4,5}_synthetic_regression.py and real/s{4,5}_real_regression.py.
#   sd       plot_size_direction (one line per sketch family)            -> <stem>.png
#   methods  plot_methods_sd (vanilla / oracle / proxy), one per family   -> <stem>_<family>.png
#   vanilla  plot_vanilla_sweep (empirical vs Theorem 1 bound)            -> <stem>.png
REG = {
    "synth_m": ("sd", "sketch size m", True),
    "synth_kappa": ("sd", "condition number", True),
    "synth_gap_top": ("sd", "group mean gap", True),
    "synth_gap_bottom": ("sd", "group mean gap", True),
    "synth_lev": ("sd", "leverage spread σ", False),
    "synth_lev_minority": ("sd", "minority leverage spread σ", False),
    "synth_spike_outlier": ("sd", "spike outlier size", True),
    "synth_spike_k": ("sd", "spike rows k", True),
    "s5_synth_m": ("methods", "sketch size m", True),
    "s5_synth_kappa": ("methods", "condition number", True),
    "s5_synth_gap": ("methods", "group mean gap", True),
    "s5_synth_lev": ("methods", "leverage spread σ", False),
    "s5_synth_lev_minority": ("methods", "minority leverage spread σ", False),
    "s5_synth_pilot": ("methods", "pilot size", True),
}
NOT_FIGURES = re.compile(r"(table_bounds_.*|strat_.*|.*_table|synth_spike_theory)")


def reg_spec(stem):
    if stem in REG:
        return REG[stem]
    if stem.startswith("real_m_"):
        return "sd", "sketch size m", True
    if stem.startswith("s5_real_m_"):
        return "methods", "sketch size m", True
    m = re.fullmatch(r"(ablate_vanilla|s5_real_ablate)_(\w+)", stem)
    if m and m.group(2) in ALTER_LABEL:
        return ("vanilla" if m.group(1) == "ablate_vanilla" else "methods"), ALTER_LABEL[m.group(2)], m.group(2) == "scale_col"
    return None


def replot_reg(f):
    """Redraw the figure(s) of one regression CSV; returns the PNGs written (empty if the CSV is not a figure)."""
    out, stem = os.path.dirname(f), os.path.basename(f)[:-len(".csv")]
    if stem == "s5_beta":
        plot_beta_gain(pd.read_csv(f), f"{out}/s5_beta.png")
        return [f"{out}/s5_beta.png"]
    spec = reg_spec(stem)
    if spec is None:
        return []
    kind, xlabel, logx = spec
    df = pd.read_csv(f)
    if kind == "sd":
        plot_size_direction(df, "x", xlabel, f"{out}/{stem}.png", logx=logx)
    elif kind == "vanilla":
        plot_vanilla_sweep(df, "x", xlabel, f"{out}/{stem}.png", logx=logx)
    else:
        pngs = []
        for fam in [f for f in SFAMS if f in set(df.family)]:          # the drivers draw these families only (not "is")
            plot_methods_sd(df, "x", xlabel, f"{out}/{stem}_{fam}.png", fam, logx=logx)
            pngs.append(f"{out}/{stem}_{fam}.png")
        return pngs
    return [f"{out}/{stem}.png"]


def load_driver(rel):
    """Import a driver script (they live in folders without __init__.py) to reuse its figure specs."""
    spec = importlib.util.spec_from_file_location(os.path.basename(rel)[:-3], os.path.join(ROOT, rel))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def class_specs(f, drivers):
    """(specs, min_base) for one classification CSV: the synthetic drivers' SPECS, or the real drivers' specs(<dataset>).
    min_base is each driver's own default (synthetic: emit's 0.03; real: --min_base 0.01)."""
    m = re.fullmatch(r"(s4c|s5c)_(.+)_(m|steps|gap|minority|mis)", os.path.basename(f)[:-len(".csv")])
    if not m:
        return None
    sec, name = m.group(1), m.group(2)
    if name == "synth":
        return drivers[sec, "synth"].SPECS, 0.03
    return drivers[sec, "real"].specs(name), 0.01


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="*", help="result folders (default: every folder under results/)")
    ap.add_argument("--latex", action="store_true", help="render plot text with LaTeX (needs LaTeX installed)")
    a = ap.parse_args()
    ensure_scienceplots()
    set_style(a.latex)
    drivers = {("s4c", "synth"): load_driver("synthetic/s4_synthetic_classification.py"),
               ("s5c", "synth"): load_driver("synthetic/s5_synthetic_classification.py"),
               ("s4c", "real"): load_driver("real/s4_real_classification.py"),
               ("s5c", "real"): load_driver("real/s5_real_classification.py")}
    dirs = a.dirs or sorted(d for d in glob.glob(os.path.join(ROOT, "results", "*")) if os.path.isdir(d))
    for d in dirs:
        done, skipped = 0, []
        for f in sorted(glob.glob(os.path.join(d, "*.csv"))):
            stem = os.path.basename(f)[:-len(".csv")]
            cs = class_specs(f, drivers)
            if cs is not None:
                specs, min_base = cs
                cl.replot(d, {stem: specs[stem]}, min_base)
                done += 1
            elif replot_reg(f):
                done += 1
            elif not NOT_FIGURES.fullmatch(stem):
                skipped.append(stem)
        print(f"{d}: redrew {done} figure CSV(s)" + (f"; no figure known for {skipped}" if skipped else ""), flush=True)


if __name__ == "__main__":
    main()
