"""Real classification datasets as lib_classification.P problems, loaded with fs_common.load_real (LSAC, ACS California 2018 and HMDA California 2018 by default).
law : LSAC, target pass_bar (1 = passed), group 1 = racetxt == 0 (the minority; racetxt == 1 is ~94% of rows, ASSUMED white: verify)
acs : ACSIncome-style, target PINCP > 50000, group 1 = female (--acs_group sex) or non-white (race); read straight from the Census zip.
hmda: home-purchase loans, target approved vs denied, group 1 = Black applicants vs White (--hmda_group); see lib_hmda_data.py."""
import numpy as np
import pandas as pd
from fs_common import LAW_URL, _standardise, load_acs, load_real, add_real_args
from lib_classification import P


def load_law_class(path=None, lam=1.0):
    df = pd.read_csv(path or LAW_URL).dropna()
    g = (df["racetxt"] == 0).astype(int).values
    num = df[["lsat", "ugpa", "fam_inc", "fulltime"]].astype(float)
    T = pd.get_dummies(df["tier"].astype(int), prefix="tier", drop_first=True).astype(float)
    X = _standardise(pd.concat([num, T], axis=1))
    X.insert(0, "intercept", 1.0)
    y = np.where(df["pass_bar"].values > 0.5, 1.0, -1.0)
    return P(X.values, y, g, lam, name="law-pass_bar")


def load_acs_class(state="CA", year=2018, n_max=50000, group="sex", seed=0, lam=1.0):
    pr = load_acs(state, year, n_max, group, seed)            # regression problem: A = features, b = log income, g = group
    y = np.where(pr.b > np.log(50000.0), 1.0, -1.0)
    return P(pr.A, y, pr.g, lam, name=f"acs-{state}{year}-{group}-income50k")


def load_hmda_class(state="CA", year=2018, group="race", n_max=50000, seed=0, path=None, lam=1.0):
    import lib_hmda_data
    A, y, g, name = lib_hmda_data.load(state, year, group, n_max, seed, path, "classification")
    return P(A, y, g, lam, name=name)


def real_class_problems(args):
    return load_real(args, lambda a: load_law_class(a.law_path),
                     lambda a: load_acs_class(a.acs_state, a.acs_year, a.acs_n, a.acs_group, a.seed),
                     lambda a: load_hmda_class(a.hmda_state, a.hmda_year, a.hmda_group, a.hmda_n, a.seed, a.hmda_path))


def add_real_class_args(ap):
    return add_real_args(ap, acs_n=50000, hmda_n=50000)