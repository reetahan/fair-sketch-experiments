"""Real classification datasets as lib_classification.P problems, loaded with fs_common.load_real (LSAC, ACS California 2018 and HMDA California 2018 by default).
law : LSAC, target pass_bar (1 = passed), group 1 = racetxt == 0 (the minority; racetxt == 1 is ~94% of rows, ASSUMED white: verify)
acs : ACSIncome-style, target PINCP > 50000, group 1 = female (--acs_group sex) or non-white (race); read straight from the Census zip.
hmda: home-purchase loans, target approved vs denied, group 1 = Black applicants vs White (--hmda_group); see lib_hmda_data.py.
diab: Diabetes 130-US Hospitals (--diab), target readmitted within 30 days, group 1 = African American vs Caucasian; see lib_extra_data.py.
enem: ENEM 2020 (--enem_path), target score above median, group 1 = not White/Asian (race) or female (--enem_group sex); see lib_extra_data.py."""
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


def load_diab_class(n_max=50000, seed=0, path=None, lam=1.0):
    import lib_extra_data
    A, y, g, name = lib_extra_data.load_diabetes(n_max, seed, path, "classification")
    return P(A, y, g, lam, name=name)


def load_enem_class(path, group="race", score="NU_NOTA_CH", n_max=50000, seed=0, lam=1.0):
    import lib_extra_data
    A, y, g, name = lib_extra_data.load_enem(path, group, score, n_max, seed, "classification")
    return P(A, y, g, lam, name=name)


def real_class_problems(args):
    extra = [("Diabetes 130-US Hospitals", not args.diab,
              lambda a: load_diab_class(a.diab_n, a.seed, a.diab_path), "drop --diab"),
             (f"ENEM 2020 ({args.enem_path})", not args.enem_path,
              lambda a: load_enem_class(a.enem_path, a.enem_group, a.enem_score, a.enem_n, a.seed), "drop --enem_path")]
    return load_real(args, lambda a: load_law_class(a.law_path),
                     lambda a: load_acs_class(a.acs_state, a.acs_year, a.acs_n, a.acs_group, a.seed),
                     lambda a: load_hmda_class(a.hmda_state, a.hmda_year, a.hmda_group, a.hmda_n, a.seed, a.hmda_path),
                     extra)


def add_real_class_args(ap):
    return add_real_args(ap, acs_n=50000, hmda_n=50000)        # also adds --diab / --enem_path (shared with regression)