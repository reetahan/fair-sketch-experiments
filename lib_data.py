import io, os, shutil, urllib.parse, urllib.request, zipfile
import numpy as np
import pandas as pd
from pandas.api.types import union_categoricals
from fs_common import Problem
from lib_classification import P

LAW_URL = "https://raw.githubusercontent.com/damtharvey/law-school-dataset/main/law_dataset.csv"
DIAB_NUM = ["num_lab_procedures", "num_procedures", "num_medications", "number_diagnoses"]
DIAB_CAT = ["gender", "age", "primary_diagnosis", "max_glu_serum", "A1Cresult", "insulin", "change", "diabetesMed",
            "medicare", "medicaid", "had_emergency", "had_inpatient_days", "had_outpatient_days"]
ENEM_PRES = ["TP_PRESENCA_CN", "TP_PRESENCA_CH", "TP_PRESENCA_LC", "TP_PRESENCA_MT"]
ENEM_Q = [f"Q{i:03d}" for i in range(1, 25)]
HMDA_API = "https://ffiec.cfpb.gov/v2/data-browser-api/view/csv"
HMDA_HF = "https://huggingface.co/datasets/retrogradespace/hmda_2024/resolve/main/hmda_2024.parquet"
HMDA_WANT = {"action_taken", "derived_race", "derived_ethnicity", "derived_sex", "loan_amount", "income",
             "property_value", "combined_loan_to_value_ratio", "loan_to_value_ratio", "debt_to_income_ratio",
             "loan_type", "occupancy_type", "applicant_age", "ageapplicant", "interest_rate", "total_units",
             "business_or_commercial_purpose", "open-end_line_of_credit", "loan_purpose", "lien_status"}
HMDA_DTI = {"<20%": 15, "20%-<30%": 25, "30%-<36%": 33, "50%-60%": 55, ">60%": 65}
HMDA_AGE = {"<25": 22, "25-34": 30, "35-44": 40, "45-54": 50, "55-64": 60, "65-74": 70, ">74": 80}
HMDA_GROUPS = {"race": ("derived_race", "Black or African American", "White"),
               "ethnicity": ("derived_ethnicity", "Hispanic or Latino", "Not Hispanic or Latino"),
               "sex": ("derived_sex", "Female", "Male")}
HEADERS = [{"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 "
                          "Safari/537.36", "Accept": "text/csv,application/csv,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9", "Referer": "https://ffiec.cfpb.gov/data-browser/",
            "Origin": "https://ffiec.cfpb.gov"},
           {"User-Agent": "curl/8.5.0", "Accept": "*/*"},
           {"User-Agent": "python-requests/2.31.0", "Accept": "*/*"}]


def _standardise(df_num):
    out = df_num.copy()
    for c in out.columns:
        s = out[c].std()
        if s > 0:
            out[c] = (out[c] - out[c].mean()) / s
    return out


def _design(num, cat):
    D = pd.get_dummies(cat.astype(str), drop_first=True).astype(float)
    X = _standardise(pd.concat([num.astype(float), D], axis=1))
    X = X.loc[:, X.std() > 0]
    X.insert(0, "intercept", 1.0)
    return X.values


def _subsample(df, n_max, seed):
    if n_max and len(df) > n_max:
        df = df.sample(n=n_max, random_state=seed)
    return df.reset_index(drop=True)


def load_law(path=None, task="regression"):
    df = pd.read_csv(path or LAW_URL).dropna()
    num = df[["lsat", "ugpa", "fam_inc", "fulltime"]].astype(float)
    T = pd.get_dummies(df["tier"].astype(int), prefix="tier", drop_first=True).astype(float)
    X = _standardise(pd.concat([num, T], axis=1))
    X.insert(0, "intercept", 1.0)
    if task == "regression":
        return X.values, df["zfygpa"].values.astype(float), (df["racetxt"] == 1).astype(int).values, "law-race"
    return X.values, np.where(df["pass_bar"].values > 0.5, 1.0, -1.0), (df["racetxt"] == 0).astype(int).values, \
        "law-pass_bar"


def load_acs(state="CA", year=2018, n_max=200000, group="sex", seed=0, task="regression"):
    url = f"https://www2.census.gov/programs-surveys/acs/data/pums/{year}/1-Year/csv_p{state.lower()}.zip"
    raw = urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"}), timeout=300).read()
    z = zipfile.ZipFile(io.BytesIO(raw))
    cols = ["AGEP", "COW", "SCHL", "MAR", "RELP", "WKHP", "SEX", "RAC1P", "PINCP", "PWGTP"]
    df = pd.read_csv(z.open([n for n in z.namelist() if n.lower().endswith(".csv")][0]), usecols=cols)
    df = df[(df.AGEP > 16) & (df.PINCP > 100) & (df.WKHP > 0) & (df.PWGTP >= 1)].dropna()
    if len(df) > n_max:
        df = df.sample(n_max, random_state=seed)
    g = (df["SEX"] == 2).astype(int).values if group == "sex" else (df["RAC1P"] != 1).astype(int).values
    age = df.AGEP.astype(float)
    num = pd.DataFrame({"AGEP": age, "AGEP2": age**2, "SCHL": df.SCHL.astype(float), "WKHP": df.WKHP.astype(float)})
    cat = pd.get_dummies(df[["COW", "MAR", "RELP"]].astype(int).astype(str), drop_first=True).astype(float)
    X = _standardise(pd.concat([num.reset_index(drop=True), cat.reset_index(drop=True)], axis=1))
    X.insert(0, "intercept", 1.0)
    y = np.log(df["PINCP"].values.astype(float))
    if task == "regression":
        return X.values, y, g, f"acs-{state}{year}-{group}"
    return X.values, np.where(y > np.log(50000.0), 1.0, -1.0), g, f"acs-{state}{year}-{group}-income50k"


def load_diabetes(n_max=None, seed=0, path=None, task="classification"):
    if path:
        df = pd.read_csv(path, keep_default_na=False, na_values=[""])
    else:
        from fairlearn.datasets import fetch_diabetes_hospital

        df = fetch_diabetes_hospital(as_frame=True).frame
    df = df[df["race"].isin(["AfricanAmerican", "Caucasian"])]
    df = _subsample(df[df["gender"].isin(["Male", "Female"])].dropna(subset=DIAB_NUM + DIAB_CAT), n_max, seed)
    g = (df["race"] == "AfricanAmerican").astype(int).values
    A = _design(df[DIAB_NUM], df[DIAB_CAT])
    if task == "regression":
        return A, df["time_in_hospital"].astype(float).values, g, "diab-length_of_stay"
    return A, np.where(df["readmit_30_days"].astype(int).values == 1, 1.0, -1.0), g, "diab-readmit30"


def load_enem(path, group="race", score="NU_NOTA_CH", n_max=None, seed=0, task="classification"):
    cols = [score, "TP_COR_RACA", "TP_SEXO", "SG_UF_PROVA", "TP_FAIXA_ETARIA", "IN_TREINEIRO", "TP_ST_CONCLUSAO"] \
        + ENEM_PRES + ENEM_Q
    cat_dtypes = {c: "category" for c in ENEM_Q + ["TP_SEXO", "SG_UF_PROVA"] if c != "Q005"}
    keep = []
    for ch in pd.read_csv(path, sep=";", encoding="latin-1", usecols=cols, dtype=cat_dtypes, chunksize=500_000):
        ch = ch[(ch[ENEM_PRES] == 1).all(axis=1)]
        ch = ch[(ch["IN_TREINEIRO"] == 0) & (ch["TP_ST_CONCLUSAO"] == 1)]
        keep.append(ch[ch["TP_COR_RACA"].between(1, 5)].dropna())
    df = pd.DataFrame({c: (union_categoricals([ch[c] for ch in keep]) if isinstance(keep[0][c].dtype, pd.CategoricalDtype)
                           else pd.concat([ch[c] for ch in keep], ignore_index=True)) for c in cols})
    med = df[score].median()
    df = _subsample(df, n_max, seed)
    if group == "race":
        g = (~df["TP_COR_RACA"].isin([1, 4])).astype(int).values
    else:
        g = (df["TP_SEXO"] == "F").astype(int).values
    num = pd.DataFrame({"household": df["Q005"].astype(float)})
    if group == "race":
        num["female"] = (df["TP_SEXO"] == "F").astype(float)
    A = _design(num, df[["SG_UF_PROVA", "TP_FAIXA_ETARIA"] + [q for q in ENEM_Q if q != "Q005"]])
    subj = score.replace("NU_NOTA_", "").lower()
    if task == "regression":
        return A, df[score].astype(float).values, g, f"enem-{subj}-{group}"
    return A, np.where(df[score].values > med, 1.0, -1.0), g, f"enem-{subj}-{group}-above_median"


def _num(s):
    return pd.to_numeric(s, errors="coerce")


def _download(url, dest, headers, timeout):
    for i, h in enumerate(headers):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=h), timeout=timeout) as r, \
                    open(dest + ".part", "wb") as f:
                shutil.copyfileobj(r, f)
            os.replace(dest + ".part", dest)
            return
        except Exception:
            if i == len(headers) - 1:
                raise


def read_hmda(state="CA", year=2018, path=None):
    if path is None:
        os.makedirs("hmda_cache", exist_ok=True)
        if year == 2024:
            path = os.path.join("hmda_cache", f"hmda_{year}.parquet")
            if not os.path.exists(path):
                _download(HMDA_HF, path, HEADERS[1:2], 1800)
        else:
            path = os.path.join("hmda_cache", f"hmda_{state}_{year}.csv")
            url = HMDA_API + "?" + urllib.parse.urlencode(
                dict(states=state, years=year, actions_taken="1,3", loan_purposes="1"), safe=",")
            if not os.path.exists(path):
                try:
                    _download(url, path, HEADERS, 900)
                except Exception as e:
                    raise RuntimeError(
                        f"{e}. The CFPB server refused the request (tried {len(HEADERS)} header sets). Download the "
                        f"file in a browser instead: open https://ffiec.cfpb.gov/data-browser/data/{year}?category="
                        f"states&items={state}&actions_taken=1,3&loan_purposes=1 (or paste this URL into the browser "
                        f"address bar: {url}), save the CSV, then rerun with --hmda_path <file.csv>.")
    if not str(path).endswith(".parquet"):
        return pd.read_csv(path, usecols=lambda c: c in HMDA_WANT, dtype=str, low_memory=False)
    import pyarrow.parquet as pq

    names = set(pq.ParquetFile(path).schema_arrow.names)
    df = pd.read_parquet(path, columns=sorted(HMDA_WANT & names | ({"state_code"} & names)),
                         filters=[("state_code", "==", state)] if "state_code" in names else None)
    df = df.drop(columns=["state_code"], errors="ignore")
    return pd.DataFrame({c: (df[c].map(lambda v: None if pd.isna(v) else (str(int(v)) if float(v).is_integer() else str(v)))
                             if pd.api.types.is_numeric_dtype(df[c]) else df[c].astype(object).where(df[c].notna(), None))
                         for c in df.columns})


def clean_hmda(raw, group="race", n_max=200000, seed=0):
    df = raw.copy()
    if "combined_loan_to_value_ratio" not in df.columns and "loan_to_value_ratio" in df.columns:
        df = df.rename(columns={"loan_to_value_ratio": "combined_loan_to_value_ratio"})
    col, g1, g0 = HMDA_GROUPS[group]
    missing = {"action_taken", col, "loan_amount", "income", "property_value", "combined_loan_to_value_ratio",
               "debt_to_income_ratio", "loan_type"} - set(df.columns)
    if missing:
        raise ValueError(f"HMDA file is missing columns {sorted(missing)}; columns present: {sorted(df.columns)}")
    age_col = "applicant_age" if "applicant_age" in df.columns else "ageapplicant"
    df["action"] = _num(df["action_taken"])
    df = df[df.action.isin([1, 3])]
    for c, ok in (("occupancy_type", [1]), ("lien_status", [1]), ("loan_purpose", [1])):
        if c in df.columns:
            df = df[_num(df[c]).isin(ok)]
    for c in ("business_or_commercial_purpose", "open-end_line_of_credit"):
        if c in df.columns:
            df = df[~(_num(df[c]) == 1)]
    if "total_units" in df.columns:
        df = df[df["total_units"].isin(["1", "2", "3", "4"])]
    df = df[df[col].isin([g1, g0])]
    df["g"] = (df[col] == g1).astype(int)
    dti = _num(df["debt_to_income_ratio"])
    out = pd.DataFrame(
        {"action": df.action, "g": df.g, "income": _num(df["income"]), "loan": _num(df["loan_amount"]),
         "value": _num(df["property_value"]), "cltv": _num(df["combined_loan_to_value_ratio"]),
         "dti": df["debt_to_income_ratio"].map(HMDA_DTI).fillna(dti.where((dti >= 36) & (dti <= 49))),
         "age": df[age_col].map(HMDA_AGE) if age_col in df.columns else np.nan, "ltype": _num(df["loan_type"]),
         "rate": _num(df["interest_rate"]) if "interest_rate" in df.columns else np.nan})
    out = out[(out.income > 0) & (out.loan > 0) & (out.value > 0) & out.cltv.notna() & out.dti.notna() & out.age.notna()
              & out.ltype.notna()]
    out["cltv"] = out.cltv.clip(0, 250)
    out = out.reset_index(drop=True)
    if len(out) > n_max:
        out = out.sample(n_max, random_state=seed).reset_index(drop=True)
    return out


def load_hmda(state="CA", year=2018, group="race", n_max=200000, seed=0, path=None, task="regression"):
    d = clean_hmda(read_hmda(state, year, path), group, n_max if task == "classification" else 10**9, seed)
    if task == "regression":
        d = d[(d.action == 1) & (d.rate > 0) & (d.rate < 20)].reset_index(drop=True)
        if len(d) > n_max:
            d = d.sample(n_max, random_state=seed).reset_index(drop=True)
        y = d.rate.values.astype(float)
    else:
        y = np.where(d.action.values == 1, 1.0, -1.0)
    name = f"hmda-{state}{year}-{group}-" + ("rate" if task == "regression" else "approval")
    print(f"HMDA {name}: n={len(d)}, minority share={d.g.mean():.3f}, "
          + (f"mean rate={y.mean():.2f}" if task == "regression" else f"approval rate={(y > 0).mean():.3f}"), flush=True)
    X = pd.DataFrame({"log_income": np.log(d.income), "log_loan": np.log(d.loan), "log_value": np.log(d.value),
                      "cltv": d.cltv, "dti": d.dti, "age": d.age})
    lt = pd.get_dummies(d.ltype.astype(int).astype(str), prefix="loan_type", drop_first=True).astype(float)
    X = pd.concat([X, lt], axis=1)
    X = (X - X.mean()) / X.std().replace(0, 1)
    X.insert(0, "intercept", 1.0)
    return X.values, y, d.g.values.astype(int), name


def real_problems(a, task="regression"):
    """Problem (regression) or P (classification) for each real data set selected by the flags of add_real_args."""
    wrap = (lambda A, y, g, name: Problem(A, y, g, name)) if task == "regression" else \
        (lambda A, y, g, name: P(A, y, g, 1.0, name=name))
    sources = [("LSAC (law school)", a.no_law, lambda: load_law(a.law_path, task), "pass --no_law"),
               (f"ACS {a.acs_state} {a.acs_year}", a.no_acs,
                lambda: load_acs(a.acs_state, a.acs_year, a.acs_n, a.acs_group, a.seed, task), "pass --no_acs"),
               (f"HMDA {a.hmda_state} {a.hmda_year}", a.no_hmda,
                lambda: load_hmda(a.hmda_state, a.hmda_year, a.hmda_group, a.hmda_n, a.seed, a.hmda_path, task),
                "pass --no_hmda"),
               ("Diabetes 130-US Hospitals", not a.diab, lambda: load_diabetes(a.diab_n, a.seed, a.diab_path, task),
                "drop --diab"),
               (f"ENEM 2020 ({a.enem_path})", not a.enem_path,
                lambda: load_enem(a.enem_path, a.enem_group, a.enem_score, a.enem_n, a.seed, task),
                "drop --enem_path")]
    out = []
    for name, skip, fn, how in sources:
        if skip:
            continue
        print(f"loading {name} ...", flush=True)
        try:
            out.append(wrap(*fn()))
        except Exception as e:
            msg = f"could not load {name} ({type(e).__name__}: {e})"
            if not a.allow_missing:
                raise SystemExit(f"ERROR: {msg}. Fix the connection or path, or {how} to skip it, or --allow_missing "
                                 "to continue without it.")
            print(f"WARNING: {msg}; continuing without it.", flush=True)
    return out


def add_real_args(ap, n=200000):
    ap.add_argument("--law_path", default=None, help="local csv instead of the GitHub url")
    ap.add_argument("--no_law", action="store_true", help="skip LSAC")
    ap.add_argument("--no_acs", action="store_true", help="skip ACS")
    ap.add_argument("--acs_state", default="CA", help="ACS state (downloaded into memory)")
    ap.add_argument("--acs_year", type=int, default=2018)
    ap.add_argument("--acs_n", type=int, default=n, help="max ACS rows (random subsample)")
    ap.add_argument("--acs_group", default="sex", choices=["sex", "race"])
    ap.add_argument("--no_hmda", action="store_true", help="skip HMDA")
    ap.add_argument("--hmda_state", default="CA")
    ap.add_argument("--hmda_year", type=int, default=2018)
    ap.add_argument("--hmda_n", type=int, default=n, help="max HMDA rows (random subsample)")
    ap.add_argument("--hmda_group", default="race", choices=["race", "ethnicity", "sex"],
                    help="race: Black vs White applicants")
    ap.add_argument("--hmda_path", default=None, help="local CSV from https://ffiec.cfpb.gov/data-browser/")
    ap.add_argument("--diab", action="store_true", help="add Diabetes 130-US Hospitals")
    ap.add_argument("--diab_path", default=None, help="local CSV of the Fairlearn frame; default fetches from OpenML")
    ap.add_argument("--diab_n", type=int, default=50000)
    ap.add_argument("--enem_path", default=None, help="MICRODADOS_ENEM_2020.csv from INEP; ENEM is skipped if unset")
    ap.add_argument("--enem_group", default="race", choices=["race", "sex"])
    ap.add_argument("--enem_score", default="NU_NOTA_CH", choices=["NU_NOTA_CH", "NU_NOTA_LC", "NU_NOTA_MT", "NU_NOTA_CN"])
    ap.add_argument("--enem_n", type=int, default=50000)
    ap.add_argument("--allow_missing", action="store_true", help="warn and continue if a data set cannot be loaded")
    return ap
