
import os, urllib.parse, urllib.request
import numpy as np
import pandas as pd

BASE = "https://ffiec.cfpb.gov/v2/data-browser-api/view/csv"
WANT = {"action_taken", "derived_race", "derived_ethnicity", "derived_sex", "loan_amount", "income", "property_value",
        "combined_loan_to_value_ratio", "debt_to_income_ratio", "loan_type", "occupancy_type", "applicant_age", "ageapplicant",
        "interest_rate", "total_units", "business_or_commercial_purpose", "open-end_line_of_credit", "loan_purpose", "lien_status"}
DTI = {"<20%": 15, "20%-<30%": 25, "30%-<36%": 33, "50%-60%": 55, ">60%": 65}
AGE = {"<25": 22, "25-34": 30, "35-44": 40, "45-54": 50, "55-64": 60, "65-74": 70, ">74": 80}
GROUPS = {"race": ("derived_race", "Black or African American", "White"),
          "ethnicity": ("derived_ethnicity", "Hispanic or Latino", "Not Hispanic or Latino"),
          "sex": ("derived_sex", "Female", "Male")}


def url(state="CA", year=2018):
    q = dict(states=state, years=year, actions_taken="1,3", loan_purposes="1", lien_statuses="1")
    return BASE + "?" + urllib.parse.urlencode(q, safe=",")


def _num(s):
    return pd.to_numeric(s, errors="coerce")


def _dti(s):
    out = s.map(DTI)
    plain = _num(s)                       # the 36-49 bins appear as plain numbers
    return out.fillna(plain.where((plain >= 36) & (plain <= 49)))


HEADERS = [
    {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36",
     "Accept": "text/csv,application/csv,*/*;q=0.8", "Accept-Language": "en-US,en;q=0.9",
     "Referer": "https://ffiec.cfpb.gov/data-browser/", "Origin": "https://ffiec.cfpb.gov"},
    {"User-Agent": "curl/8.5.0", "Accept": "*/*"},
    {"User-Agent": "python-requests/2.31.0", "Accept": "*/*"},
]


def _fetch(u, dest, headers):
    import shutil
    req = urllib.request.Request(u, headers=headers[0])
    with urllib.request.urlopen(req, timeout=1800) as r, open(dest + ".part", "wb") as f:
        shutil.copyfileobj(r, f)
    os.replace(dest + ".part", dest)


def download(state, year, dest):
    """Stream the filtered CSV to `dest`, trying several header sets. Raises RuntimeError with manual-download steps on failure."""
    import shutil
    last = None
    for h in HEADERS:
        try:
            req = urllib.request.Request(url(state, year), headers=h)
            with urllib.request.urlopen(req, timeout=900) as r, open(dest + ".part", "wb") as f:
                shutil.copyfileobj(r, f)
            os.replace(dest + ".part", dest)
            return dest
        except Exception as e:                       # HTTPError 403, timeouts, proxies
            last = e
    raise RuntimeError(
        f"{last}. The CFPB server refused the request (tried {len(HEADERS)} header sets). Download the file in a browser instead: "
        f"open https://ffiec.cfpb.gov/data-browser/data/{year}?category=states&items={state}&actions_taken=1,3&loan_purposes=1&lien_statuses=1 "
        f"(or paste this URL into the browser address bar: {url(state, year)}), save the CSV, then rerun with --hmda_path <file.csv>.")


HF_URL = "https://huggingface.co/datasets/retrogradespace/hmda_2024/resolve/main/hmda_2024.parquet"   # unmodified CFPB 2024 LAR, 538 MB, CC0
HF_YEARS = {2024}


def _as_text(df):
    """Parquet keeps numeric dtypes; the cleaner expects the CSV text form ('1', not 1.0 or 1)."""
    out = {}
    for c in df.columns:
        s = df[c]
        if pd.api.types.is_numeric_dtype(s):
            out[c] = s.map(lambda v: None if pd.isna(v) else (str(int(v)) if float(v).is_integer() else str(v)))
        else:
            out[c] = s.astype(object).where(s.notna(), None)
    return pd.DataFrame(out)


def read_parquet(path, state):
    cols = None
    import pyarrow.parquet as pq
    names = set(pq.ParquetFile(path).schema_arrow.names)
    cols = sorted(WANT & names | ({"state_code"} & names))
    df = pd.read_parquet(path, columns=cols, filters=[("state_code", "==", state)] if "state_code" in names else None)
    return _as_text(df.drop(columns=["state_code"], errors="ignore"))


def read_raw(state="CA", year=2018, path=None):
    """Source order: --hmda_path (.csv or .parquet); else year 2024 -> Hugging Face parquet (cached in hmda_cache/); else the CFPB API (cached)."""
    if path is None:
        os.makedirs("hmda_cache", exist_ok=True)
        if year in HF_YEARS:
            path = os.path.join("hmda_cache", f"hmda_{year}.parquet")
            if not os.path.exists(path):
                _fetch(HF_URL, path, HEADERS[1:2])
        else:
            path = os.path.join("hmda_cache", f"hmda_{state}_{year}.csv")
            if not os.path.exists(path):
                download(state, year, path)
    if str(path).endswith(".parquet"):
        return read_parquet(path, state)
    return pd.read_csv(path, usecols=lambda c: c in WANT, dtype=str, low_memory=False)


def clean(raw, group="race", n_max=200000, seed=0):
    """Return a DataFrame with numeric columns: features, action (1/3), rate, g."""
    df = raw.copy()
    col, g1, g0 = GROUPS[group]
    need = {"action_taken", col, "loan_amount", "income", "property_value", "combined_loan_to_value_ratio", "debt_to_income_ratio", "loan_type"}
    missing = need - set(df.columns)
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
    out = pd.DataFrame({"action": df.action, "g": df.g,
                        "income": _num(df["income"]), "loan": _num(df["loan_amount"]), "value": _num(df["property_value"]),
                        "cltv": _num(df["combined_loan_to_value_ratio"]), "dti": _dti(df["debt_to_income_ratio"]),
                        "age": df[age_col].map(AGE) if age_col in df.columns else np.nan,
                        "ltype": _num(df["loan_type"]),
                        "rate": _num(df["interest_rate"]) if "interest_rate" in df.columns else np.nan})
    out = out[(out.income > 0) & (out.loan > 0) & (out.value > 0) & out.cltv.notna() & out.dti.notna() & out.age.notna() & out.ltype.notna()]
    out["cltv"] = out.cltv.clip(0, 250)
    out = out.reset_index(drop=True)
    if len(out) > n_max:
        out = out.sample(n_max, random_state=seed).reset_index(drop=True)
    return out


def features(d):
    X = pd.DataFrame({"log_income": np.log(d.income), "log_loan": np.log(d.loan), "log_value": np.log(d.value),
                      "cltv": d.cltv, "dti": d.dti, "age": d.age})
    lt = pd.get_dummies(d.ltype.astype(int).astype(str), prefix="loan_type", drop_first=True).astype(float)
    X = pd.concat([X, lt], axis=1)
    X = (X - X.mean()) / X.std().replace(0, 1)
    X.insert(0, "intercept", 1.0)
    return X.values


def load(state="CA", year=2018, group="race", n_max=200000, seed=0, path=None, task="regression"):
    """Returns (A, y, g, name). regression: y = interest rate of originated loans. classification: y in {-1,+1} (+1 = approved)."""
    d = clean(read_raw(state, year, path), group, n_max if task == "classification" else 10 ** 9, seed)
    if task == "regression":
        d = d[(d.action == 1) & (d.rate > 0) & (d.rate < 20)].reset_index(drop=True)
        if len(d) > n_max:
            d = d.sample(n_max, random_state=seed).reset_index(drop=True)
        y = d.rate.values.astype(float)
    else:
        y = np.where(d.action.values == 1, 1.0, -1.0)
    name = f"hmda-{state}{year}-{group}-" + ("rate" if task == "regression" else "approval")
    print(f"HMDA {name}: n={len(d)}, minority share={d.g.mean():.3f}, " + (f"mean rate={y.mean():.2f}" if task == "regression" else f"approval rate={(y > 0).mean():.3f}"), flush=True)
    return features(d), y, d.g.values.astype(int), name