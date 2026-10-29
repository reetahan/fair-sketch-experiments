
import numpy as np
import pandas as pd
from pandas.api.types import union_categoricals
from fs_common import _standardise

DIAB_NUM = ["num_lab_procedures", "num_procedures", "num_medications", "number_diagnoses"]
DIAB_CAT = ["gender", "age", "primary_diagnosis", "max_glu_serum", "A1Cresult", "insulin", "change",
            "diabetesMed", "medicare", "medicaid", "had_emergency", "had_inpatient_days", "had_outpatient_days"]

ENEM_PRES = ["TP_PRESENCA_CN", "TP_PRESENCA_CH", "TP_PRESENCA_LC", "TP_PRESENCA_MT"]
ENEM_Q = [f"Q{i:03d}" for i in range(1, 25)]


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


def load_diabetes(n_max=None, seed=0, path=None, task="classification"):
    if path:
        df = pd.read_csv(path, keep_default_na=False, na_values=[""])   
    else:
        from fairlearn.datasets import fetch_diabetes_hospital  
        df = fetch_diabetes_hospital(as_frame=True).frame
    df = df[df["race"].isin(["AfricanAmerican", "Caucasian"])]
    df = df[df["gender"].isin(["Male", "Female"])].dropna(subset=DIAB_NUM + DIAB_CAT)
    df = _subsample(df, n_max, seed)
    g = (df["race"] == "AfricanAmerican").astype(int).values
    A = _design(df[DIAB_NUM], df[DIAB_CAT])
    if task == "regression":
        return A, df["time_in_hospital"].astype(float).values, g, "diab-length_of_stay"
    y = np.where(df["readmit_30_days"].astype(int).values == 1, 1.0, -1.0)
    return A, y, g, "diab-readmit30"


def load_enem(path, group="race", score="NU_NOTA_CH", n_max=None, seed=0, task="classification"):
    cols = [score, "TP_COR_RACA", "TP_SEXO", "SG_UF_PROVA", "TP_FAIXA_ETARIA",
            "IN_TREINEIRO", "TP_ST_CONCLUSAO"] + ENEM_PRES + ENEM_Q
    cat_dtypes = {c: "category" for c in ENEM_Q + ["TP_SEXO", "SG_UF_PROVA"] if c != "Q005"}   
    keep = []
    for ch in pd.read_csv(path, sep=";", encoding="latin-1", usecols=cols, dtype=cat_dtypes, chunksize=500_000):   # 5.8M rows, ~2 GB: filter per chunk
        ch = ch[(ch[ENEM_PRES] == 1).all(axis=1)]
        ch = ch[(ch["IN_TREINEIRO"] == 0) & (ch["TP_ST_CONCLUSAO"] == 1)]
        keep.append(ch[ch["TP_COR_RACA"].between(1, 5)].dropna())
    df = pd.DataFrame({c: union_categoricals([ch[c] for ch in keep]) if isinstance(keep[0][c].dtype, pd.CategoricalDtype)
                       else pd.concat([ch[c] for ch in keep], ignore_index=True) for c in cols})   # chunk categories differ
    med = df[score].median()
    df = _subsample(df, n_max, seed)
    if group == "race":
        g = (~df["TP_COR_RACA"].isin([1, 4])).astype(int).values  
    else:
        g = (df["TP_SEXO"] == "F").astype(int).values
    num = pd.DataFrame({"household": df["Q005"].astype(float)})
    if group == "race":                                        
        num["female"] = (df["TP_SEXO"] == "F").astype(float)
    cat = df[["SG_UF_PROVA", "TP_FAIXA_ETARIA"] + [q for q in ENEM_Q if q != "Q005"]]
    A = _design(num, cat)
    subj = score.replace("NU_NOTA_", "").lower()
    if task == "regression":
        return A, df[score].astype(float).values, g, f"enem-{subj}-{group}"
    y = np.where(df[score].values > med, 1.0, -1.0)
    return A, y, g, f"enem-{subj}-{group}-above_median"