# fair-sketch-experiments

Experiments on how sketching the data distorts fairness metrics of least squares and ridge logistic regression (Section 4), and on sketches that reduce the distortion (Section 5).

## Setup

Python 3 (tested on 3.12) with `numpy`, `pandas`, `scipy` and `matplotlib`. Optional: `fairlearn` (130-Hospitals data), `pyarrow` (HMDA 2024), `torch` (MLP extension). SciencePlots is installed into `.pydeps/` on first use.

## Layout

| File | Contents |
|---|---|
| `s4_regression.py`, `s4_classification.py` | Section 4: distortion of Gaussian, CountSketch, uniform and leverage sketches |
| `s5_regression.py`, `s5_classification.py` | Section 5: group-stratified sketches (regression) and targeted coresets (classification) |
| `fs_common.py` | Regression problem, sketches, bounds, tables and plotting |
| `lib_classification.py` | Logistic model, Newton and LogReg sketches, targeted coresets |
| `lib_data.py` | Loaders for the real data sets |
| `extensions/` | Further experiments: pilot and mixing settings, runtime, MLPs, when targeting beats leverage |

## Running

Each script writes CSVs, tables and figures to `--out` (default under `results/`). Add `--quick` for a small test run.

```bash
python s4_regression.py --synthetic                  # synthetic data
python s4_regression.py --no_acs --no_hmda           # LSAC only
python s5_classification.py --diab --enem_path MICRODADOS_ENEM_2020.csv
python extensions/s5c_pilot.py --sweep d             # pilot size; --sweep frac or mu for the others
```

## Data

Synthetic runs (`--synthetic`) need no data. Real-data runs download LSAC, ACS and HMDA by default, so they need internet access. Only ENEM needs a file you download yourself: get the 2020 microdata [from here]
(https://www.gov.br/inep/pt-br/acesso-a-informacao/dados-abertos/microdados/enem) and pass the CSV with `--enem_path`. Without that flag ENEM is skipped, so the ENEM results in the paper need the file. If the CFPB API refuses the HMDA request, the error message gives a browser link; save the CSV and pass it with `--hmda_path`.

| Data set | Source | Flag |
|---|---|---|
| LSAC | downloaded from GitHub | on by default; `--no_law` to skip |
| ACS Income (CA 2018) | downloaded from the Census Bureau | on by default; `--no_acs` |
| HMDA (CA 2018) | CFPB API, cached in `hmda_cache/` | on by default; `--no_hmda`, or `--hmda_path` for a saved CSV |
| 130-Hospitals | `fairlearn` | `--diab` |
| ENEM 2020 | INEP microdata, downloaded by hand | `--enem_path` |

## License

MIT
