# Do Explanations Survive Realistic Attacks?

Code and results for **"Do Explanations Survive Realistic Attacks? SHAP Stability of Intrusion Detectors Under Constrained vs. Unconstrained Adversarial Perturbations"** by Snigda Chandanala (Department of Information Technology, Vellore Institute of Technology).

We run one score-based black-box evasion attack in two modes that differ only in the threat model (unconstrained vs. constrained to realistic, valid network flows) against Random Forest and XGBoost intrusion detectors, and measure how much their TreeSHAP explanations change.

## Repository layout
| Path | Contents |
| --- | --- |
| `src/schema.py` | Feature-constraint schemas (adapted from FEINT, MIT) |
| `src/preprocess_unsw.py`, `src/preprocess_cic.py` | Cleaning, deduplication, correlation filter |
| `src/train.py` | Random Forest and XGBoost detectors, 3 seeds |
| `src/attack_explain.py` | Constrained / unconstrained attacks, noise baseline, SHAP drift metrics |
| `src/analyze.py` | Tables, ESI, statistical tests, figures |
| `results/tables`, `results/figures` | All results reported in the paper |
| `paper/` | LaTeX source (IEEE conference format) |

## Data
Datasets are not included. Download them from the official sources:
- **UNSW-NB15**: `UNSW_NB15_training-set.csv`, `UNSW_NB15_testing-set.csv` from https://research.unsw.edu.au/projects/unsw-nb15-dataset
- **CIC-IDS2017**: `MachineLearningCSV.zip` from https://www.unb.ca/cic/datasets/ids-2017.html

Place them in a folder and set the `RAW` / `ZIP` path at the top of the preprocessing scripts.

## Reproduce
```
pip install -r requirements.txt
python src/preprocess_unsw.py
python src/train.py unsw
python src/attack_explain.py unsw
python src/analyze.py unsw
# CIC-IDS2017: same with preprocess_cic.py and "cic"
# TTL ablation: python src/train.py unsw_nosttl && python src/attack_explain.py unsw_nosttl
```
Runs on a laptop CPU; the attack + SHAP step takes about 1-2 hours per dataset.

## Acknowledgements
Constraint schemas and the black-box search are adapted from [FEINT](https://github.com/rakshit-737/feint-adversarial-ids) (MIT License).

## License
MIT (see `LICENSE`).
