# Buenos Aires Apartment Price Prediction

A portfolio project on supervised machine learning: predicting apartment
sale prices in Buenos Aires from size and location.

## Repo layout

.
├── data/
│ ├── raw/ # buenos-aires-real-estate-1..5.csv (not committed)
│ └── processed/ # any intermediate outputs (not committed)
├── notebooks/
│ └── 2.1-data-loading-cleaning-feature-engineering.ipynb
├── pyproject.toml
└── README.md


## Notebooks

| Lesson | Notebook | Covers |
|---|---|---|
| 2.1 | `2.1-data-loading-cleaning-feature-engineering.ipynb` | Loading, filtering, outlier trim, feature extraction, missingness, multicollinearity, train/test split |
| 2.2 | (to come) | Baseline + linear regression |
| 2.3 | (to come) | Ridge and Lasso regularization |
| 2.4 | (to come) | Model comparison and selection |

## Reproducing

Requires Python 3.12 and [`uv`](https://docs.astral.sh/uv/).

```bash
uv venv --python 3.12
source .venv/bin/activate
uv pip install pandas numpy scikit-learn matplotlib seaborn missingno category-encoders jupyter