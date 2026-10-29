# Buenos Aires Apartment Price Prediction

Can we accurately predict apartment sale prices in Buenos Aires from size, type and location? This portfolio project works through the full supervised-learning workflow with scikit-learn: data cleaning, feature engineering, a leakage-free train/test split, a baseline, linear regression, and regularized models (Ridge and Lasso).

**Status:** Lesson 2.1 (data preparation) complete. Modeling notebooks in progress.

## Headline results

_To be filled in after lesson 2.4: best model, test MAE, RMSE and R², and one chart._

## Data

The raw listing data is not included in this repository (it comes from [SOURCE] and is not mine to redistribute).

1. Download `buenos-aires-real-estate-1.csv` through `buenos-aires-real-estate-5.csv` from [LINK OR INSTRUCTIONS].
2. Place them in `data/raw/`.
3. Run the notebooks in `notebooks/` in order.

`data/processed/` is created by the pipeline. Both data folders are git-ignored except for `.gitkeep` placeholders.

## Approach (Lesson 2.1)

- Loaded five yearly CSVs with `glob` and combined them with `pd.concat`.
- Restricted the market to Capital Federal apartments priced under $400,000, so the model learns one consistent segment.
- Trimmed outliers to the 10th-90th percentile of surface area (about [N]% of listings removed, leaving 6,507 rows).
- Extracted latitude, longitude and neighborhood from raw string columns.
- Dropped columns with more than 50% missing values, one of each highly correlated feature pair, price-derived columns (leakage), identifiers and single-value columns.
- One-hot encoded `neighborhood` with `category_encoders`, fitted on the training set only.
- 80/20 train/test split with `random_state=42`.

## Limitations

- The model only applies to mid-sized apartments in Capital Federal under $400,000.
- Outlier bounds were computed on the full dataset before the split, a small source of leakage.
- [Add anything else you noticed.]

## Notebooks

| Lesson | Notebook | Covers |
|---|---|---|
| 2.1 | `2.1_data_loading_cleaning_feature_engineering.ipynb` | Loading, filtering, outlier trim, feature extraction, missingness, multicollinearity, encoding, train/test split |
| 2.2 | (to come) | Baseline and linear regression |
| 2.3 | (to come) | Ridge and Lasso regularization |
| 2.4 | (to come) | Model comparison and selection |

## Repo layout

```text
.
├── data/
│   ├── raw/          # source CSVs (not committed)
│   └── processed/    # intermediate outputs (not committed)
├── notebooks/
│   └── 2.1_data_loading_cleaning_feature_engineering.ipynb
├── pyproject.toml
├── uv.lock
└── README.md
```

## Reproducing

Requires Python [3.12] and [`uv`](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/xrasana-lang/Buenos_Aires_price_prediction_using_Linear_Regression.git
cd Buenos_Aires_price_prediction_using_Linear_Regression
uv sync
```

Then open the notebooks in `notebooks/` in VS Code or Jupyter and select the `.venv` kernel.
