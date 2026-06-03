# Price Intelligence - MrScraper Take-Home Test

Predicting e-commerce product prices on a scraping outage day using historical data
and a small **anchor set** of 100 manually collected prices.

---

## Problem Statement

When scraping is unavailable, a team manually collects prices for ~100 products.
Given this anchor set + full historical data, reconstruct prices for all other
tracked products on that same day.

---

## Project Structure

```
├── 1_eda.ipynb                 <- Exploratory Data Analysis
├── 2_preprocessing.ipynb       <- Data cleaning and preparation
├── 3_feature_engineering.ipynb <- Feature creation and train/val split
├── 4_modelling.ipynb           <- Tier 1 & Tier 2 model training + evaluation
├── 5_prediction.ipynb          <- Generate predictions for test file
├── requirements.txt
└── README.md
```

**Intermediate files generated automatically between notebooks:**
```
data_raw.csv          <- output of 1_eda.ipynb
data_preprocessed.csv <- output of 2_preprocessing.ipynb
data_train.csv        <- output of 3_feature_engineering.ipynb
data_val.csv          <- output of 3_feature_engineering.ipynb
feature_cols.pkl      <- output of 3_feature_engineering.ipynb
model.pkl             <- output of 4_modelling.ipynb
predictions_3days.csv <- output of 5_prediction.ipynb : all 25,900 rows with prices filled in
```

---

## How to Run

### 1. Download the data files

Download both files and place them in the same directory as the notebooks:

- **Train set**: https://drive.google.com/file/d/1ZOYuyrcBJF7fvnG6kBva1m8og5urXHjU/view
- **Test set (3 days)**: https://drive.google.com/file/d/1Ni2aBrOaV1YEWspZVmBw__HV1a7M37cd/view

After downloading, rename them to:
```
ecommerce_price_prediction-train.csv
ecommerce_price_prediction-test-3-days.csv
```

### 2. Install dependencies

Requires **Python 3.9**. Then install the required packages:
```bash
pip install -r requirements.txt
```

### 3. Run notebooks in order
Run each notebook top to bottom using **Kernel -> Restart & Run All**:

```
1_eda.ipynb                 -> generates data_raw.csv
2_preprocessing.ipynb       -> generates data_preprocessed.csv
3_feature_engineering.ipynb -> generates data_train.csv, data_val.csv, feature_cols.pkl
4_modelling.ipynb           -> generates model.pkl
5_prediction.ipynb          -> generates predictions_3days.csv
```

## Approach

### Feature Engineering

| Feature | Description |
|---|---|
| `hour`, `dayofweek`, `month`, `day` | Temporal signals extracted from `capturedAt` |
| `discount_depth` | `raw_discount / 100`, discount as a decimal (0.0-1.0) |
| `has_promotion` | 1 if active promotion exists, 0 otherwise |
| `has_discount` | 1 if `priceBeforeDiscount > 0`, 0 otherwise |
| `log_price` | Log-transformed target for more stable model training |
| `model_last_price` | Last known price for this exact `modelId`, strongest predictor |
| `model_median_price` | Median historical price per `modelId` |
| `model_mean_price` | Mean historical price per `modelId` |
| `model_count` | Number of historical observations per `modelId` |

### Validation Methodology

Time-based split to simulate the outage scenario:
- **Train**: all data before 2025-03-22 -> 301,355 rows
- **Validation**: 2025-03-22 (held out entirely) -> 4,871 rows
- **Anchor simulation**: 100 random rows sampled from validation day

This mirrors the exact test conditions, where the model only knows the past.

### Tier 1 - Global Model (RandomForest)

One `RandomForestRegressor` trained on all 301,355 training rows.
Target is `log(price)` to reduce the impact of extreme outliers.
Predictions are back-transformed with `np.expm1()`.

**Why RandomForest?**
- No feature scaling required, tree-based models are scale-invariant
- Robust to outliers, averaging across 100 trees reduces noise
- Suitable for tabular data, consistently outperforms neural networks on structured data

### Tier 2 - Product-Level Model (Last Price)

For each `modelId`, use its last known price as the prediction,
adjusted by the anchor calibration ratio:

```python
prediction = model_last_price * anchor_ratio
```

**Why this works**: e-commerce prices are highly stable from day to day.
The last known price is the strongest predictor for the current price.

### Anchor Calibration

The 100 manually collected anchor prices reveal the day-level price shift:

```python
anchor_ratio = median(anchor_true_price / anchor_last_price)
```

- `ratio > 1.0` -> prices shifted up on this day (e.g. promotions ended)
- `ratio < 1.0` -> prices shifted down (e.g. flash sale active)
- `ratio = 1.0` -> no shift detected, prices are stable

The ratio is applied to all predictions for that day.

---

## Validation Results

| Model | MAE | MAPE |
|---|---|---|
| Tier 1 - Global Model (no calibration) | Rp 46,269 | 0.20% |
| Tier 1 - Global Model + anchor calibration | Rp 47,124 | 0.20% |
| **Tier 2 - Last Price + anchor calibration** | **Rp 18,005** | **0.06%** |

**Tier 2 is 2.6x more accurate** than Tier 1 because product prices
are sticky and rarely change from one day to the next.

---

## Key Findings

**1. Price stickiness is the dominant signal**
`model_last_price` is by far the most predictive feature. Most products
have identical prices across consecutive days, making "yesterday's price"
an extremely strong predictor.

**2. Anchor calibration catches day-level shifts**
On stable days, the ratio = 1.0 and calibration has no effect.
On days with platform-wide events like flash sales or promotions ending,
the ratio would deviate and calibration becomes critical.

**3. Anchor calibration slightly worsened Tier 1 on this validation day**
Since the ratio = 1.0 exactly, the small noise from estimating the ratio
using only 100 samples caused a slight increase in MAE (Rp 46,269 -> Rp 47,124).
This is expected on stable days and highlights that calibration is most
valuable when there is a real price shift to detect.

**4. Tier 1 is still essential as a cold-start fallback**
Only 2 products in the 3-day test set had no training history.
Tier 1 handles these cases where Tier 2 has no `model_last_price` to use.

**5. Category is a strong price signal**
Median prices vary up to 120x across categories, from Rp 3.5M to Rp 419.9M.
`cat_id` is one of the most important features for Tier 1.

**6. Data anomaly in early January**
Very few rows in early January create an outlier spike in the daily
median price chart. This is a data collection artefact, not a real
price movement.

---

## Limitations

- `stock` and `normal_stock` were dropped (98.8% missing). If available, could improve Tier 1.
- Anchor calibration uses a single global ratio per day. A per-category ratio could be more precise.
- Tier 2 cannot handle truly new products (cold start). Tier 1 is the fallback for these cases.
- Non-anchor rows in the test file have very few filled columns, limiting Tier 1's effectiveness on the test set.
