# Raw IEEE-CIS data (not included in this repository)

The IEEE-CIS Fraud Detection dataset is distributed by Kaggle under
competition-specific terms that do not permit redistribution. To reproduce
Section 5.5 of the paper:

1. Go to the competition page:
   https://www.kaggle.com/competitions/ieee-fraud-detection/data
2. Sign in / accept the competition rules.
3. Download `train_transaction.csv` (part of `train_transaction.csv.zip`).
4. Place the extracted file here, so the path is:
   `data/raw/train_transaction.csv`
5. Run `code/section5.5_ieee_cis/01_load_and_clean.py` to regenerate
   `data/fraud_clean.csv` and reproduce all Section 5.5 results.

Note: `card4`, `card6`, `email domains`, `M1`-`M9`, and other categorical
columns are dropped by the cleaning script (see the docstring in
`01_load_and_clean.py` for the exact rationale). Only the resulting numeric
feature matrix is used downstream.
