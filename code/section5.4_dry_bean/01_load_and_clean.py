# -*- coding: utf-8 -*-
"""
Load and clean the real UCI Dry Bean dataset. This dataset is used for the
external-validity check of the "Blind Spot Map" simulation results in the
paper.

Source: Koklu, M. and Ozkan, I.A., (2020), "Multiclass Classification of Dry
Beans Using Computer Vision and Machine Learning Techniques", Computers and
Electronics in Agriculture, 174, 105507. https://doi.org/10.24432/C50S4B

Note: this script expects the raw file "Dry_Bean_Dataset_RAW.xlsx" (as
distributed by the UCI Machine Learning Repository) under data/. The cleaned
output (data/dry_bean_clean.csv) is already included in this repository, so
running this script is only necessary to regenerate it from scratch.
"""
import pandas as pd

import os
# repo/code/section5.4_dry_bean/script.py -> repo root is 2 levels up
BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SRC = os.path.join(BASE, "data", "Dry_Bean_Dataset_RAW.xlsx")
OUT = os.path.join(BASE, "data", "dry_bean_clean.csv")

df = pd.read_excel(SRC)
print("Raw shape:", df.shape)
print("Missing values:", int(df.isna().sum().sum()))

n_dup = int(df.duplicated().sum())
print("Duplicate rows:", n_dup)
df = df.drop_duplicates().reset_index(drop=True)
print("Shape after removing duplicates:", df.shape)

print("\nClass distribution:")
print(df["Class"].value_counts())

df.to_csv(OUT, index=False)
print(f"\nSaved: {OUT}")
