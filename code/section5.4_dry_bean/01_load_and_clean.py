# -*- coding: utf-8 -*-
"""
بارگذاری و پاکسازی دیتاست واقعی Dry Bean (UCI Machine Learning Repository).
این دیتاست برای اعتبارسنجی خارجی (External Validity) نتایج شبیه‌سازی مقاله
"Blind Spot Map" استفاده می‌شود.

منبع: Koklu, M. and Ozkan, I.A., (2020), "Multiclass Classification of Dry Beans
Using Computer Vision and Machine Learning Techniques", Computers and Electronics
in Agriculture, 174, 105507.
"""
import pandas as pd

import os
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # blindspot_real/
SRC = os.path.join(BASE, "data", "Dry_Bean_Dataset_RAW.xlsx")
OUT = os.path.join(BASE, "data", "dry_bean_clean.csv")

df = pd.read_excel(SRC)
print("ابعاد خام:", df.shape)
print("مقادیر گمشده:", int(df.isna().sum().sum()))

n_dup = int(df.duplicated().sum())
print("ردیف‌های تکراری:", n_dup)
df = df.drop_duplicates().reset_index(drop=True)
print("ابعاد پس از حذف تکراری‌ها:", df.shape)

print("\nتوزیع کلاس‌ها:")
print(df["Class"].value_counts())

df.to_csv(OUT, index=False)
print(f"\nذخیره شد: {OUT}")
