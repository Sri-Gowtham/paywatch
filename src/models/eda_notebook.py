"""PayWatch — build and EXECUTE the EDA notebook (roadmap step notebooks/01_eda.ipynb) on Kaggle.

The notebook is generated with nbformat and executed with nbclient on the raw IEEE-CIS data, so every number and
plot in it is computed, not typed. Output: /kaggle/working/01_eda.ipynb
"""

import glob
import os
import subprocess
import sys

try:
    import nbclient  # noqa: F401
    import nbformat  # noqa: F401
except ImportError:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "nbformat", "nbclient", "ipykernel"], check=False)

import nbformat  # noqa: E402
from nbclient import NotebookClient  # noqa: E402

RAW = os.path.dirname(glob.glob("/kaggle/input/**/train_transaction.csv", recursive=True)[0])
md, code = nbformat.v4.new_markdown_cell, nbformat.v4.new_code_cell

CELLS = [
    md("# PayWatch: exploratory data analysis\n\nIEEE-CIS Fraud Detection (`train_transaction.csv` joined to `train_identity.csv`). "
       "Every number below is computed when the notebook runs on Kaggle. The goal is to justify the modelling choices: "
       "metric, split, features and missing-value handling."),
    code('''import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")
%matplotlib inline
pd.set_option("display.width", 140)
RAW = "<RAW>"
tx = pd.read_csv(f"{RAW}/train_transaction.csv")
idn = pd.read_csv(f"{RAW}/train_identity.csv")
df = tx.merge(idn, on="TransactionID", how="left", indicator="has_identity")
df["has_identity"] = (df["has_identity"] == "both").astype(int)
df["day"] = df["TransactionDT"] / 86400
df["hour"] = (df["TransactionDT"] // 3600) % 24
fraud_rate = df.isFraud.mean()
print("rows x columns:", df.shape, "| time span (days):", round(df.day.max() - df.day.min(), 1))
print("fraud rows:", int(df.isFraud.sum()), "| fraud rate:", round(fraud_rate, 4))'''.replace("<RAW>", RAW)),
    md("## 1. Class balance: why accuracy is the wrong metric"),
    code('''print(f"A model that always predicts 'not fraud' is {1 - fraud_rate:.1%} accurate and catches 0 fraud.")
print("-> evaluate with PR-AUC and recall at a fixed false-positive rate, not accuracy.")'''),
    md("## 2. Time structure: why the split must be time-ordered"),
    code('''weekly = df.groupby((df.day // 7).astype(int)).agg(rows=("isFraud", "size"), fraud_rate=("isFraud", "mean"),
                                                 mean_amount=("TransactionAmt", "mean"))
display(weekly.round(4))
fig, ax = plt.subplots(1, 3, figsize=(15, 3.5))
weekly.rows.plot(ax=ax[0], marker="o", title="transactions per week")
weekly.fraud_rate.plot(ax=ax[1], marker="o", title="fraud rate per week")
hourly = df.groupby("hour").isFraud.mean()
hourly.plot(kind="bar", ax=ax[2], title="fraud rate by hour of day (TransactionDT offset hour)")
plt.tight_layout(); plt.show()
print("fraud-rate range across weeks:", round(weekly.fraud_rate.min(), 4), "to", round(weekly.fraud_rate.max(), 4))'''),
    md("## 3. Amounts"),
    code('''fig, ax = plt.subplots(1, 2, figsize=(13, 3.5))
for label, g in df.groupby("isFraud"):
    ax[0].hist(np.log10(g.TransactionAmt.clip(lower=0.01)), bins=60, alpha=0.5, density=True, label="fraud" if label else "legit")
ax[0].set_title("log10(amount) by class"); ax[0].legend()
df["amount_decile"] = pd.qcut(df.TransactionAmt, 10, duplicates="drop")
df.groupby("amount_decile").isFraud.mean().plot(kind="bar", ax=ax[1], title="fraud rate by amount decile")
plt.tight_layout(); plt.show()
print(df.groupby("isFraud").TransactionAmt.describe().round(2))'''),
    md("## 4. Risk by category"),
    code('''for col in ["ProductCD", "card4", "card6", "DeviceType", "has_identity"]:
    t = df.groupby(col, dropna=False).isFraud.agg(rows="size", fraud_rate="mean").sort_values("rows", ascending=False)
    print(f"\\n{col}"); print(t.head(8).round(4).to_string())
top = df.P_emaildomain.value_counts().head(12).index
print("\\nP_emaildomain (12 largest)")
print(df[df.P_emaildomain.isin(top)].groupby("P_emaildomain").isFraud.agg(rows="size", fraud_rate="mean")
      .sort_values("fraud_rate", ascending=False).round(4).to_string())
identity_cov = df.has_identity.mean()
print("\\nshare of transactions with identity information:", round(identity_cov, 3))'''),
    md("## 5. Missing values"),
    code('''miss = df.isna().mean().sort_values(ascending=False)
print("columns:", df.shape[1], "| >90% missing:", int((miss > 0.9).sum()), "| >50% missing:", int((miss > 0.5).sum()))
miss.head(25).plot(kind="barh", figsize=(7, 6), title="25 columns with the most missing values")
plt.gca().invert_yaxis(); plt.tight_layout(); plt.show()
d1 = df.D1.isna()
print("fraud rate when D1 is missing / present:", round(df[d1].isFraud.mean(), 4), "/", round(df[~d1].isFraud.mean(), 4))
print("-> missingness itself carries signal, so missing values are filled with a sentinel instead of dropped.")'''),
    md("## 6. Entity structure: why lagged fraud history can help\n\n"
       "No real user id exists, so a card composite (`card1`, `card2`, `card3`, `card5`) stands in for the user."),
    code('''d = df.sort_values("TransactionDT")[["TransactionDT", "isFraud", "card1", "card2", "card3", "card5"]].copy()
d["card_key"] = d[["card1", "card2", "card3", "card5"]].astype(str).agg("_".join, axis=1)
d["n_before"] = d.groupby("card_key").cumcount()
fraud = d[d.isFraud == 1].copy()
known_card_share = (fraud.n_before > 0).mean()
fraud["first_fraud_t"] = fraud.groupby("card_key").TransactionDT.transform("min")
repeat7_share = ((fraud.TransactionDT - fraud.first_fraud_t) >= 7 * 86400).mean()
print("distinct card composites:", d.card_key.nunique())
print("share of fraud on a card that had an earlier transaction:", round(known_card_share, 3))
print("share of fraud on a card whose FIRST fraud was at least 7 days earlier:", round(repeat7_share, 3))'''),
    md("## 7. What this means for the model"),
    code('''print(f"1. Fraud is rare ({fraud_rate:.1%}): use PR-AUC / recall at fixed FPR and class weights, never accuracy.")
print(f"2. Fraud rate moves from {weekly.fraud_rate.min():.1%} to {weekly.fraud_rate.max():.1%} week to week: split by time, not at random.")
print(f"3. {known_card_share:.0%} of fraud is on a card seen before and {repeat7_share:.0%} follows an earlier fraud on the same card "
      "by 7+ days: lagged entity fraud history is worth building.")
print(f"4. Only {identity_cov:.0%} of rows have identity data and many columns are mostly missing: keep missingness as a signal.")
print("5. A reconstructed user id is a proxy (no real UPI ids): say so wherever the features are quoted.")'''),
]

nb = nbformat.v4.new_notebook()
nb.cells = CELLS
nb.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
NotebookClient(nb, timeout=1800, kernel_name="python3", resources={"metadata": {"path": "/kaggle/working"}}).execute()
nbformat.write(nb, "/kaggle/working/01_eda.ipynb")
size = os.path.getsize("/kaggle/working/01_eda.ipynb")
print(f"wrote 01_eda.ipynb ({size:,} bytes, {len(nb.cells)} cells)")
for c in nb.cells:
    if c.cell_type == "code":
        for out in c.get("outputs", []):
            if out.get("output_type") == "stream":
                print(out["text"][:900])
