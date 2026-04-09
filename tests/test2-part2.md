---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.19.1
kernelspec:
  name: python3
  display_name: Python 3
  language: python
---

# Test 2, Part 2 — Applied Classification (Bonus)

+++

## Introduction

This is the optional take-home bonus for Test 2. It is worth up to 5 points added to your Test 2 score.

You are given a dataset with a brief exploratory analysis. Your job is to build a classification pipeline, interpret the results in a business context, and make a cost-based recommendation. The emphasis is entirely on interpretation - working code that produces results without explanation will receive no credit.

### Rules

- Open book, open notes, open documentation
- GenAI tools allowed with disclosure (same policy as homework)
- Individual work. No collaboration of any kind.
- Due: Sunday, April 13

### Scoring (5 points)

| Component | Points |
|-----------|-------:|
| Confusion matrix interpretation | 1 |
| Cost-based tradeoff analysis | 2 |
| Recommendation with justification | 2 |
| **Total** | **5** |

All points are for interpretation. There are no points for code.

### Time Estimate

This should take under 2 hours. If you are spending significantly more time, you are overcomplicating things. The modeling portion requires fewer than 30 lines of Python.

+++

## The Problem

The Paranormal Property Assessment Bureau (PPAB) receives reports of potentially haunted houses from concerned homeowners and prospective buyers. Each report triggers an on-site inspection by a certified paranormal investigator. The bureau wants to build a model to prioritize which reports to inspect first, based on data collected during the initial phone intake.

### The Costs

- Each **inspection** costs ＄200 in investigator time and equipment
- A **missed haunting** (genuine report classified as false alarm, i.e., a false negative) results in an average ＄2,000 liability claim when the new occupants discover the haunting themselves
- A **false dispatch** (false alarm classified as genuine, i.e., a false positive) wastes the ＄200 inspection cost but causes no further harm

The bureau currently inspects every report. They want to know: can a model help them skip the obvious false alarms while still catching the real hauntings?

### Data Dictionary

| Feature | Description | Units |
|---------|-------------|-------|
| `emf_reading` | Electromagnetic field reading during phone call | mV |
| `house_age` | Age of the house | years |
| `prev_reports` | Number of previous paranormal reports at address | count |
| `neighbor_reports` | Reports from neighboring properties | count |
| `cold_spots` | Cold spots reported by caller | count |
| `flickering` | Electrical flickering events per day | count/day |
| `pet_behavior` | Pet behavioral anomaly score | 0-10 scale |
| `foundation_cracks` | Visible foundation cracks | count |
| `basement` | Has a basement | 0 or 1 |
| `tree_proximity` | Distance to nearest large tree | meters |
| `wind_exposure` | Wind exposure index | composite score |
| `pipes_age` | Age of plumbing | years |
| **`haunted`** | **Confirmed haunting after inspection (target)** | **0 or 1** |

+++

## Setup

```{code-cell} ipython3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (classification_report, confusion_matrix,
                             ConfusionMatrixDisplay, precision_score, recall_score)

# Add any additional imports you need
```

```{code-cell} ipython3
url = "https://raw.githubusercontent.com/olearydj/INSY7120/refs/heads/main/tests/test2-part2-data.csv"
df = pd.read_csv(url)
print(f"Shape: {df.shape}")
df.head()
```

## Exploratory Analysis

```{code-cell} ipython3
# Class distribution
print("Class distribution:")
print(df["haunted"].value_counts())
print(f"\nHaunting rate: {df['haunted'].mean():.1%}")
```

```{code-cell} ipython3
# Correlation with target
target_corr = df.corr()["haunted"].drop("haunted").sort_values(key=abs, ascending=False)
print("Correlation with haunted:")
print(target_corr.round(3).to_string())
```

Note the class imbalance. Consider what this means for your choice of metrics and modeling approach.

+++

## Your Work

Build a classification model and answer the interpretation prompts below. Your submission should include:

1. **A classification pipeline** with appropriate preprocessing and a classifier. Use stratified cross-validation to evaluate. Report precision, recall, and F1 for the haunted class (not just accuracy).

2. **A confusion matrix** on the test set, using your default model (threshold = 0.5).

3. **An improvement** — apply class weights, threshold adjustment, or another technique to improve the model's ability to catch real hauntings. Generate a second confusion matrix for comparison.

The code is the means, not the end. Keep it simple and focus your effort on the interpretation prompts.

+++

*Your modeling code here.*

+++

## Interpretation Prompts

Answer each prompt using specific numbers from your results. Generic answers that could apply to any dataset will receive no credit.

### 1. Read Your Confusion Matrix (1 pt)

Look at your default model's confusion matrix (before any adjustment). How many false negatives did the model produce? In the PPAB context, what does each false negative represent, and what is the total financial cost of those missed hauntings?

---

*Your response here.*

---

+++

### 2. Evaluate the Tradeoff (2 pts)

After applying your improvement (class weights, threshold change, etc.), compare the before and after:

- How did precision and recall for the haunted class change?
- Calculate the total cost of all errors (FP + FN) for both the default and improved models. Use the PPAB cost structure: ＄200 per false positive, ＄2,000 per false negative.
- Was the adjustment worth it? Why or why not?

---

*Your response here.*

---

+++

### 3. Recommendation (2 pts)

Based on your analysis, what model configuration would you recommend the PPAB deploy? State the specific threshold or class weight setting you recommend, and justify it with:

- The expected number of false positives and false negatives per 100 inspections
- The expected total cost of errors per 100 inspections
- Why this is a better operating point than the default

---

*Your response here.*

---
