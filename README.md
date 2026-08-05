# Cardiovascular Risk Classifier — PCLP3 Part I

**Popescu Petruț-Alin | Grupa 312 CA**

Binary classification task: predict whether a patient has **high cardiovascular risk** (Yes / No).  
Risk factor thresholds based on [WHO Cardiovascular Diseases](https://www.who.int/health-topics/cardiovascular-diseases).

---

## Project Structure

| File | Role |
|---|---|
| `data_frame_generation.py` | Synthetic dataset generation |
| `data_eda.py` | Exploratory data analysis |
| `train_mode.py` | Model training & evaluation |
| `graphic_design.py` | Gradio GUI (Bonus 4.1) |
| `data_training.csv` | 700-instance training split |
| `data_testing.csv` | 300-instance test split |

---

## 1. Dataset Generation

1000 synthetic instances built with NumPy using realistic distributions:

- **Integer:** Age (16–90), Pulse (60–120), Colesterol (100–300)
- **Float:** Corporal body fat ~N(26, 5), Sleeped hours/week ~U(3, 10)×7
- **Categorical:** Gender, Smoker (25% Yes), Family history (20% Yes)

**Target label** derived from a composite risk score (threshold = 5):

```python
base_risk = (
    (age > 60) * 2.0 + (body_fat > 29) * 1.5 +
    (pulse > 85) * 1.25 + (colesterol > 250) * 2.5 +
    (smoker == 'Yes') * 3.0 + (fam_hist == 'Yes') * 2.0 +
    (sleep_h < 35) * 1.0
)
risk = np.where(base_risk + noise > 5, 'Yes', 'No')
```

**Deliberate noise:** measurement errors on 10 Pulse (200–270 bpm) and 10 body fat (60–84%) instances; 7 young (<28 y.o., non-smokers) labelled Yes; 7 elderly (>70 y.o., smokers) labelled No; ~5% NaN on Colesterol and body fat.

**Split:** stratified 70/30 via sklearn.

---

## 2. EDA Highlights

- **Numeric distributions** stable across train/test; body fat follows a visible normal, Age and Pulse are uniform.
- **Categorical:** gender 50/50, smokers ~25%, family history ~20%; target mildly imbalanced (more No).
- **Outliers** (Pulse & body fat) clearly visible in boxplots — clipping or removal recommended.
- **Correlation matrix:** all inter-feature correlations near 0 (features generated independently, no redundancy).
- **Violin plots vs target:** high-risk class skews older; cholesterol >250 and pulse >85 bpm more frequent in Yes class.

---

## 3. Model

**Random Forest Classifier** — 100 trees, `random_state=53`

Chosen for robustness to outliers and native handling of mixed feature types without mandatory normalisation.

**Preprocessing:** mean/mode imputation for missing values + manual label encoding (`Masculin→1`, `Feminin→0`, `Yes→1`, `No→0`).

**Results:** accuracy ~85–90%; low-risk class predicted more precisely due to mild class imbalance.

```
Confusion matrix (test set, 300 instances):
              Predicted No   Predicted Yes
Actual No         207              4
Actual Yes         16             73
```

---

## 4. Bonus — Gradio GUI

```bash
pip install gradio
python graphic_design.py
# opens at http://127.0.0.1:7860
```

Input sliders/radio buttons for: Age, Gender, Body fat, Pulse, Cholesterol, Smoker, Sleep hours/week, Family history.  
Output: **High** or **Small cardiovascular risk**.
