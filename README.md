# Credit Scoring Model

CodeAlpha Machine Learning internship, **Task 1**.

Predict whether a loan applicant is a **good (1)** or **bad (0)** credit risk, and choose the model and decision threshold that suit a bank.

## Dataset

Kaggle dataset *Creditability - German Credit Data* (`german.csv`, `;`-separated).
1000 customers, 20 features (account balance, loan duration and amount, payment history, savings, employment length, age, purpose, ...) and the target `Creditability`.

- 700 good, 300 bad (70/30, imbalanced), no missing values.
- Most columns are integer codes of categories (for example `Purpose` has 10 codes), not real numbers. The dataset ships without a data dictionary, so this README does not interpret individual codes.
- The data file is not included in this repo. Attach the dataset to a Kaggle notebook, or download it and change `path` in the first cell.

## Approach

1. **Split:** 80/20 stratified train/test (800 / 200), `random_state=42`.
2. **Preprocessing** (fit on the training set only): `StandardScaler` for the 3 numeric columns (duration, amount, age) and `OneHotEncoder` for the 17 code columns, giving 71 features. Everything sits inside a scikit-learn `Pipeline`, so cross-validation does not leak.
3. **Imbalance:** `class_weight="balanced"` where the model supports it.
4. **Models:** Logistic Regression, Decision Tree, Random Forest, Gradient Boosting.
5. **Threshold:** a model outputs P(good). A customer is approved only if P(good) >= threshold. The threshold is chosen from out-of-fold predictions on the **training set**, never from the test set.
6. **Metrics:** precision, recall, F1, ROC-AUC, and a bank-cost view.

Accuracy alone is misleading here: approving everyone scores 70% accuracy and catches no bad customer. The metric that matters is **recall on the bad class** (bad customers caught), balanced against how many good customers are rejected.

## Results

### 5-fold CV on the training set (threshold 0.5)

| Model | Accuracy | Recall (bad) | ROC-AUC |
|---|---|---|---|
| Logistic Regression | 0.698 | 0.650 | 0.763 |
| Decision Tree (depth 5) | 0.660 | 0.708 | 0.696 |
| Random Forest | 0.751 | 0.317 | 0.775 |

Random Forest has the best accuracy but catches only about a third of the bad customers at the default threshold.

### Four algorithms, threshold chosen to minimise bank cost, evaluated on the test set

Bank cost follows the German Credit convention: approving a bad customer costs 5, rejecting a good one costs 1. This 5:1 ratio is an assumption taken from the original dataset, not from a real bank. Approving everyone costs 1.5 per customer.

| Model | CV AUC | Threshold | Test AUC | Recall (bad) | Good kept | Cost / customer |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.763 ± 0.036 | 0.70 | 0.828 | 0.900 | 0.486 | 0.510 |
| Decision Tree | 0.696 ± 0.039 | 0.65 | 0.689 | 0.750 | 0.571 | 0.675 |
| Random Forest | 0.775 ± 0.036 | 0.75 | 0.827 | 0.867 | 0.600 | 0.480 |
| Gradient Boosting | 0.763 ± 0.016 | 0.90 | 0.818 | 0.933 | 0.300 | 0.590 |

Logistic Regression, Random Forest and Gradient Boosting are statistically close. The Decision Tree is clearly weaker. With only 60 bad customers in the test set, the 0.03 cost gap between Random Forest and Logistic Regression is about one bad customer.

### Chosen model: Logistic Regression, threshold 0.70

Chosen for similar performance to the best model plus the clearest explanation of each decision (coefficients per feature), which matters when a bank must justify a rejection.

Test set (200 customers):

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| bad (0) | 0.43 | 0.90 | 0.58 | 60 |
| good (1) | 0.92 | 0.49 | 0.64 | 140 |

- ROC-AUC **0.828**, accuracy 0.61, bank cost 0.51 per customer (vs 1.5 for approving everyone).
- Confusion matrix `[[54, 6], [72, 68]]`: 54 of 60 bad customers caught, 6 approved by mistake, 72 good customers rejected by mistake.
- Strongest signs in the model: account balance, savings/stocks, length of employment and previous payment status push towards "good" at their highest levels and towards "bad" at their lowest.

## Limitations

- The test set is small (60 bad customers), so single numbers move noticeably between splits. Read results as ranges, not exact values.
- The threshold depends on the 5:1 cost assumption. A different cost ratio needs a different threshold.
- A high threshold means many good customers are rejected (about half here). That is the price of catching 90% of bad ones.
- The data is old and small, and `Foreign_Worker`, `Sex_Marital_Status` and similar features appear in it. A real credit model must be checked for fairness and legal compliance before use.

## How to run

```bash
pip install -r requirements.txt
jupyter notebook credit_scoring.ipynb
```

Use the saved model:

```python
import json, joblib
import pandas as pd

model = joblib.load("credit_model.joblib")
threshold = json.load(open("threshold.json"))["threshold"]

df = pd.read_csv("german.csv", sep=";")
X = df.drop(columns="Creditability")
p_good = model.predict_proba(X)[:, 1]
approve = p_good >= threshold
```

The model was saved with scikit-learn 1.6.1; load it with the same version.

## Files

- `credit_scoring.ipynb`: full analysis with outputs
- `credit_model.joblib`: fitted pipeline (preprocessing + Logistic Regression)
- `threshold.json`: decision threshold (0.70)
- `requirements.txt`: dependencies
