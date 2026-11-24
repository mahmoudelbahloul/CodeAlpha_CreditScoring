"""Score loan applicants with the saved credit model.

    python predict.py applicants.csv

The CSV must have the 20 feature columns of the German Credit data (the same names as
german.csv, ';' or ',' separated). A 'Creditability' column, if present, is ignored.
Output: P(good) per applicant and the decision at the threshold in threshold.json.
The model was saved with scikit-learn 1.6.1.
"""
import argparse, json, os
import joblib
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csv", help="CSV with the 20 feature columns")
    ap.add_argument("--out", help="optionally write the results to this CSV")
    a = ap.parse_args()

    model = joblib.load(os.path.join(HERE, "credit_model.joblib"))
    threshold = json.load(open(os.path.join(HERE, "threshold.json")))["threshold"]
    df = pd.read_csv(a.csv, sep=None, engine="python")  # detects ';' or ','
    cols = list(model.feature_names_in_)
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise SystemExit(f"missing columns: {missing[:5]}{' ...' if len(missing) > 5 else ''}")

    p_good = model.predict_proba(df[cols])[:, 1]
    out = pd.DataFrame({"p_good": p_good.round(3), "decision": ["approve" if p >= threshold else "reject" for p in p_good]})
    print(f"threshold = {threshold} (approve only if P(good) >= threshold)")
    print(out.to_string())
    if a.out:
        out.to_csv(a.out, index=False)


if __name__ == "__main__":
    main()
