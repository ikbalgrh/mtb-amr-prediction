import numpy as np, pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, confusion_matrix
from xgboost import XGBClassifier

d = pd.read_pickle("data/matrix.pkl")
X = d.drop(columns=["y_rif", "y_inh"]).astype("float32")

models = {
    "LogReg": LogisticRegression(max_iter=2000),
    "RandomForest": RandomForestClassifier(n_estimators=300, n_jobs=-1, random_state=0),
    "XGBoost": XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.1,
                             n_jobs=-1, eval_metric="logloss"),
}
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)

rows = []
for lab, drug in [("y_rif", "Rifampicin"), ("y_inh", "Isoniazid")]:
    y = d[lab].values
    for name, mdl in models.items():
        aucs, sens, specs = [], [], []
        for tr, te in cv.split(X, y):
            mdl.fit(X.iloc[tr], y[tr])
            p = mdl.predict_proba(X.iloc[te])[:, 1]
            tn, fp, fn, tp = confusion_matrix(y[te], p >= 0.5).ravel()
            aucs.append(roc_auc_score(y[te], p))
            sens.append(tp / (tp + fn))
            specs.append(tn / (tn + fp))
        rows.append((drug, name, np.mean(aucs), np.std(aucs), np.mean(sens), np.mean(specs)))
        print(f"{drug:11s} {name:13s} AUC {np.mean(aucs):.3f} ± {np.std(aucs):.3f} | "
              f"sens {np.mean(sens):.3f} | spec {np.mean(specs):.3f}")

pd.DataFrame(rows, columns=["drug", "model", "auc_mean", "auc_sd", "sensitivity", "specificity"]) \
  .to_csv("results/cv_results.csv", index=False)
print("saved results/cv_results.csv")