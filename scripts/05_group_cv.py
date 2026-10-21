import numpy as np, pandas as pd
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.model_selection import GroupKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier

d = pd.read_pickle("data/matrix.pkl")
X = d.drop(columns=["y_rif", "y_inh"]).astype("float32")

pcs = PCA(n_components=10, random_state=0).fit_transform(X.values)
groups = KMeans(n_clusters=8, n_init=10, random_state=0).fit_predict(pcs)
print("cluster sizes:", np.bincount(groups))

models = {
    "LogReg": LogisticRegression(max_iter=2000),
    "XGBoost": XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.1,
                             n_jobs=-1, eval_metric="logloss"),
}
gkf = GroupKFold(n_splits=4)
rows = []
for lab, drug in [("y_rif", "Rifampicin"), ("y_inh", "Isoniazid")]:
    y = d[lab].values
    for name, mdl in models.items():
        aucs = []
        for tr, te in gkf.split(X, y, groups):
            if len(set(y[te])) < 2:
                continue
            mdl.fit(X.iloc[tr], y[tr])
            aucs.append(roc_auc_score(y[te], mdl.predict_proba(X.iloc[te])[:, 1]))
        rows.append((drug, name, np.mean(aucs), np.min(aucs), len(aucs)))
        print(f"{drug:11s} {name:8s} lineage-held-out AUC {np.mean(aucs):.3f} (worst fold {np.min(aucs):.3f}, {len(aucs)} folds)")

pd.DataFrame(rows, columns=["drug", "model", "auc_mean", "auc_worst", "folds"]).to_csv("results/group_cv.csv", index=False)