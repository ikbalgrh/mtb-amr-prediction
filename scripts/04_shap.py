import numpy as np, pandas as pd, shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from xgboost import XGBClassifier

d = pd.read_pickle("data/matrix.pkl")
X = d.drop(columns=["y_rif", "y_inh"]).astype("float32")

targets = {"y_rif": "761155_C_T",    # rpoB S450L
           "y_inh": "2155168_C_G"}   # katG S315T

for lab, target in targets.items():
    model = XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.1, eval_metric="logloss")
    model.fit(X, d[lab])
    sv = shap.TreeExplainer(model).shap_values(X)
    imp = pd.Series(np.abs(sv).mean(0), index=X.columns).sort_values(ascending=False)

    print(f"\n=== {lab}: top 15 features ===")
    print(imp.head(15))
    if target in imp.index:
        print(f"{target} is ranked #{list(imp.index).index(target) + 1}")
    else:
        print(f"{target} not found, search for its position:", [c for c in imp.index if c.startswith(target.split('_')[0])])

    imp.head(50).to_csv(f"results/shap_top50_{lab}.csv")
    shap.summary_plot(sv, X, max_display=15, show=False)
    plt.tight_layout()
    plt.savefig(f"results/shap_{lab}.png", dpi=200)
    plt.close()
print("\nsaved plots in results/")