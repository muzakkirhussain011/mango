import json, pandas as pd, matplotlib.pyplot as plt, seaborn as sns, os
H = json.load(open("reports/history.json"))
df = pd.DataFrame(H)
plt.figure()
df["metrics"].apply(lambda m: m.get("accuracy", None)).plot()
plt.title("Global Accuracy over Rounds")
plt.savefig("paper/figs/accuracy.png", dpi=200, bbox_inches="tight")
