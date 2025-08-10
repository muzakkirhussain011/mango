import json, pandas as pd, os
H = json.load(open("reports/history.json"))
df = pd.DataFrame([h["metrics"] for h in H])
os.makedirs("paper/tables", exist_ok=True)
df.to_csv("paper/tables/summary.csv", index=False)
