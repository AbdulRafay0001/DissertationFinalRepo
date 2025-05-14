import pandas as pd
import numpy as np

# 1. Load
df = pd.read_excel("flat_results_with_items.xlsx")

# 2. Parse scores & items
df["score_list"] = df["scores"].str.split(",").apply(lambda L: [float(x) for x in L])
df["item_list"]  = df["items"].str.split("|")

# 3. ColdStart@3 & Recall@3 (cold rows only)
cold = df[df.prompt_type=="cold"].copy()
cold["mean_rel"] = cold["score_list"].apply(np.mean)
cold["hit"]      = cold["score_list"].apply(lambda L: max(L)>0)

cold_summary = (
    cold.groupby(["domain","pipeline"])
        .agg(
            ColdStart3 = ("mean_rel","mean"),
            Recall3    = ("hit",     "mean")
        )
        .round(3)
        .reset_index()
)

# 4. SynHit@3 (synonym rows, grouped by pair_id)
syn = df[df.prompt_type=="synonym"].copy()
def synhit(group):
    # group has exactly 2 rows per (domain, pipeline, pair_id)
    a, b = group["item_list"].iloc[0], group["item_list"].iloc[1]
    return len(set(a).intersection(b)) / 3

syn_summary = (
    syn.groupby(["domain","pipeline","pair_id"])
       .apply(synhit)
       .reset_index(name="SynHit3")
       .groupby(["domain","pipeline"])
       .SynHit3.mean()
       .round(3)
       .reset_index()
)

# 5. Merge all metrics
metrics = (
    cold_summary
    .merge(syn_summary, on=["domain","pipeline"])
)

# 6. Save and print
metrics.to_excel("metrics_summary.xlsx", index=False)
print(metrics)
