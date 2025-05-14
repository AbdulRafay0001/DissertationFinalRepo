import os
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Ensure output folder exists
out_dir = "figs"
os.makedirs(out_dir, exist_ok=True)

# Config
pipelines = ["P-TF", "P-AUTO", "P-MAN", "P-HYB"]
domains   = ["movies", "books"]

# 1. Load summary & flat data
metrics = pd.read_excel("metrics_summary.xlsx")
flat    = pd.read_excel("flat_results_with_items.xlsx")
flat["score_list"] = flat["scores"].str.split(",").apply(lambda L: [float(x) for x in L])
flat["mean_rel"]   = flat["score_list"].apply(np.mean)

# 2. Core bar‐charts (ColdStart, Recall, SynHit)
x = np.arange(len(pipelines))
width = 0.35

def bar_chart(df_pivot, metric, ylabel):
    fig, ax = plt.subplots()
    for i, domain in enumerate(domains):
        ax.bar(x + (i-0.5)*width, df_pivot[domain], width, label=domain.capitalize())
    ax.set_xticks(x)
    ax.set_xticklabels(pipelines)
    ax.set_ylabel(ylabel)
    ax.set_title(f"{metric} by Pipeline and Domain")
    ax.legend()
    fig.savefig(f"{out_dir}/{metric}.png", bbox_inches="tight")
    plt.close(fig)

cold_df   = metrics.pivot(index="pipeline", columns="domain", values="ColdStart3").reindex(pipelines)
recall_df = metrics.pivot(index="pipeline", columns="domain", values="Recall3").reindex(pipelines)
synhit_df = metrics.pivot(index="pipeline", columns="domain", values="SynHit3").reindex(pipelines)

bar_chart(cold_df,   "ColdStart3", "ColdStart@3")
bar_chart(recall_df, "Recall3",    "Recall@3")
bar_chart(synhit_df, "SynHit3",    "SynHit@3")

# 3. Boxplots of ColdStart distribution
for domain in domains:
    subset = flat[(flat.prompt_type=="cold") & (flat.domain==domain)]
    data = [subset[subset.pipeline==p]["mean_rel"] for p in pipelines]
    fig, ax = plt.subplots()
    ax.boxplot(data)
    ax.set_xticklabels(pipelines)
    ax.set_title(f"ColdStart@3 Distribution — {domain.capitalize()}")
    ax.set_ylabel("Per-prompt mean relevance")
    fig.savefig(f"{out_dir}/box_cold_{domain}.png", bbox_inches="tight")
    plt.close(fig)

# 4. Radar (spider) charts
categories = ["ColdStart3","Recall3","SynHit3"]
N = len(categories)
angles = [n/float(N)*2*math.pi for n in range(N)]
angles += angles[:1]

metrics_idx = metrics.set_index(["domain","pipeline"])
for domain in domains:
    df_dom = metrics_idx.loc[domain]
    fig, ax = plt.subplots(subplot_kw=dict(polar=True))
    for p in pipelines:
        vals = df_dom.loc[p, categories].tolist()
        vals += vals[:1]
        ax.plot(angles, vals, label=p)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories)
    ax.set_title(f"Performance Radar — {domain.capitalize()}")
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))
    fig.savefig(f"{out_dir}/radar_{domain}.png", bbox_inches="tight")
    plt.close(fig)

# 5. Correlation‐matrix heatmap with colorbar
corr = metrics[["ColdStart3","Recall3","SynHit3"]].corr()
fig, ax = plt.subplots()
im = ax.imshow(corr.values, vmin=-1, vmax=1)
ax.set_xticks(range(3)); ax.set_yticks(range(3))
ax.set_xticklabels(corr.columns); ax.set_yticklabels(corr.index)
# Add colorbar (key)
cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
cbar.set_label("Pearson r", rotation=270, labelpad=15)
# Annotate cells
for i in range(3):
    for j in range(3):
        ax.text(j, i, f"{corr.iloc[i,j]:.2f}", ha="center", va="center", color="white" if abs(corr.iloc[i,j])>0.5 else "black")
ax.set_title("Metrics Correlation Matrix")
fig.savefig(f"{out_dir}/corr_matrix.png", bbox_inches="tight")
plt.close(fig)

print(f"All charts written to {out_dir}/")
