## Lamarck &nbsp; &nbsp; &nbsp; 2025-8-26
#### Python实现高斯混合模型可视化
---
```
输入文件目录：C:\Users\Lamarck\Desktop\input.csv
列名：data
```

```python
# -*- coding: utf-8 -*-

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.mixture import GaussianMixture
from scipy.stats import skew, norm

# ====== 配置 ======
CSV_PATH = r"C:\Users\Lamarck\Desktop\input.csv"
OUTPUT_FIG = r"C:\Users\Lamarck\Desktop\mass_histogram_fit.png"

BIN_NUM = 100
ALPHA_HIST = 0.35
RANDOM_STATE = 42
MAX_COMPONENTS = 6

# 过滤与显示范围
MIN_KEEP = 150       # 仅保留 >=0
MAX_KEEP = 1000.0      # 仅保留 <=1000
X_MIN, X_MAX = 0.0, 1000.0  # 画图显示范围

# ====== 读取与清洗数据 ======
if not os.path.exists(CSV_PATH):
    raise FileNotFoundError(f"CSV not found: {CSV_PATH}")

# 只读取 calibrated_values 一列
df = pd.read_csv(CSV_PATH, usecols=['calibrated_values'])

# 转为 float；无法解析的设为 NaN 并丢弃
s = pd.to_numeric(df['calibrated_values'], errors="coerce").dropna()

# 仅保留 0–1000
s = s[(s >= MIN_KEEP) & (s <= MAX_KEEP)]

data = s.values.astype(float)
n_total = data.size
if n_total == 0:
    raise ValueError("No valid numeric data found after filtering to [150, 1000].")

# ====== 选择 GMM 峰数（按 BIC） ======
X = data.reshape(-1, 1)

bics, gmms = [], []
for k in range(1, MAX_COMPONENTS + 1):
    gmm = GaussianMixture(
        n_components=k,
        covariance_type="full",
        random_state=RANDOM_STATE,
        n_init=5,
    ).fit(X)
    bics.append(gmm.bic(X))
    gmms.append(gmm)

best_idx = int(np.argmin(bics))
best_gmm = gmms[best_idx]
n_components = best_gmm.n_components

# ====== 提取参数并统计每峰指标 ======
weights = best_gmm.weights_                    # (K,)
means = best_gmm.means_.ravel()                # (K,)
sigmas = np.sqrt(best_gmm.covariances_.reshape(-1))  # (K,)

resp = best_gmm.predict_proba(X)               # (N, K)
hard_labels = resp.argmax(axis=1)

counts_per_comp = np.array([(hard_labels == k).sum() for k in range(n_components)])
props_per_comp = counts_per_comp / n_total

skews_per_comp = []
for k in range(n_components):
    subset = data[hard_labels == k]
    skews_per_comp.append(skew(subset, bias=False) if subset.size >= 3 else np.nan)
skews_per_comp = np.array(skews_per_comp)

# 按均值升序排序，便于展示
order = np.argsort(means)
means = means[order]
sigmas = sigmas[order]
weights = weights[order]
counts_per_comp = counts_per_comp[order]
props_per_comp = props_per_comp[order]
skews_per_comp = skews_per_comp[order]

# ====== 绘图：直方图 + 各峰曲线 + 总曲线 ======
fig, ax = plt.subplots(figsize=(11, 7))

# 固定横坐标范围：0–1000，并据此设定直方图的箱边界
bins = np.linspace(X_MIN, X_MAX, BIN_NUM + 1)
counts, bins, _ = ax.hist(
    data, bins=bins, alpha=ALPHA_HIST, edgecolor="white", linewidth=0.6
)

bin_w = (X_MAX - X_MIN) / BIN_NUM
x = np.linspace(X_MIN, X_MAX, 2000)

# 单峰曲线
for k in range(n_components):
    y_counts = weights[k] * norm.pdf(x, means[k], sigmas[k]) * n_total * bin_w
    ax.plot(x, y_counts, linewidth=2.0)

# 总曲线
y_total_pdf = sum(weights[k] * norm.pdf(x, means[k], sigmas[k]) for k in range(n_components))
ax.plot(x, y_total_pdf * n_total * bin_w, "k--", linewidth=2.0, label=f"GMM (K={n_components})")

# 注释
for k in range(n_components):
    mu, sd = means[k], sigmas[k]
    cnt, prop, skw = int(counts_per_comp[k]), props_per_comp[k], skews_per_comp[k]
    y_peak = weights[k] * norm.pdf(mu, mu, sd) * n_total * bin_w
    txt = (
        f"{mu:.0f} kDa\n"
        f"σ {sd:.1f} kDa\n"
        f"{cnt} counts ({prop:.0%})\n"
        f"Skewness: {0.0 if np.isnan(skw) else skw:.3f}"
    )
    ax.text(mu, y_peak * 1.05, txt, ha="center", va="bottom", fontsize=10)

ax.set_xlabel("Mass [kDa]", fontsize=12)
ax.set_ylabel("Counts", fontsize=12)
ax.set_title("Mass distribution with GMM fit", fontsize=14)
ax.legend(loc="upper right", frameon=False)
ax.set_xlim(X_MIN, X_MAX)
ax.grid(alpha=0.25, linestyle="--", linewidth=0.5)
plt.tight_layout()

fig.savefig(OUTPUT_FIG, dpi=300)
print(f"Done. Saved figure to: {OUTPUT_FIG}")
```
