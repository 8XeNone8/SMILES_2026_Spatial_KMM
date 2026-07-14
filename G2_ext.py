"""
temp file 14.07
"""


import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

from CV_Methods import (
    generate_spatial_data,
    random_cv,
    spatial_cv,
    kmm_weighted_spatial_cv,
    kmm_weighted_random_cv,
)

FUNCS = ['linear', 'exponential', 'periodic', 'random', 'random_autocorr']
METHODS = [
    'Random CV',
    'Spatial CV',
    'KMM-spatial (fold)',
    'KMM-spatial (global)',
    'KMM-random (fold)',
    'KMM-random (global)',
]


def run_g2_comparison(n_folds=5, n_samples=300, noise_std=0.3, gamma=0.5, B=10):
    results = []
    for func in FUNCS:
        X, y = generate_spatial_data(n_samples=n_samples, func=func,
                                      noise_std=noise_std, coord_range=(-5, 5))
        X_scaled = StandardScaler().fit_transform(X)
        y_scaled = StandardScaler().fit_transform(y.reshape(-1, 1)).flatten()

        mse_rand, std_rand = random_cv(X_scaled, y_scaled, n_folds=n_folds)
        mse_sp, std_sp = spatial_cv(X_scaled, y_scaled, n_folds=n_folds)

        mse_sp_f, std_sp_f = kmm_weighted_spatial_cv(
            X_scaled, y_scaled, n_folds=n_folds, gamma=gamma, B=B, weighting='fold')
        mse_sp_g, std_sp_g = kmm_weighted_spatial_cv(
            X_scaled, y_scaled, n_folds=n_folds, gamma=gamma, B=B, weighting='global')

        mse_rd_f, std_rd_f = kmm_weighted_random_cv(
            X_scaled, y_scaled, n_folds=n_folds, gamma=gamma, B=B, weighting='fold')
        mse_rd_g, std_rd_g = kmm_weighted_random_cv(
            X_scaled, y_scaled, n_folds=n_folds, gamma=gamma, B=B, weighting='global')

        results.append({
            'Function': func,
            'Random CV_mean': mse_rand, 'Random CV_std': std_rand,
            'Spatial CV_mean': mse_sp, 'Spatial CV_std': std_sp,
            'KMM-spatial (fold)_mean': mse_sp_f, 'KMM-spatial (fold)_std': std_sp_f,
            'KMM-spatial (global)_mean': mse_sp_g, 'KMM-spatial (global)_std': std_sp_g,
            'KMM-random (fold)_mean': mse_rd_f, 'KMM-random (fold)_std': std_rd_f,
            'KMM-random (global)_mean': mse_rd_g, 'KMM-random (global)_std': std_rd_g,
        })

    df = pd.DataFrame(results)
    display_df = df[['Function']].copy()
    for m in METHODS:
        display_df[m] = (df[f'{m}_mean'].map('{:.4f}'.format) + ' +- ' +
                          df[f'{m}_std'].map('{:.4f}'.format))
    return results, display_df


def main():
    results, display_df = run_g2_comparison(n_folds=5, n_samples=300, noise_std=0.3, gamma=0.5, B=10)

    print("global vs fold-specific KMM weighting:")
    print(display_df.to_string(index=False))

    with open("results_g2.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print("saved to results_g2.json")

    df = pd.DataFrame(results)
    funcs = df['Function'].to_list()
    data = np.zeros((len(funcs), len(METHODS)))
    for i, func in enumerate(funcs):
        for j, meth in enumerate(METHODS):
            data[i, j] = df.loc[df['Function'] == func, f'{meth}_mean'].values[0]

    x = np.arange(len(funcs))
    width = 0.13
    plt.figure(figsize=(13, 6))
    for j, meth in enumerate(METHODS):
        plt.bar(x + j * width, data[:, j], width, label=meth)
    plt.xlabel('Spatial function')
    plt.ylabel('MSE')
    plt.title('G2: fold-specific vs global KMM weighting (spatial and random folds)')
    plt.xticks(x + width * (len(METHODS) - 1) / 2, list(funcs))
    plt.legend(fontsize=8, ncol=2)
    plt.tight_layout()
    plt.savefig("g2_comparison.png", dpi=150)
    print("Figure saved to g2_comparison.png")

    table_md = display_df.to_markdown(index=False)
    try:
        with open("SOLUTION.md", "r", encoding="utf-8") as f:
            content = f.read()
    except FileNotFoundError:
        content = ""
    marker = "## Results"
    g2_section = (
        "\n\n## G2 results: global vs. fold-specific KMM weighting\n\n"
        "`KMM-spatial (fold)` re-solves the KMM program per test fold, matching that "
        "fold's train covariates to its own held-out covariates. `KMM-spatial (global)` "
        "solves KMM once for the whole sample against a fixed proxy of the deployment "
        "distribution (uniform coverage of the domain) and reuses those weights across "
        "folds. The `-random` variants apply the same two weighting schemes on top of "
        "random (non-spatial) folds, as a control.\n\n" + table_md + "\n"
    )
    if marker in content and "## G2 results" not in content:
        content = content.rstrip("\n") + g2_section
        with open("SOLUTION.md", "w", encoding="utf-8") as f:
            f.write(content)
        print("SOLUTION.md updated with G2 section")


if __name__ == "__main__":
    main()
