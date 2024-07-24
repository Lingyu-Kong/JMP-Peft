from collections.abc import Mapping

import matplotlib.pyplot as plt
import nshutils.typecheck as tc
import numpy as np
import seaborn as sns


def plot_values_buckets(
    data: Mapping[str, tc.Float[np.ndarray, "num_items"]],
    bucket_idx: tc.Int[np.ndarray, "num_items"],
    quantiles_list: tc.Float[np.ndarray, "num_buckets"],
    *,
    ncols: int,
):
    nrows = len(data) // ncols + 1
    if len(data) % ncols == 0:
        nrows -= 1

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(5 * ncols, 5 * nrows))
    axes = axes.flatten()

    for i, (name, m) in enumerate(data.items()):
        ax: plt.Axes = axes[i]
        for j in range(bucket_idx.max() + 1):
            # Set title to "{min} - {max}"
            q_left = quantiles_list[j]
            q_right = quantiles_list[j + 1]
            label = f"{q_left:.0f} - {q_right:.0f}"
            sns.histplot(
                m[bucket_idx == j],
                kde=True,
                label=label,
                ax=ax,
                # Show the bars very transparently
                alpha=0.05,
            )

        if i == (ncols - 1):
            ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.1))

            # Make sure legend bar colors are not transparent (like alpha above)
            for lh in ax.get_legend().legend_handles:
                if lh is None:
                    continue
                lh.set_alpha(1)

        ax.set_title(name)

    fig.tight_layout()
    plt.show()


def plot_values_buckets_sampled(
    data: Mapping[str, tc.Float[np.ndarray, "num_items"]],
    bucket_idx: tc.Int[np.ndarray, "num_items"],
    quantiles_list: tc.Float[np.ndarray, "num_buckets"],
    *,
    ncols: int,
):
    # Sample the same number of systems from each bucket
    unique_buckets = np.unique(bucket_idx)
    min_count = min(np.sum(bucket_idx == bucket) for bucket in unique_buckets)

    sampled_indices = []
    for bucket in unique_buckets:
        bucket_indices = np.where(bucket_idx == bucket)[0]
        sampled_indices.extend(
            np.random.choice(bucket_indices, min_count, replace=False)
        )

    sampled_indices = np.array(sampled_indices)
    buckets_sampled = bucket_idx[sampled_indices]
    data_sampled = {layer: values[sampled_indices] for layer, values in data.items()}

    plot_values_buckets(data_sampled, buckets_sampled, quantiles_list, ncols=ncols)
