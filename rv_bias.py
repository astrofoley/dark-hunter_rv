import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def read_data(directory):
    data = []
    for file in os.listdir(directory):
        if file.endswith("_orders.txt"):
            filepath = os.path.join(directory, file)
            star_name = "_".join(file.split("_")[:3])  # Extract star name
            epoch = int(file.split("epoch_")[1].split("_")[0])  # Extract epoch number
            df = pd.read_csv(filepath, sep='\s+', comment='#',
                             names=["Order", "Suborder", "RV", "RV_Error"])
            df["Order"] = pd.to_numeric(df["Order"], errors="coerce")
            df["Suborder"] = pd.to_numeric(df["Suborder"], errors="coerce")
            df["RV"] = pd.to_numeric(df["RV"], errors="coerce")
            df["RV_Error"] = pd.to_numeric(df["RV_Error"], errors="coerce")
            df["Star"] = star_name
            df["Epoch"] = epoch
            data.append(df)
    return pd.concat(data, ignore_index=True)


def compute_bias(df, sigma=2.2, max_iter=20, tol=1e-3):
    biases = []
    for (star, epoch), group in df.groupby(["Star", "Epoch"]):
        prev_len = len(group)

        for _ in range(max_iter):
            median_rv = np.median(group["RV"])
            std_rv = np.std(group["RV"])

            # Sigma clipping: Remove outliers beyond sigma threshold
            filtered_group = group[np.abs(group["RV"] - median_rv) <= sigma * std_rv]

            if len(filtered_group) == prev_len or abs(len(filtered_group) - prev_len) / prev_len < tol:
                break  # Convergence reached

            prev_len = len(filtered_group)
            group = filtered_group  # Update the group for the next iteration

        if len(group) > 0:
            weights = 1 / group["RV_Error"] ** 2
            weighted_mean_rv = np.sum(group["RV"] * weights) / np.sum(weights)
            group["Bias"] = group["RV"] - weighted_mean_rv
            biases.append(group)

    return pd.concat(biases, ignore_index=True) if biases else pd.DataFrame(columns=df.columns)


def compute_statistics(df, groupby_cols):
    stats = df.groupby(groupby_cols).apply(lambda g: pd.Series({
        "Bias_Mean": np.average(g["Bias"], weights=1 / g["RV_Error"] ** 2),
        "Bias_Error": np.sqrt(1 / np.sum(1 / g["RV_Error"] ** 2)),
        "Bias_RMS": np.sum(
            (g["Bias"] - np.average(g["Bias"], weights=1 / g["RV_Error"] ** 2)) ** 2 / g["RV_Error"] ** 2) / np.sum(
            1 / g["RV_Error"] ** 2)

    })).reset_index()
    return stats


def plot_bias(df, stats, title):
    plt.figure(figsize=(10, 6))
    for star, group in df.groupby("Star"):
        plt.scatter(group["Order"], group["Bias"], label=f"{star}", alpha=0.5)
        plt.errorbar(stats["Order"], stats["Bias_Mean"],
                     yerr=stats["Bias_Error"], fmt='o', capsize=5,
                     color='black', label='Mean Bias Error')
        plt.errorbar(stats["Order"], stats["Bias_Mean"],
                     yerr=stats["Bias_RMS"], fmt='o', capsize=5,
                     color='red', label='RMS Bias Error')
    plt.axhline(0, color='gray', linestyle='--')
    plt.xlabel("Order")
    plt.ylabel("Bias (km/s)")
    plt.legend()
    plt.title(title)
    plt.show()


def main():
    directory = "output/"
    df = read_data(directory)
    df_filtered = compute_bias(df)

    all_stats = compute_statistics(df_filtered, ["Order", "Suborder"])
    per_star_stats = compute_statistics(df_filtered, ["Star", "Order", "Suborder"])

    plot_bias(df_filtered, all_stats, "Bias")

    # plot_bias(df_filtered, all_stats, "Radial Velocity Bias Across All Data")

    ''' for star, group in df_filtered.groupby("Star"):
        star_stats = per_star_stats[per_star_stats["Star"] == star]
        plot_bias(group, star_stats, f"Radial Velocity Bias for {star}") '''

    all_stats.to_csv("bias_statistics.txt", index=False, sep=" ")


if __name__ == "__main__":
    main()

