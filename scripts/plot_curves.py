import os, glob, argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--algo_dirs", nargs="+", default=["logs/ppo", "logs/dqn"])
    p.add_argument("--out", default="results/figures/learning_curves.png")
    return p.parse_args()

def load_progress_csvs(algo_dir):
    # SB3 CSV logs live at tb/progress.csv per seed (we configured CSV logging)
    dfs = []
    for seed_dir in sorted(glob.glob(os.path.join(algo_dir, "seed_*"))):
        csvs = glob.glob(os.path.join(seed_dir, "tb", "progress.csv"))
        if not csvs: 
            continue
        df = pd.read_csv(csvs[0])
        df["seed_dir"] = os.path.basename(seed_dir)
        dfs.append(df)
    return dfs

def mean_ci(df, x_col, y_col, num_bins=200):
    # align different lengths by binning timesteps
    x = df[x_col].values
    y = df[y_col].values
    return x, y

def main():
    args = parse_args()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    plt.figure(figsize=(8,5))

    for algo_dir in args.algo_dirs:
        algo_name = os.path.basename(algo_dir)
        dfs = load_progress_csvs(algo_dir)
        if not dfs:
            continue
        # Concatenate and compute group stats by 'timesteps'
        # SB3 writes 'time/total_timesteps' and 'rollout/ep_rew_mean'
        merged = []
        for df in dfs:
            keep = df[["time/total_timesteps","rollout/ep_rew_mean"]].dropna()
            keep = keep.rename(columns={"time/total_timesteps":"t", "rollout/ep_rew_mean":"rew"})
            merged.append(keep)
        # resample to common grid
        grid = np.linspace(0, min(m["t"].max() for m in merged), 200)
        Ys = []
        for m in merged:
            y = np.interp(grid, m["t"].values, m["rew"].values)
            Ys.append(y)
        Ys = np.vstack(Ys)
        mean = Ys.mean(axis=0)
        std = Ys.std(axis=0, ddof=1)
        n = Ys.shape[0]
        ci95 = 1.96 * std / np.sqrt(n)
        plt.plot(grid, mean, label=algo_name.upper())
        plt.fill_between(grid, mean - ci95, mean + ci95, alpha=0.2)

    plt.xlabel("Timesteps")
    plt.ylabel("Episode reward (mean)")
    plt.title("Learning curves (mean ± 95% CI across seeds)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(args.out, dpi=200)
    print(f"Saved {args.out}")

if __name__ == "__main__":
    main()
