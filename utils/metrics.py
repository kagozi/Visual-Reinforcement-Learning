import os
import csv
import numpy as np
from stable_baselines3.common.evaluation import evaluate_policy

def evaluate_and_save(model, env, n_eval_episodes: int, out_csv: str, seed: int):
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=n_eval_episodes, return_episode_rewards=False)
    with open(out_csv, "a", newline="") as f:
        w = csv.writer(f)
        if f.tell() == 0:
            w.writerow(["seed", "mean_reward", "std_reward"])
        w.writerow([seed, mean_reward, std_reward])

def aggregate_csvs(csv_paths, out_csv):
    rows = []
    for p in csv_paths:
        if os.path.exists(p):
            rows.append(np.genfromtxt(p, delimiter=",", names=True, dtype=None, encoding=None))
    if not rows:
        return
    # concatenate seeds rows
    stacked = np.concatenate(rows)
    mean = stacked["mean_reward"].mean()
    std = stacked["mean_reward"].std(ddof=1)
    n = stacked.shape[0]
    # 95% CI via normal approx
    ci95 = 1.96 * std / np.sqrt(n)
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["n_seeds", "mean_reward", "std_reward", "ci95"])
        w.writerow([n, mean, std, ci95])
