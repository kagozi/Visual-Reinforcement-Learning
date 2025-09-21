# scripts/evaluate.py
import os, argparse, glob
import pandas as pd
from stable_baselines3 import PPO, DQN
from envs.chrome_dino_env import ChromeDinoEnv
from utils.metrics import evaluate_and_save, aggregate_csvs

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--algo", choices=["ppo","dqn"], required=True)
    p.add_argument("--seeds", type=int, nargs="+", default=[0,1,2])
    p.add_argument("--ckpt_step", type=int, default=None)
    p.add_argument("--n_eval_episodes", type=int, default=20)
    # simple env toggles
    p.add_argument("--no_blur", action="store_true")
    p.add_argument("--no_hist_eq", action="store_true")
    p.add_argument("--obs_resolution", type=str, default=None)
    p.add_argument("--obs_channels", type=str, default=None)
    return p.parse_args()

def main():
    args = parse_args()
    algo_cls = PPO if args.algo == "ppo" else DQN

    per_seed_csvs = []
    base_dir = os.path.join("logs", args.algo)
    os.makedirs("results/raw", exist_ok=True)
    os.makedirs("results/aggregates", exist_ok=True)

    for seed in args.seeds:
        run_dir = os.path.join(base_dir, f"seed_{seed}")
        ckpt_dir = os.path.join(run_dir, "checkpoints")
        model_path = None
        if args.ckpt_step:
            cand = os.path.join(ckpt_dir, f"model_{args.ckpt_step}.zip")
            if os.path.exists(cand):
                model_path = cand
        if model_path is None:
            zips = sorted(glob.glob(os.path.join(ckpt_dir, "model_*.zip")))
            if zips:
                model_path = zips[-1]
        if model_path is None:
            cand = os.path.join(run_dir, "final_model.zip")
            if os.path.exists(cand):
                model_path = cand

        if not model_path:
            print(f"[WARN] No checkpoint for seed {seed} in {run_dir}")
            continue

        model = algo_cls.load(model_path)

        env_kwargs = dict(
            blur=(not args.no_blur),
            hist_eq=(not args.no_hist_eq),
        )
        if args.obs_resolution: env_kwargs["obs_resolution"] = args.obs_resolution
        if args.obs_channels:   env_kwargs["obs_channels"]   = args.obs_channels

        env = ChromeDinoEnv(**env_kwargs, seed=seed)
        out_csv = os.path.join("results", "raw", f"{args.algo}_seed_{seed}.csv")
        evaluate_and_save(model, env, n_eval_episodes=args.n_eval_episodes, out_csv=out_csv, seed=seed)
        env.close()
        per_seed_csvs.append(out_csv)

    agg_out = os.path.join("results", "aggregates", f"{args.algo}_aggregate.csv")
    aggregate_csvs(per_seed_csvs, agg_out)
    if os.path.exists(agg_out):
        print(pd.read_csv(agg_out))

if __name__ == "__main__":
    main()
