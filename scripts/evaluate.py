import os, argparse, glob
import pandas as pd
from stable_baselines3 import PPO, DQN
from envs.chrome_dino_env import ChromeDinoEnv
from utils.metrics import evaluate_and_save, aggregate_csvs

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--algo", choices=["ppo","dqn"], required=True)
    p.add_argument("--seeds", type=int, nargs="+", default=[0,1,2,3,4])
    p.add_argument("--ckpt_step", type=int, default=None, help="load checkpoints/model_{step}.zip; if None, the latest")
    p.add_argument("--n_eval_episodes", type=int, default=20)
    p.add_argument("--reward_mode", choices=["sparse","shaped"], default="sparse")
    p.add_argument("--termination", choices=["ocr","pixeldiff"], default="ocr")
    p.add_argument("--no_blur", action="store_true")
    p.add_argument("--no_hist_eq", action="store_true")
    return p.parse_args()

def main():
    args = parse_args()
    algo = PPO if args.algo == "ppo" else DQN

    per_seed_csvs = []
    for seed in args.seeds:
        run_dir = os.path.join("logs", args.algo, f"seed_{seed}")
        ckpt_dir = os.path.join(run_dir, "checkpoints")
        if args.ckpt_step:
            model_path = os.path.join(ckpt_dir, f"model_{args.ckpt_step}.zip")
        else:
            # latest checkpoint
            paths = sorted(glob.glob(os.path.join(ckpt_dir, "model_*.zip")))
            model_path = paths[-1] if paths else None
        if not model_path:
            print(f"[WARN] No checkpoint for seed {seed}")
            continue

        model = algo.load(model_path)

        env = ChromeDinoEnv(
            blur=(not args.no_blur),
            hist_eq=(not args.no_hist_eq),
            reward_mode=args.reward_mode,
            termination_method=args.termination,
            seed=seed,
        )
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
