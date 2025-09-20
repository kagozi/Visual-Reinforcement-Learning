import os, time, argparse, webbrowser
from stable_baselines3 import PPO
from stable_baselines3.common.logger import configure
from envs.chrome_dino_env import ChromeDinoEnv
from utils.callbacks import CheckpointEveryN, WallClockLogger
from utils.seeding import set_all_seeds
from utils.metadata import save_run_config

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, nargs="+", default=[0,1,2,3,4])
    p.add_argument("--total_timesteps", type=int, default=100_000)
    p.add_argument("--n_steps", type=int, default=2048)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--n_epochs", type=int, default=10)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--gae_lambda", type=float, default=0.95)
    p.add_argument("--ent_coef", type=float, default=0.01)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--clip_range", type=float, default=0.2)
    p.add_argument("--check_freq", type=int, default=10_000)

    # Ablations
    p.add_argument("--no_blur", action="store_true")
    p.add_argument("--no_hist_eq", action="store_true")
    p.add_argument("--reward_mode", choices=["sparse","shaped"], default="sparse")
    p.add_argument("--termination", choices=["ocr","pixeldiff"], default="ocr")
    p.add_argument("--pixeldiff_thresh", type=int, default=40_000)
    p.add_argument("--action_sleep", type=float, default=0.10)
    return p.parse_args()

def main():
    args = parse_args()
    try:
        webbrowser.open('chrome://dino/')
    except Exception:
        pass

    for seed in args.seeds:
        set_all_seeds(seed)
        run_dir = os.path.join("logs", "ppo", f"seed_{seed}")
        ckpt_dir = os.path.join(run_dir, "checkpoints")
        tb_dir   = os.path.join(run_dir, "tb")
        os.makedirs(run_dir, exist_ok=True)

        env_cfg = dict(
            blur=(not args.no_blur),
            hist_eq=(not args.no_hist_eq),
            reward_mode=args.reward_mode,
            termination_method=args.termination,
            pixeldiff_thresh=args.pixeldiff_thresh,
            action_sleep=args.action_sleep,
            seed=seed,
        )
        save_run_config(os.path.join(run_dir, "run_config.json"), env_cfg)

        env = ChromeDinoEnv(**env_cfg)

        model = PPO(
            "CnnPolicy",
            env,
            verbose=1,
            n_steps=args.n_steps,
            batch_size=args.batch_size,
            n_epochs=args.n_epochs,
            gamma=args.gamma,
            gae_lambda=args.gae_lambda,
            ent_coef=args.ent_coef,
            learning_rate=args.lr,
            clip_range=args.clip_range,
            tensorboard_log=tb_dir,
            seed=seed,
        )

        logger = configure(tb_dir, ["stdout", "csv", "tensorboard"])
        model.set_logger(logger)

        callback = CheckpointEveryN(args.check_freq, ckpt_dir)
        meta_cb  = WallClockLogger(save_dir=run_dir, algo="PPO", seed=seed, env_cfg=env_cfg)

        model.learn(total_timesteps=args.total_timesteps, callback=[callback, meta_cb])

        env.close()

if __name__ == "__main__":
    main()
