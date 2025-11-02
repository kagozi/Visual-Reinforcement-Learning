# scripts/train_dqn.py
import argparse, json, os, yaml
from copy import deepcopy


def load_yaml(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def apply_overrides(cfg, sets):
    if not sets:
        return cfg
    for kv in sets:
        if "=" not in kv:
            raise ValueError(f"--set expects key=value, got: {kv}")
        key, val = kv.split("=", 1)
        lv = val.lower()
        if lv in ("true", "false"):
            cast_val = (lv == "true")
        else:
            try:
                cast_val = int(val)
            except ValueError:
                try:
                    cast_val = float(val)
                except ValueError:
                    cast_val = val
        cur = cfg
        parts = key.split(".")
        for p in parts[:-1]:
            if p not in cur or not isinstance(cur[p], dict):
                cur[p] = {}
            cur = cur[p]
        cur[parts[-1]] = cast_val
    return cfg


def ensure_dir(p):
    os.makedirs(p, exist_ok=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="configs/dqn_baseline.yaml")
    ap.add_argument("--seeds", nargs="+", type=int, default=None)
    ap.add_argument("--total_timesteps", type=int, default=None)
    ap.add_argument("--log_dir", type=str, default=None)
    ap.add_argument("--device", type=str, default=None)  # cpu|cuda|mps|auto

    # env convenience (same surface as PPO script)
    for k, t in [
        ("temporal_stack", int), ("frame_skip", int), ("action_repeat", int),
        ("obs_resolution", str), ("obs_channels", str),
        ("blur", str), ("hist_eq", str), ("edge_enhance", str),
        ("noise_level", float), ("brightness_var", float), ("contrast_var", float),
        ("reward_mode", str), ("reward_scaling", float), ("action_sleep", float),
        ("termination_method", str), ("template_thr", float),
        ("input_backend", str), ("auto_calibrate", str), ("monitor_index", int),
        ("focus_on_reset", str)
    ]:
        ap.add_argument(f"--{k}", type=t, default=None)

    ap.add_argument("--set", nargs="*", default=[])
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    if args.seeds is not None:
        cfg.setdefault("experiment", {})["seeds"] = args.seeds
    if args.total_timesteps is not None:
        cfg.setdefault("experiment", {})["total_timesteps"] = args.total_timesteps
    if args.log_dir is not None:
        cfg.setdefault("logging", {})["log_dir"] = args.log_dir
    if args.device is not None:
        cfg.setdefault("experiment", {})["device"] = args.device

    # env convenience overrides
    env_cfg = cfg.setdefault("env", {})

    def maybe(k, v):
        if v is not None:
            env_cfg[k] = (str(v).lower() == "true") if k in (
                "blur", "hist_eq", "edge_enhance", "auto_calibrate", "focus_on_reset"
            ) else v

    for k in [
        "temporal_stack", "frame_skip", "action_repeat", "obs_resolution", "obs_channels",
        "blur", "hist_eq", "edge_enhance", "noise_level", "brightness_var", "contrast_var",
        "reward_mode", "reward_scaling", "action_sleep", "termination_method", "template_thr",
        "input_backend", "auto_calibrate", "monitor_index", "focus_on_reset"
    ]:
        maybe(k, getattr(args, k))

    # generic nested overrides
    cfg = apply_overrides(cfg, args.set)

    # --- delay heavy imports ---
    import numpy as np, torch
    from stable_baselines3 import DQN
    from stable_baselines3.common.logger import configure as sb3_configure
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor, VecTransposeImage
    from stable_baselines3.common.callbacks import CheckpointCallback
    from envs.chrome_dino_env import ChromeDinoEnv

    exp     = cfg.get("experiment", {})
    seeds   = exp.get("seeds", [0])
    steps   = int(exp.get("total_timesteps", 50000))
    device  = exp.get("device", "cpu")
    dqn_cfg = cfg.get("dqn", {})
    log_cfg = cfg.get("logging", {})

    base_log_dir = log_cfg.get("log_dir", "logs/dqn")
    ensure_dir(base_log_dir)

    backends = ["stdout"]
    if log_cfg.get("csv", True):
        backends.append("csv")
    if log_cfg.get("tensorboard", False):
        backends.append("tensorboard")

    print("=== Resolved Config ===")
    print(json.dumps(cfg, indent=2))

    for seed in seeds:
        run_dir = os.path.join(base_log_dir, f"seed_{seed}")
        ckpt_dir = os.path.join(run_dir, "checkpoints")
        ensure_dir(run_dir); ensure_dir(ckpt_dir)

        # optional debug dump directory for env (screenshots etc.)
        if cfg.get("experiment", {}).get("debug_dump", False):
            env_cfg["debug_dump_dir"] = os.path.join(run_dir, "debug")
            env_cfg.setdefault("debug_dump_once", True)
            env_cfg.setdefault("debug_tag", "calibration")

        # persist resolved config
        with open(os.path.join(run_dir, "run_config.json"), "w") as f:
            json.dump(cfg, f, indent=2)

        # seeds
        os.environ["PYTHONHASHSEED"] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        # ---- Build a fresh env stack per seed ----
        mon_path = os.path.join(run_dir, "monitor.csv")

        def make_env():
            def _thunk():
                e = ChromeDinoEnv(**deepcopy(env_cfg), seed=seed)
                return Monitor(e, filename=mon_path)  # per-episode rewards & lengths
            return _thunk

        venv = DummyVecEnv([make_env()])
        venv = VecMonitor(venv)          # adds episode stats to progress.csv when possible
        venv = VecTransposeImage(venv)   # HWC->CHW for CnnPolicy

        # --- Build DQN ---
        model = DQN(
            "CnnPolicy",
            venv,
            device=device,  # SB3 DQN supports 'device'
            verbose=1,
            learning_rate=dqn_cfg.get("lr", 1e-4),
            buffer_size=dqn_cfg.get("buffer_size", 200_000),   # saner default than 1.2M
            learning_starts=dqn_cfg.get("learning_starts", 1_000),
            batch_size=dqn_cfg.get("batch_size", 32),
            gamma=dqn_cfg.get("gamma", 0.99),
            tau=dqn_cfg.get("tau", 1.0),
            target_update_interval=dqn_cfg.get("target_update", 10_000),
            exploration_initial_eps=dqn_cfg.get("eps_start", 1.0),
            exploration_final_eps=dqn_cfg.get("eps_final", 0.1),
            exploration_fraction=dqn_cfg.get("eps_fraction", 0.1),
            tensorboard_log=(run_dir if "tensorboard" in backends else None),
            seed=seed,
        )

        model.set_logger(sb3_configure(run_dir, backends))
        cb = CheckpointCallback(
            save_freq=log_cfg.get("checkpoint_freq", 10_000),
            save_path=ckpt_dir,
            name_prefix="model"
        )

        print(f"[seed {seed}] training for {steps} timesteps on device='{device}' …")
        model.learn(total_timesteps=steps, callback=cb)
        model.save(os.path.join(run_dir, "final_model"))

        venv.close()


if __name__ == "__main__":
    main()