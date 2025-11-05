import yaml, os, subprocess, argparse

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--algo", choices=["ppo", "dqn"], required=True)
    p.add_argument("--suite", type=str, required=True)  # e.g., configs/ppo_ablation_suite.yaml
    p.add_argument("--total_timesteps", type=int, default=100000)
    p.add_argument("--device", type=str, default="cpu")
    args = p.parse_args()

    with open(args.suite, "r") as f:
        suite = yaml.safe_load(f)

    base_cfg = {}  # Merge with default.yaml if needed
    seeds = suite["experiment"]["seeds"]  # [0,1,2]

    for entry in suite["sweep"]:
        name = entry["name"]
        ablation_cfg = entry.get("env", {})
        full_cfg = {**base_cfg, "env": ablation_cfg, "experiment": suite["experiment"]}

        temp_cfg_path = f"temp_{args.algo}_{name}.yaml"
        with open(temp_cfg_path, "w") as f:
            yaml.dump(full_cfg, f)

        cmd = [
            "python", "-m", f"scripts.train_{args.algo}",
            "--config", temp_cfg_path,
            "--seeds", *(str(s) for s in seeds),
            "--total_timesteps", str(args.total_timesteps),
            "--device", args.device,
            "--log_dir", f"logs/{args.algo}_{name}",
            "--set", "experiment.debug_dump=true", "env.input_backend=xdotool",
            "env.auto_calibrate=true", "env.termination_method=template",
            "env.template_thr=0.62"
        ]
        print(f"Running ablation: {name}")
        subprocess.run(cmd, check=True)
        os.remove(temp_cfg_path)  # Cleanup

if __name__ == "__main__":
    main()