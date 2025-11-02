---

# Vision-Only RL on Chrome Dino

Train **PPO** and **DQN** agents to play the **Chrome Dino** game using **raw pixels only**. This repo includes:

* ✅ Multi-seed training with CSV / TensorBoard logs
* ✅ Evaluation with **mean ± 95% CI** across seeds
* ✅ Compact, high-impact **ablations** (preprocessing, stacking, frame-skip, domain randomization, channels, resolution)
* ✅ Robust termination via **template matching** or **pixel-diff** (no OCR required)
* ✅ macOS/Windows/Linux key-sending via a portable `KeySender`
* ✅ **Docker** + `docker-compose` for headless, reproducible runs (Chromium + Xvfb)

---

## 0) Requirements

* Python **3.10–3.12**
* **Chrome** (or **Chromium**) installed
* OS input permissions (see macOS notes below)

### macOS permissions (important)

* System Settings → **Privacy & Security**

  * Grant **Screen Recording** to your terminal / Python
  * Grant **Accessibility** to your terminal / Python (for keystrokes)

---

## 1) Install

```bash
# (Recommended) Use a virtualenv
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

---

## 2) How the game window is handled

By default, the environment **does not open new tabs** on every reset. You have three options:

1. **Manual (simple)** — Open the game yourself, then start training:

   * `chrome://dino/` (Chrome only), or
   * `https://chromedino.com` (works everywhere)

2. **Trainer opens once** — Add a flag so the trainer opens it **one time**:

   ```bash
   --set experiment.open_chrome=true
   ```

   (On macOS this runs `open -a "Google Chrome" "chrome://dino/"` once.)

3. **Docker** — The container entrypoint launches **Chromium** (app mode) under Xvfb; no manual steps required.

> Tip: If you want the env to bring the already-open window to the foreground (without opening tabs) on reset, use:
> `--set env.focus_on_reset=true`

---

## 3) Repo layout

```
.
├─ envs/
│  └─ chrome_dino_env.py
├─ scripts/
│  ├─ train_ppo.py
│  ├─ train_dqn.py
│  ├─ evaluate.py
│  └─ plot_curves.py
├─ utils/
│  ├─ input_backends.py      # KeySender (Quartz / AppleScript / pydirectinput / pyautogui / xdotool)
│  ├─ metrics.py
│  └─ ...
├─ configs/
│  ├─ ppo_baseline.yaml
│  ├─ dqn_baseline.yaml
│  ├─ ppo_ablation_suite.yaml
│  └─ dqn_ablation_suite.yaml
├─ templates/                # optional (dino.png, game_over.png) for template matching
├─ logs/                     # created on first run
├─ results/                  # created on evaluation
├─ Dockerfile
└─ docker-compose.yml
```

Always run scripts from the repo root:

```bash
python -m scripts.train_ppo --help
```

---

## 4) Quick smoke test (env only)

```python
from envs.chrome_dino_env import ChromeDinoEnv
env = ChromeDinoEnv(auto_calibrate=True, monitor_index=1)  # tweak monitor_index as needed
obs, info = env.reset()
for _ in range(5):
    obs, r, d, tr, info = env.step(env.action_space.sample())
    if d: break
env.close()
print("ok")
```

If this prints `ok`, your permissions and capture path are good.

---

## 5) Training — PPO

### Minimal baseline (3 seeds, 50k timesteps)

```bash
python -m scripts.train_ppo --seeds 0 1 2 --total_timesteps 50000
```

### Device selection

* macOS CPU (safe default):

  ```bash
  --device cpu
  ```
* macOS Apple GPU (experimental):

  ```bash
  --device mps
  ```
* CUDA:

  ```bash
  --device cuda
  ```

### Examples

Baseline grayscale + resize only (no blur, no hist-eq), template termination:

```bash
python -m scripts.train_ppo \
  --seeds 0 1 2 --total_timesteps 50000 --device cpu \
  --set env.noise_level=0 env.blur=false env.hist_eq=false env.termination_method=template
```

Temporal stack 4:

```bash
python -m scripts.train_ppo --seeds 0 1 2 --total_timesteps 50000 \
  --set env.temporal_stack=4 env.blur=false env.hist_eq=false
```

Frame skip 4:

```bash
python -m scripts.train_ppo --seeds 0 1 2 --total_timesteps 50000 \
  --set env.frame_skip=4 env.blur=false env.hist_eq=false
```

Domain randomization:

```bash
python -m scripts.train_ppo --seeds 0 1 2 --total_timesteps 50000 \
  --set env.brightness_var=0.10 env.contrast_var=0.10 env.noise_level=0.02
```

Mixed channels (RGB + edges):

```bash
python -m scripts.train_ppo --seeds 0 1 2 --total_timesteps 50000 \
  --set env.obs_channels=mixed env.temporal_stack=1
```

High-res grayscale (166×200):

```bash
python -m scripts.train_ppo --seeds 0 1 2 --total_timesteps 50000 \
  --set env.obs_channels=grayscale env.obs_resolution=high
```

---

## 6) Training — DQN

Same interface:

```bash
python -m scripts.train_dqn --seeds 0 1 2 --total_timesteps 50000 \
  --set env.blur=false env.hist_eq=false env.termination_method=template
```

---

## 7) Configuration system

All defaults live in YAML (e.g., `configs/ppo_baseline.yaml`). Anything can be overridden at the CLI with `--set`:

```bash
# override nested keys:
--set env.temporal_stack=4 env.obs_channels=grayscale logging.tensorboard=false
```

Common env keys (all map to `ChromeDinoEnv`):

| Key                       | Values / Type                                                                    | Notes                                             |
| ------------------------- | -------------------------------------------------------------------------------- | ------------------------------------------------- |
| `env.input_backend`       | `auto` \| `quartz` \| `osascript` \| `pydirectinput` \| `pyautogui` \| `xdotool` | Portable keystrokes (OS-specific)                 |
| `env.auto_calibrate`      | bool                                                                             | Try to auto-find the canvas region                |
| `env.monitor_index`       | int                                                                              | 0 = “all monitors” (virtual), 1..N = real screens |
| `env.termination_method`  | `template` \| `pixeldiff` \| `either`                                            | Game-over detection mode                          |
| `env.template_thr`        | float (0.55–0.70 typical)                                                        | Template match threshold                          |
| `env.temporal_stack`      | int                                                                              | Frame stacking (1, 2, 4, …)                       |
| `env.frame_skip`          | int                                                                              | Skip frames to speed up                           |
| `env.action_repeat`       | int                                                                              | Repeat action across frames                       |
| `env.obs_channels`        | `grayscale` \| `rgb` \| `edges` \| `mixed`                                       | Observation channel layout                        |
| `env.obs_resolution`      | `low` \| `default` \| `high`                                                     | 42×50, 83×100, 166×200                            |
| `env.blur`, `env.hist_eq` | bool                                                                             | Preprocessing toggles                             |
| `env.edge_enhance`        | bool                                                                             | Canny + blend                                     |
| `env.brightness_var`      | float                                                                            | Gaussian brightness jitter                        |
| `env.contrast_var`        | float                                                                            | Gaussian contrast jitter                          |
| `env.noise_level`         | float                                                                            | Additive Gaussian noise                           |
| `env.reward_mode`         | `sparse` \| `dense` \| `distance` \| `survival`                                  | Reward shaping                                    |
| `env.reward_scaling`      | float                                                                            | Global reward scale                               |
| `env.action_sleep`        | float (seconds)                                                                  | Controls step pacing                              |
| `env.focus_on_reset`      | bool                                                                             | Bring Chrome/Chromium to front (no new tab)       |

Experiment keys:

| Key                          | Values / Type            | Notes                           |
| ---------------------------- | ------------------------ | ------------------------------- |
| `experiment.seeds`           | list\[int]               | e.g. `[0,1,2]`                  |
| `experiment.total_timesteps` | int                      | Budget per seed                 |
| `experiment.device`          | `cpu` \| `mps` \| `cuda` | Torch device                    |
| `experiment.open_chrome`     | bool                     | Trainer opens the game **once** |

---

## 8) Ablation suites (YAML)

Run a compact, high-impact sweep via provided configs:

```bash
# PPO sweep (edit configs/ppo_ablation_suite.yaml if desired)
# (You can loop over items in "sweep" using a tiny bash/python driver; or use docker-compose profiles below.)
```

A corresponding `configs/dqn_ablation_suite.yaml` is provided with the same ablations adapted to DQN.

---

## 9) Evaluation & Aggregation

Evaluate latest checkpoints and write per-seed CSVs + aggregate CSV:

```bash
# PPO
python -m scripts.evaluate --algo ppo --n_eval_episodes 20

# DQN
python -m scripts.evaluate --algo dqn --n_eval_episodes 20
```

Match eval env to training if needed:

```bash
# e.g., evaluating with template termination and grayscale-only preprocessing
--set env.termination_method=template env.blur=false env.hist_eq=false
```

Outputs:

* `results/raw/{algo}_seed_{k}.csv`
* `results/aggregates/{algo}_aggregate.csv` (mean, std, 95% CI)

---

## 10) Plot learning curves (mean ± 95% CI)

```bash
python -m scripts.plot_curves --algo_dirs logs/ppo logs/dqn
# -> results/figures/learning_curves.png
```

---

## 11) Docker (headless, reproducible)

Build once:

```bash
docker build -t dino-rl:latest .
```

Run specific services via profiles:

```bash
# PPO baseline
docker compose --profile ppo up ppo_baseline

# PPO ablations (examples)
docker compose --profile ablations up ppo_stack4 ppo_fs4 ppo_domain_rand ppo_mixed_edges ppo_highres_gray

# DQN baseline
docker compose --profile dqn up dqn_baseline

# Everything
docker compose --profile all up
```

The compose file:

* starts **Xvfb** + **Chromium** in app/fullscreen mode pointing to `https://chromedino.com`
* uses `xdotool` input backend inside the container
* logs to `./logs` (bound volume)

> After runs complete, evaluate/plot on the host as usual.

---

## 12) What’s a “timestep”? How many episodes?

* **Timestep** = one environment step (`env.step`) after any `frame_skip`/`action_repeat` logic.
* With PPO, the agent updates every `n_steps` timesteps; the number of **episodes** completed during `total_timesteps` depends on survival length and early terminations. There’s no fixed episodes count—shorter episodes → more episodes within the same timestep budget.

---

## 13) Troubleshooting

* **Blank/incorrect crop**: try `--set env.monitor_index=0` (virtual full desktop), ensure browser zoom is **100%**.
* **No keystrokes on macOS**: double-check **Accessibility** permission; try `--set env.focus_on_reset=true`.
* **Template termination too sensitive/insensitive**: adjust `--set env.template_thr=0.60` (try 0.55–0.70), ensure `templates/game_over.png` exists.
* **Perf / crashes on macOS GPU**: use `--device cpu` or `--device mps` with `PYTORCH_ENABLE_MPS_FALLBACK=1` (already set in trainer).
* **Inside Docker**: everything is headless; don’t expect a visible window. Use logs and results to verify.

---

## 14) Reproducibility artifacts

* Per-seed resolved config: `logs/{algo}/seed_{k}/run_config.json`
* Training logs (CSV/TensorBoard): `logs/{algo}/seed_{k}/…`
* Evaluation CSVs: `results/raw/`
* Aggregates: `results/aggregates/`
* Figures: `results/figures/`

---

## 15) Citation

If this benchmark or framework is useful in your research, please cite your paper and/or acknowledge:

> *Vision-only RL on Chrome Dino (benchmark + ablations framework)*

---

Happy dino-running! 🦖

docker build --no-cache -t dino-rl:latest .
docker compose --profile dqn up --build dqn_baseline
