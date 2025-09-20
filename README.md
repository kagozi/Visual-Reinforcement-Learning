# Vision-Only RL on Chrome Dino

Reproducible experiments for training **PPO** and **DQN** agents on the **Chrome Dino** game using **pixel observations only**. Includes:

* Multi-seed training with CSV/TensorBoard logs
* Evaluation with **mean ± 95% CI** aggregation
* Ablations: preprocessing (blur, histogram equalization), reward shaping, and termination detection (OCR vs pixel-diff)
* Optional macOS auto-calibration of the game region

---

## 1) Setup

```bash
# Create & activate a venv (optional but recommended)
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

**Permissions (macOS):**

* System Settings → **Privacy & Security** → grant **Screen Recording** to your terminal/Python
* …and **Accessibility** (for keystrokes)

---

## 2) Launch the game

You can open the game manually in Chrome, or let the scripts open it. For manual:

* Chrome URL: `chrome://dino/` (requires Chrome)
* Public clone: [https://chromedino.com](https://chromedino.com) (fallback if `chrome://` doesn’t work)

**Tip:** Ensure Chrome zoom is **100%** and the game is visible on the target monitor.

---

## 3) Folder layout (expected)

```
vision-rl-dino/
├─ envs/
│  ├─ chrome_dino_env.py
│  └─ dino_playwright_env.py        # optional, if you use the Playwright backend
├─ scripts/
│  ├─ train_ppo.py
│  ├─ train_dqn.py
│  ├─ evaluate.py
│  └─ plot_curves.py
├─ utils/
│  ├─ callbacks.py
│  ├─ seeding.py
│  ├─ metrics.py
│  └─ metadata.py
├─ logs/
├─ results/
│  ├─ raw/
│  ├─ aggregates/
│  └─ figures/
└─ templates/                        # OPTIONAL: for template matching (game_over.png, dino.png)
```

Run scripts **from the repo root** using module syntax:

```bash
python -m scripts.train_ppo --help
```

---

## 4) Training – PPO

### Quick start (baseline training, 3 seeds, 50k steps)

```bash
python -m scripts.train_ppo --seeds 0 1 2 --total_timesteps 50000
```

Outputs per-seed logs to `logs/ppo/seed_{k}/`.

### Flags

| Flag                |       Type |     Default | Purpose                          |
| ------------------- | ---------: | ----------: | -------------------------------- |
| `--seeds`           | list\[int] | `0 1 2 3 4` | Random seeds to run              |
| `--total_timesteps` |        int |    `100000` | Training budget per seed         |
| `--n_steps`         |        int |      `2048` | PPO rollout length               |
| `--batch_size`      |        int |        `64` | PPO minibatch size               |
| `--n_epochs`        |        int |        `10` | PPO epochs per update            |
| `--gamma`           |      float |      `0.99` | Discount factor                  |
| `--gae_lambda`      |      float |      `0.95` | GAE lambda                       |
| `--ent_coef`        |      float |      `0.01` | Entropy coef                     |
| `--lr`              |      float |      `3e-4` | Learning rate                    |
| `--clip_range`      |      float |       `0.2` | PPO clip range                   |
| `--check_freq`      |        int |     `10000` | Checkpoint frequency (timesteps) |

**Ablation flags** (forwarded to the environment):

| Flag                 |     Type |     Default | Effect                                           |                         |
| -------------------- | -------: | ----------: | ------------------------------------------------ | ----------------------- |
| `--no_blur`          |   toggle |         off | Disable Gaussian blur                            |                         |
| `--no_hist_eq`       |   toggle |         off | Disable histogram equalization                   |                         |
| `--reward_mode`      | \`sparse |    shaped\` | `sparse`                                         | Reward shaping ablation |
| `--termination`      |    \`ocr | pixeldiff\` | `ocr`                                            | Termination detector    |
| `--pixeldiff_thresh` |      int |     `40000` | Threshold if `pixeldiff` is used                 |                         |
| `--action_sleep`     |    float |      `0.10` | Delay (s) after each action (controls game pace) |                         |

> **Environment advanced options** (for `envs/chrome_dino_env.py`): `monitor_index`, `canvas_thresh_percentile`, `calibrate_via_window`, `force_window_bounds`. These are not CLI flags by default; see **Advanced calibration** at the end to use them.

---

## 5) Training – DQN

Same CLI pattern, with DQN-specific defaults:

```bash
python -m scripts.train_dqn.py --seeds 0 1 2 --total_timesteps 50000
```

**Ablation flags available here too:** `--no_blur`, `--no_hist_eq`, `--reward_mode`, `--termination`, `--pixeldiff_thresh`, `--action_sleep`.

---

## 6) “All combinations” of ablation flags

The ablations are binary/two-way for each of four knobs:

* **Blur**: on/off (`--no_blur`)
* **HistEq**: on/off (`--no_hist_eq`)
* **Reward mode**: `sparse` or `shaped`
* **Termination**: `ocr` or `pixeldiff` (with `--pixeldiff_thresh`)

That’s **2 × 2 × 2 × 2 = 16** combinations. Below are **examples** covering *every* combination for PPO at 50k steps with seeds `0 1 2`. You can adapt the same for DQN by swapping the script name.

> **NOTE:** Running all 16 combos × 3 seeds is time-consuming. See the “recommended subset” after this section if you want a tight suite.

### Matrix (PPO, 3 seeds, 50k)

**A) Blur=ON, HistEq=ON**

```bash
# A1: reward=sparse, term=ocr
python -m scripts.train_ppo --seeds 0 1 2 --total_timesteps 50000

# A2: reward=sparse, term=pixeldiff
python -m scripts.train_ppo --seeds 0 1 2 --total_timesteps 50000 --termination pixeldiff --pixeldiff_thresh 45000

# A3: reward=shaped, term=ocr
python -m scripts.train_ppo --seeds 0 1 2 --total_timesteps 50000 --reward_mode shaped

# A4: reward=shaped, term=pixeldiff
python -m scripts.train_ppo --seeds 0 1 2 --total_timesteps 50000 --reward_mode shaped --termination pixeldiff --pixeldiff_thresh 45000
```

**B) Blur=OFF, HistEq=ON**

```bash
# B1: sparse + ocr
python -m scripts.train_ppo --seeds 0 1 2 --total_timesteps 50000 --no_blur

# B2: sparse + pixeldiff
python -m scripts.train_ppo --seeds 0 1 2 --total_timesteps 50000 --no_blur --termination pixeldiff --pixeldiff_thresh 45000

# B3: shaped + ocr
python -m scripts.train_ppo --seeds 0 1 2 --total_timesteps 50000 --no_blur --reward_mode shaped

# B4: shaped + pixeldiff
python -m scripts.train_ppo --seeds 0 1 2 --total_timesteps 50000 --no_blur --reward_mode shaped --termination pixeldiff --pixeldiff_thresh 45000
```

**C) Blur=ON, HistEq=OFF**

```bash
# C1: sparse + ocr
python -m scripts.train_ppo --seeds 0 1 2 --total_timesteps 50000 --no_hist_eq

# C2: sparse + pixeldiff
python -m scripts.train_ppo --seeds 0 1 2 --total_timesteps 50000 --no_hist_eq --termination pixeldiff --pixeldiff_thresh 45000

# C3: shaped + ocr
python -m scripts.train_ppo --seeds 0 1 2 --total_timesteps 50000 --no_hist_eq --reward_mode shaped

# C4: shaped + pixeldiff
python -m scripts.train_ppo --seeds 0 1 2 --total_timesteps 50000 --no_hist_eq --reward_mode shaped --termination pixeldiff --pixeldiff_thresh 45000
```

**D) Blur=OFF, HistEq=OFF** (grayscale+resize only)

```bash
# D1: sparse + ocr  (your “first run”)
python -m scripts.train_ppo --seeds 0 1 2 --total_timesteps 50000 --no_blur --no_hist_eq

# D2: sparse + pixeldiff
python -m scripts.train_ppo --seeds 0 1 2 --total_timesteps 50000 --no_blur --no_hist_eq --termination pixeldiff --pixeldiff_thresh 45000

# D3: shaped + ocr
python -m scripts.train_ppo --seeds 0 1 2 --total_timesteps 50000 --no_blur --no_hist_eq --reward_mode shaped

# D4: shaped + pixeldiff
python -m scripts.train_ppo --seeds 0 1 2 --total_timesteps 50000 --no_blur --no_hist_eq --reward_mode shaped --termination pixeldiff --pixeldiff_thresh 45000
```

**Loop version (bash) — run all combos automatically**

```bash
for BLUR in on off; do
  for HIST in on off; do
    for REW in sparse shaped; do
      for TERM in ocr pixeldiff; do
        CMD="python -m scripts.train_ppo --seeds 0 1 2 --total_timesteps 50000"
        [ "$BLUR" = "off" ] && CMD="$CMD --no_blur"
        [ "$HIST" = "off" ] && CMD="$CMD --no_hist_eq"
        [ "$REW" = "shaped" ] && CMD="$CMD --reward_mode shaped"
        [ "$TERM" = "pixeldiff" ] && CMD="$CMD --termination pixeldiff --pixeldiff_thresh 45000"
        echo ">>> $CMD"
        eval "$CMD"
      done
    done
  done
done
```

### Recommended subset (time-efficient)

If you don’t want all 16:

* Baseline grayscale-only (D1)
* Full pipeline (A1)
* No blur (B1)
* No hist-eq (C1)
* Reward shaped (A3)
* Pixeldiff termination (A2)
* Combo (A4)

---

## 7) Evaluation & Aggregation

Evaluate **latest checkpoints** per seed and write aggregates:

```bash
# PPO
python -m scripts.evaluate --algo ppo --n_eval_episodes 20

# DQN
python -m scripts.evaluate --algo dqn --n_eval_episodes 20
```

Optional flags to **match evaluation env** to training (use when you changed ablations in training):

* `--reward_mode {sparse|shaped}`
* `--termination {ocr|pixeldiff}`
* `--no_blur`, `--no_hist_eq`
* `--ckpt_step` to load a specific checkpoint `model_{step}.zip`

**Outputs:**

* Per-seed CSVs: `results/raw/{algo}_seed_{k}.csv`
* Aggregate CSV: `results/aggregates/{algo}_aggregate.csv` (includes mean, std, 95% CI)

---

## 8) Plot Learning Curves (mean ± 95% CI)

```bash
python -m scripts.plot_curves --algo_dirs logs/ppo logs/dqn
# -> results/figures/learning_curves.png
```

> Uses SB3 CSV logs to compute mean curve and shaded 95% CI over seeds.

---

## 9) Advanced: Environment calibration (macOS)

If your auto-calibration can’t find the canvas reliably, the env supports **window-first calibration** on macOS using Quartz + AppleScript (see the code for `calibrate_via_window`, `force_window_bounds`, `canvas_thresh_percentile`).

**Programmatic example**:

```python
from envs.chrome_dino_env import ChromeDinoEnv
env = ChromeDinoEnv(
    auto_calibrate=True,
    monitor_index=0,                    # virtual "all monitors" is safest
    calibrate_via_window=True,          # use macOS window discovery
    force_window_bounds=(100,80,1100,700),  # snap Chrome to known bounds
    canvas_thresh_percentile=0.92       # adjust 0.90–0.96 if needed
)
env.debug_show_regions()                # visualize detected regions
```

> If you want these as CLI flags, pass them through in your training script by adding arguments and forwarding to `ChromeDinoEnv(...)`.

---

## 10) Troubleshooting

* **Nothing shows / wrong crop**

  * Try `monitor_index=0` (virtual full desktop).
  * Set Chrome zoom to **100%**.
  * Lower the canvas threshold (e.g., `canvas_thresh_percentile=0.90`).
  * Ensure templates exist if using template matching for `done_region`.

* **Keystrokes not working (macOS)**

  * Grant **Accessibility** permission to the terminal/Python process.
  * Some systems require running terminal as admin.

* **Screen capture empty (macOS)**

  * Grant **Screen Recording** permission.

* **`chrome://dino` won’t open**

  * Use `https://chromedino.com` as a fallback.

* **Prefer full reproducibility**

  * Consider the **Playwright** env (`envs/dino_playwright_env.py`), which screenshots the canvas element directly and sends browser key events (no OCR or OS-level input).

---

## 11) Reproducibility artifacts

* Per-seed config: `logs/{algo}/seed_{k}/run_config.json`
* Metadata (hardware/wall-clock): `logs/{algo}/seed_{k}/metadata.jsonl`
* Raw evaluation: `results/raw/`
* Aggregates: `results/aggregates/`
* Figures: `results/figures/`

---

## 12) Cite

If this repo helps your research, please cite your own paper or include acknowledgment to “Vision-only RL on Chrome Dino (benchmark + ablations framework)”.

---

Happy dashing 🦖!
# Visual-Reinforcement-Learning
