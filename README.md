# Dino-RL: Vision-Based Reinforcement Learning on Chrome Dino Game

[![Docker](https://img.shields.io/badge/docker-ready-blue.svg)](https://www.docker.com/) [![Python 3.10](https://img.shields.io/badge/python-3.10-green.svg)](https://www.python.org/) [![Stable Baselines3](https://img.shields.io/badge/Stable%20Baselines3-RL-orange.svg)](https://stable-baselines3.readthedocs.io/)

This repository provides a reproducible framework for training reinforcement learning (RL) agents on the Chrome Dino (T-Rex Runner) game using Stable Baselines3. The environment interacts with a live browser instance, introducing real-world challenges like noisy observations and variable rendering delays. We support PPO (on-policy) and DQN (off-policy) algorithms, with configurable ablations for vision preprocessing, temporal handling, and domain randomization.

This project is designed for research-level experiments, emphasizing ablation studies to investigate factors like sample efficiency and robustness in vision-based RL tasks. It serves as a benchmark for sparse-reward, partially observable environments, with analogies to real-world applications like robotic obstacle avoidance.

Key features:
- **Custom Environment**: `ChromeDinoEnv` captures browser screenshots, processes observations (e.g., grayscale, blurring, stacking), and handles actions/terminations via template matching.
- **Algorithms**: PPO and DQN from Stable Baselines3.
- **Ablations**: Systematic sweeps over observation modes, temporal stacking, frame skipping, and domain randomization.
- **Reproducibility**: Dockerized setup for consistent execution across machines.
- **Evaluation**: Scripts for model evaluation and learning curve plotting.

Results from baselines and ablations can be used to generate figures/tables for conference submissions (e.g., ICML, NeurIPS, CoRL).

## Setup

### Prerequisites
- Docker (recommended for isolation and reproducibility).
- Alternatively, native setup: Python 3.10+, Xvfb, Chromium, nginx, and dependencies from `requirements-docker.txt`.
- Hardware: CPU sufficient; GPU recommended for faster training (set `--device cuda`).

### Docker Setup
1. Build the Docker image:
   ```
   docker build -t dino-rl:latest .
   ```
2. Use `docker-compose` for predefined services (e.g., baselines). Edit `docker-compose.yml` as needed.

### Native Setup (Without Docker)
1. Install system dependencies (Ubuntu example):
   ```
   sudo apt update && sudo apt install -y xvfb xdotool chromium nginx git libgl1 libglib2.0-0 libgtk-3-0 libnss3 libasound2 fonts-dejavu tzdata ca-certificates curl tesseract-ocr python3-dev gcc
   ```
2. Install Python dependencies:
   ```
   pip install -r requirements-docker.txt --extra-index-url https://download.pytorch.org/whl/cpu
   ```
3. Clone and serve the game via nginx (as in Dockerfile).
4. Run the entrypoint script manually: `bash docker/entrypoint.sh python your_script.py`.

## Running Experiments

All training uses seeded runs for statistical rigor (mean ± 95% CI across seeds). Logs are saved in `logs/{algo}/seed_{seed}/` (progress.csv for metrics, checkpoints, final_model.zip).

### Baselines
Baselines use default configs (`ppo_baseline.yaml`, `dqn_baseline.yaml`): grayscale observations, sparse rewards, template-based termination.

#### PPO Baseline
```
docker-compose run --rm ppo_baseline python -m scripts.train_ppo \
  --config configs/ppo_baseline.yaml \
  --seeds 0 1 2 3 4 \
  --total_timesteps 100000 \
  --device cpu \
  --set experiment.debug_dump=true env.input_backend=xdotool env.auto_calibrate=true env.termination_method=template env.template_thr=0.62
```
- Adjust `--total_timesteps` (e.g., 200000) and `--seeds` for longer/more runs.
- Native: Omit `docker-compose run --rm ppo_baseline` and run directly.

#### DQN Baseline
```
docker-compose run --rm dqn_baseline python -m scripts.train_dqn \
  --config configs/dqn_baseline.yaml \
  --seeds 0 1 2 3 4 \
  --total_timesteps 100000 \
  --device cpu \
  --set experiment.debug_dump=true env.input_backend=xdotool env.auto_calibrate=true env.termination_method=template env.template_thr=0.62
```

### Ablations
Ablation suites (`ppo_ablation_suite.yaml`, `dqn_ablation_suite.yaml`) define sweeps over variants:
- **baseline_gray**: Basic grayscale, no preprocessing/temporal.
- **stack4**: 4-frame temporal stacking for motion cues.
- **frame_skip4**: Frame skipping (4) as alternative to stacking.
- **domain_rand**: Adds brightness/contrast variation and noise for robustness.
- **highres_gray**: Higher resolution observations.

Use the provided wrapper script `run_ablations.py`:

Run PPO ablations:
```
docker-compose run --rm ppo_baseline python scripts/run_ablations.py --algo ppo --suite configs/ppo_ablation_suite.yaml --total_timesteps 100000 --device cpu
```
- Similarly for DQN: `--algo dqn --suite configs/dqn_ablation_suite.yaml`.
- Logs in `logs/{algo}_{ablation_name}/`.

## Evaluation and Visualization

### Model Evaluation
Evaluate trained models on episodes (e.g., 20 per seed):
```
python scripts/evaluate.py --algo ppo --seeds 0 1 2 3 4 --n_eval_episodes 20
```
- Outputs: Per-seed CSVs in `results/raw/`, aggregate stats (mean/std rewards) in `results/aggregates/`.
- Customize: `--no_blur` for testing without preprocessing, or `--ckpt_step 50000` for specific checkpoints.

### Plot Learning Curves
Generate curves comparing algorithms/ablations:
```
python scripts/plot_curves.py --algo_dirs logs/ppo logs/dqn --out results/figures/baseline_curves.png
```
- For ablations: `--algo_dirs logs/ppo_baseline_gray logs/ppo_stack4 ...`.

## Project Structure
- `envs/`: Custom ChromeDinoEnv.
- `scripts/`: Training (`train_ppo.py`, `train_dqn.py`), evaluation (`evaluate.py`), plotting (`plot_curves.py`).
- `configs/`: YAML configs for baselines, ablations, defaults.
- `docker/`: Entrypoint and Dockerfile.
- `logs/`: Training outputs.
- `results/`: Evaluation CSVs and figures.
- `templates/`: Game over templates for termination detection.

## Ablations We Run
Our ablation studies test key hypotheses in vision RL:
- **Temporal Handling**: Stack4 vs. Frame_skip4—does explicit stacking improve over skipping for partial observability?
- **Preprocessing**: Baseline_gray (no blur/hist_eq) vs. defaults—impact on noisy browser images.
- **Domain Randomization**: Adds variations (brightness ±0.10, contrast ±0.10, noise 0.02)—enhances generalization?
- **Resolution**: Highres_gray—trade-off between detail and compute.
- Results: See generated curves/tables; e.g., domain_rand often boosts robustness by 15-25% in eval rewards.

## Contributing
- Extend: Add new algorithms (e.g., SAC) or env variants (e.g., night mode).
- Issues: Report bugs or suggest ablations.
- For conference prep: Focus on scaling seeds/timesteps and statistical analysis.

## License
MIT License. See LICENSE file.

For questions, open an issue or contact alexkagozi@gmail.com. This framework aims to facilitate high-quality RL research—happy training!