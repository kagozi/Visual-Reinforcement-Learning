import os
import time
import json
from typing import Optional
from stable_baselines3.common.callbacks import BaseCallback

class CheckpointEveryN(BaseCallback):
    def __init__(self, check_freq: int, save_dir: str, verbose: int = 0):
        super().__init__(verbose)
        self.check_freq = check_freq
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:
            path = os.path.join(self.save_dir, f"model_{self.n_calls}.zip")
            self.model.save(path)
        return True

class WallClockLogger(BaseCallback):
    """
    Records wall-clock start/end, algo, seed, env config → JSONL in save_dir/metadata.jsonl
    """
    def __init__(self, save_dir: str, algo: str, seed: int, env_cfg: dict, verbose: int = 0):
        super().__init__(verbose)
        self.save_dir = save_dir
        self.algo = algo
        self.seed = seed
        self.env_cfg = env_cfg
        self.t0 = None
        os.makedirs(self.save_dir, exist_ok=True)

    def _on_training_start(self) -> None:
        self.t0 = time.time()
        self._dump(gpu=None, when="start")

    def _on_training_end(self) -> None:
        self._dump(gpu=None, when="end")

    def _on_step(self) -> bool:
        # No-op logger per step; required to satisfy BaseCallback's abstract method.
        return True

    def _dump(self, gpu: Optional[str], when: str) -> None:
        meta = dict(
            when=when,
            algo=self.algo,
            seed=self.seed,
            env_config=self.env_cfg,
            start_time=self.t0,
            now=time.time(),
        )
        try:
            import torch
            meta["cuda_available"] = torch.cuda.is_available()
            if torch.cuda.is_available():
                meta["gpu_name"] = torch.cuda.get_device_name(0)
                meta["gpu_capability"] = torch.cuda.get_device_capability(0)
        except Exception:
            pass

        out = os.path.join(self.save_dir, "metadata.jsonl")
        with open(out, "a") as f:
            f.write(json.dumps(meta) + "\n")
