from __future__ import annotations

import logging
from typing import Any

import numpy as np
import wandb
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import KVWriter, SeqWriter

logger = logging.getLogger(__name__)


class WandbLogger(KVWriter, SeqWriter):
    def __init__(self, verbose: int = 0):
        assert wandb.run is not None, "Wandb run must be initialized before using WandbLogger."
        self._log_to_stdout = verbose > 0

    def write(
        self, key_values: dict[str, Any], key_excluded: dict[str, tuple[str, ...]], step: int = 0
    ):
        wandb.run.log(key_values, step=step)
        if self._log_to_stdout:
            logger.info("\n" * 2 + "\n".join(f"{k}: {v}" for k, v in key_values.items()))

    def write_sequence(self, sequence: list[str]):
        pass


class WandbSuccessCallback(BaseCallback):
    def __init__(self, success_key: str):
        super().__init__(0)
        assert wandb.run is not None, "Wandb run must be initialized before using WandbLogger."
        self.success_key = success_key

    def _on_step(self) -> bool:
        # Check if the episode is done
        if any(self.locals["dones"]):
            infos = [self.locals["infos"][i] for i in np.where(self.locals["dones"])[0]]
            successes = [x[self.success_key] for x in infos]
            # Taking the mean is slightly misleading. If two episodes are done, and one is a success
            # and one is a failure, the mean has more significance than if only one env was done. We
            # do not track this in WandB, so the stats are off. However, we expect this to average
            # out over time.
            success = sum(successes) / len(successes)
            wandb.log({"rollout/success": success}, step=self.num_timesteps)
        return True
