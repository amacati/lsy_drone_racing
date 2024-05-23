"""SAC agent training script for drone racing."""
from __future__ import annotations

import logging
import time
from functools import partial
from pathlib import Path

import fire
from safe_control_gym.utils.registration import make
from stable_baselines3 import SAC

from lsy_drone_racing.constants import FIRMWARE_FREQ
from lsy_drone_racing.utils import load_config
from lsy_drone_racing.wrapper import DroneRacingWrapper, RewardWrapper

logger = logging.getLogger(__name__)


def create_race_env(config_path: Path, gui: bool = False) -> DroneRacingWrapper:
    """Create the drone racing environment."""
    # Load configuration and check if firmare should be used.
    config = load_config(config_path)
    # Overwrite config options
    config.quadrotor_config.gui = gui
    CTRL_FREQ = config.quadrotor_config["ctrl_freq"]
    # Create environment
    assert config.use_firmware, "Firmware must be used for the competition."
    pyb_freq = config.quadrotor_config["pyb_freq"]
    assert pyb_freq % FIRMWARE_FREQ == 0, "pyb_freq must be a multiple of firmware freq"
    config.quadrotor_config["ctrl_freq"] = FIRMWARE_FREQ
    env_factory = partial(make, "quadrotor", **config.quadrotor_config)
    firmware_env = make("firmware", env_factory, FIRMWARE_FREQ, CTRL_FREQ)
    return DroneRacingWrapper(firmware_env, terminate_on_lap=True)


def main(
    config: str = "config/getting_started.yaml",
    gui: bool = True,
    n_tests: int = 1,
    delay: float = 1 / FIRMWARE_FREQ,
):
    """Create the environment, check its compatibility with sb3, and run a PPO agent."""
    logging.basicConfig(level=logging.INFO)
    root_path = Path(__file__).resolve().parents[1]
    config_path = root_path / config
    env = RewardWrapper(create_race_env(config_path=config_path, gui=gui))

    model = SAC.load(Path(__file__).parents[1] / "saves/sac_drone_racing/best_model/best_model.zip")
    success = []
    for i in range(n_tests):
        obs, info = env.reset()
        terminated, truncated = False, False
        while not (terminated or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            time.sleep(delay)
        success.append(info["task_completed"])
    print(f"Success rate: {sum(success) / n_tests:.2f}")


if __name__ == "__main__":
    fire.Fire(main)
