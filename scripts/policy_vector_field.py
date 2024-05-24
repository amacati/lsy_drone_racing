"""Policy analysis script."""
from __future__ import annotations

import logging
from functools import partial
from pathlib import Path

import fire
import matplotlib.pyplot as plt
import numpy as np
from safe_control_gym.utils.registration import make
from stable_baselines3 import SAC, TD3
from stable_baselines3.common.env_util import DummyVecEnv, make_vec_env
from stable_baselines3.common.vec_env import VecNormalize

from lsy_drone_racing.constants import FIRMWARE_FREQ, GateDesc
from lsy_drone_racing.utils import load_config
from lsy_drone_racing.wrapper import DroneRacingWrapper, RewardWrapper

logger = logging.getLogger(__name__)

algos = {"sac": SAC, "td3": TD3}


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
    return RewardWrapper(DroneRacingWrapper(firmware_env, terminate_on_lap=True))


def plot_policy_field(
    action: np.ndarray, x: np.ndarray, y: np.ndarray, obs: np.ndarray, info: dict
):
    fig, ax = plt.subplots()
    assert action.shape == (len(x), 4)
    assert x.shape == y.shape
    ax.quiver(x, y, action[:, 0], action[:, 1])
    ax.add_patch(plt.Circle(obs[:2], 0.03, color="g"))
    for obstacle in info["obstacles_pose"]:
        ax.add_patch(plt.Circle(obstacle[:2], 0.05, color="r"))
    for gate in info["gates_pose"]:
        ax.add_patch(
            plt.Rectangle(
                gate[:2] - np.array([GateDesc.edge / 2, 0]),
                GateDesc.edge,
                0.02,
                rotation_point="center",
                angle=gate[5] * 180 / np.pi,
                color="b",
            )
        )
    ax.set_aspect("equal")
    plt.show()


def main(config: str = "config/getting_started.yaml", algo: str = "sac"):
    """Create the environment, check its compatibility with sb3, and run a PPO agent."""
    algo = algo.lower()
    assert algo in algos, f"Algorithm {algo} not supported. Choose from {algos.keys()}."
    logging.basicConfig(level=logging.INFO)
    root_path = Path(__file__).resolve().parents[1]
    save_path = root_path / "saves" / algo
    config_path = root_path / config
    env = make_vec_env(
        lambda: create_race_env(config_path=config_path, gui=False), 1, vec_env_cls=DummyVecEnv
    )
    env = VecNormalize.load(save_path / "env.pkl", env)

    obs = env.reset()
    _, _, _, info = env.step(np.array([[0, 0, 0, 0]], np.float32))
    obs = env.unnormalize_obs(obs)[0]
    obs[2] = 0.5
    x, y = np.meshgrid(np.linspace(-2, 2, 100), np.linspace(-2, 2, 100))
    x, y = x.flatten(), y.flatten()
    obs_batch = np.stack([obs] * len(x))
    obs_batch[:, 0] = x
    obs_batch[:, 1] = y
    model = algos[algo].load(save_path / "model.zip")
    action, _ = model.predict(env.normalize_obs(obs_batch), deterministic=True)
    plot_policy_field(action, x, y, obs, info[0])


if __name__ == "__main__":
    fire.Fire(main)
