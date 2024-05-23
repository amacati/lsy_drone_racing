"""SAC agent training script for drone racing."""
from __future__ import annotations

import logging
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING

import fire
import pip._vendor.tomli as tomllib
import torch
import wandb
from munch import munchify
from safe_control_gym.utils.registration import make
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import CallbackList, EvalCallback
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.env_util import SubprocVecEnv, make_vec_env
from stable_baselines3.common.logger import Logger

from lsy_drone_racing.constants import FIRMWARE_FREQ
from lsy_drone_racing.utils import load_config
from lsy_drone_racing.utils.sb3 import WandbLogger, WandbSuccessCallback
from lsy_drone_racing.wrapper import DroneRacingWrapper, RewardWrapper

if TYPE_CHECKING:
    from wandb.sdk.wandb_run import Run

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


def init_run(init_wandb: bool = True) -> tuple[Run, dict]:
    """Initialize the wandb run and load the configuration."""
    with open(Path(__file__).parents[1] / "config/train_sac.toml", "rb") as f:
        config = munchify(tomllib.load(f))
    if getattr(config.rng, "seed", None) is not None:
        torch.manual_seed(config.rng.seed)
    torch.backends.cudnn.benchmark = False  # Avoid selecting different algorithms on different runs
    run = None
    if init_wandb:
        save_path = Path(__file__).parents[1] / "saves"
        save_path.mkdir(exist_ok=True, parents=True)
        with open(Path(__file__).resolve().parents[1] / "secrets/wandb_api_key.secret", "r") as f:
            wandb_api_key = f.read()
        wandb.login(key=wandb_api_key)
        run = wandb.init(
            project=config.wandb.project,
            entity=config.wandb.entity,
            group=config.wandb.group,
            config=config,
            dir=save_path,
        )
    return run, config


def main(config: str = "config/getting_started.yaml", init_wandb: bool = True):
    """Create the environment, check its compatibility with sb3, and run a PPO agent."""
    logging.basicConfig(level=logging.INFO)
    root_path = Path(__file__).resolve().parents[1]
    config_path = root_path / config
    env = create_race_env(config_path=config_path, gui=False)
    check_env(env)  # Sanity check to ensure the environment conforms to the sb3 API

    run, cfg = init_run(init_wandb=init_wandb)

    env = make_vec_env(
        lambda: RewardWrapper(create_race_env(config_path)),
        cfg.n_envs,
        vec_env_cls=SubprocVecEnv,
        vec_env_kwargs={"start_method": "spawn"},
    )

    model = SAC(
        "MlpPolicy",
        env,
        learning_rate=cfg.lr,
        buffer_size=int(cfg.buffer_size),
        batch_size=cfg.batch_size,
        learning_starts=cfg.learning_starts,
        train_freq=cfg.train_freq,
        gradient_steps=cfg.gradient_steps,
        ent_coef=cfg.ent_coef,
        target_update_interval=cfg.target_update_interval,
        tau=cfg.tau,
        gamma=cfg.gamma,
        verbose=1,
    )
    if run is not None:
        model.set_logger(Logger(folder=None, output_formats=[WandbLogger(verbose=1)]))

    eval_env = make_vec_env(
        lambda: RewardWrapper(create_race_env(config_path)),
        cfg.n_envs,
        vec_env_cls=SubprocVecEnv,
        vec_env_kwargs={"start_method": "spawn"},
    )

    eval_callback = EvalCallback(
        eval_env=eval_env,
        best_model_save_path=root_path / "saves/sac_drone_racing/best_model",
        eval_freq=cfg.eval_freq,
        n_eval_episodes=cfg.n_eval_episodes,
        callback_on_new_best=None,
        verbose=1,
    )

    callbacks = [eval_callback]
    if run:
        callbacks.append(WandbSuccessCallback("task_completed"))
    model.learn(total_timesteps=cfg.n_timesteps, callback=CallbackList(callbacks))
    model.save(root_path / "saves" / "sac_drone_racing")


if __name__ == "__main__":
    fire.Fire(main)
