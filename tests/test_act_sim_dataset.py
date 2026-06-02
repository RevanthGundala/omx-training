from pathlib import Path

import h5py
import numpy as np
import torch

from benchmarks.act_sim.dataset import (
    ACTION,
    ACTION_IS_PAD,
    OBSERVATION_STATE,
    ACTHDF5Dataset,
    EpisodeFile,
    make_act_hdf5_dataloaders,
)
from configs.act.sim_transfer_cube_smoke import config as smoke_config
from dataclasses import replace


def write_episode(path: Path, episode_len: int = 5) -> None:
    with h5py.File(path, "w") as root:
        root.attrs["sim"] = True
        obs = root.create_group("observations")
        images = obs.create_group("images")
        obs.create_dataset("qpos", data=np.arange(episode_len * 14, dtype=np.float32).reshape(episode_len, 14))
        obs.create_dataset("qvel", data=np.zeros((episode_len, 14), dtype=np.float32))
        images.create_dataset("top", data=np.zeros((episode_len, 8, 8, 3), dtype=np.uint8))
        root.create_dataset("action", data=np.ones((episode_len, 14), dtype=np.float32))


def test_act_hdf5_dataset_returns_chunked_actions_with_padding(tmp_path):
    episode_path = tmp_path / "episode_0.hdf5"
    write_episode(episode_path)

    dataset = ACTHDF5Dataset(
        [EpisodeFile(0, episode_path)],
        camera_names=("top",),
        chunk_size=3,
        deterministic_start=True,
        seed=0,
    )

    batch = dataset[0]

    assert batch[OBSERVATION_STATE].shape == (14,)
    assert batch["observation.images.top"].shape == (8, 8, 3)
    assert batch["observation.images.top"].dtype == torch.uint8
    assert batch[ACTION].shape == (3, 14)
    assert batch[ACTION_IS_PAD].tolist() == [False, False, True]


def test_make_act_hdf5_dataloaders_returns_training_batches(tmp_path):
    write_episode(tmp_path / "episode_0.hdf5")
    write_episode(tmp_path / "episode_1.hdf5")
    config = replace(
        smoke_config,
        benchmark_dataset_dir=str(tmp_path),
        benchmark_num_episodes=2,
        benchmark_episode_len=5,
        chunk_size=3,
        train_split=0.5,
        batch_size=1,
    )

    train_loader, val_loader, norm_stats = make_act_hdf5_dataloaders(config, torch.device("cpu"))
    train_batch = next(iter(train_loader))

    assert train_batch[OBSERVATION_STATE].shape == (1, 14)
    assert train_batch[ACTION].shape == (1, 3, 14)
    assert train_batch[ACTION_IS_PAD].shape == (1, 3)
    assert val_loader is not None
    assert norm_stats[OBSERVATION_STATE]["mean"].shape == (14,)
    assert norm_stats[ACTION]["std"].min().item() >= 1e-2 - 1e-8
