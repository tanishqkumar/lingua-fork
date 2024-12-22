# Copyright (c) Meta Platforms, Inc. and affiliates.

import json
import logging
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
import time
from typing import Callable, List, Optional, Tuple

import shutil
import torch
import torch.distributed as dist
from copy import deepcopy
import torch.distributed.checkpoint as dcp
import torch.nn as nn
import torch.optim.optimizer
from omegaconf import OmegaConf
from torch.distributed._tensor import DeviceMesh
from torch.distributed.checkpoint.format_utils import dcp_to_torch_save
from torch.distributed.checkpoint.state_dict import (
    get_model_state_dict,
    get_state_dict,
    set_state_dict,
)
from torch.distributed.checkpoint.format_utils import (
    torch_save_to_dcp,
    dcp_to_torch_save,
)
import torch.optim.optimizer
from multiprocessing import Process

from lingua.distributed import get_is_master

logger = logging.getLogger("CHECKPOINT")

FOLDER_NAME = "{:010d}"
RE_FOLDER = r"\d{10}"

RE_CKPT = r"__\d_\d\.distcp"

CONSOLIDATE_FOLDER = "consolidated"
CONSOLIDATE_NAME = "consolidated.pth"

CONFIG_NAME = "params.json"
TRAIN_STATE_NAME = "train_state_{:05d}.json"
RE_DIGITS = re.compile(r"\d+")


@dataclass
class SaveEvery:
    every: int = 1000
    keep: int = 0


@dataclass
class CheckpointArgs:
    dump: SaveEvery = field(default_factory=SaveEvery)
    eval: SaveEvery = field(default_factory=SaveEvery)
    path: Optional[str] = None
    init_ckpt_path: Optional[str] = None
    continue_training_from_init: bool = False
    # "shm" or "None". shm uses /dev/shm as a fast checkpoint target and async copies to the ckpt path. None is the usual sync setup, which is safe but slow.
    async_save_mode: Optional[str] = None
    async_cleanup: bool = False  # New field to control async cleanup
    thread_debug: bool = True # Enables debug print statements within save / cleanup threads.
    


def _get_key_step(name: str):
    return int(re.findall(RE_DIGITS, name)[-1])


def consolidate_checkpoints(ckpt_dir: str):
    """
    Consolidates all FSDP checkpoints in a directory to a single file
    Consolidate checkpoint is saved in a subdirectory of ckpt_dir

    Parameters:
        ckpt_dir: str - path to the directory containing the checkpoints

    Returns the path to the consolidated checkpoint
    """
    consolidate_path = Path(ckpt_dir) / CONSOLIDATE_FOLDER
    if not (consolidate_path / CONSOLIDATE_NAME).exists():
        consolidate_path.mkdir(exist_ok=True)
        logger.info(f"Consolidating to: {str(consolidate_path)}")
        dcp_to_torch_save(ckpt_dir, str(consolidate_path / CONSOLIDATE_NAME))
        (consolidate_path / CONFIG_NAME).write_text(
            (Path(ckpt_dir) / CONFIG_NAME).read_text()
        )
        logger.info("Consolidated !")
    return consolidate_path


def load_from_checkpoint(
    ckpt_dir: str,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    model_key: str = "model",
    optim_key: str = "optim",
):
    if not (Path(ckpt_dir) / ".metadata").exists():
        raise ValueError(
            f"Please convert the checkpoint distcp format using `torch.distributed.checkpoint.format_utils.torch_save_to_dcp` before loading it"
        )

    state_dict = {}
    if optimizer is not None:
        state_dict[model_key], state_dict[optim_key] = get_state_dict(model, optimizer)
    else:
        state_dict[model_key] = get_model_state_dict(model)
        if model_key == "":  # If only loading a model directly, the key should be empty
            state_dict = state_dict.pop(model_key)

    dcp.load(state_dict, checkpoint_id=ckpt_dir)
    logger.info(f"Loaded state dict: {state_dict}")


class CheckpointManager:
    def __init__(self, args: CheckpointArgs):
        self.path = args.path
        self.dump_every = args.dump
        self.eval_every = args.eval
        self.init_ckpt_path = args.init_ckpt_path
        self.continue_training_from_init = args.continue_training_from_init
        self.async_save_mode = args.async_save_mode
        self.async_cleanup = args.async_cleanup
        self.thread_debug = args.thread_debug
        assert os.path.exists(self.path), f"Path {self.path} does not exist and needs to be created before using CheckpointManager (use instantiate_and_make_dir)"
        self.existing_saves = self.get_existing_saves()
        # used for async saving
        self._shm_save_hash = hex(hash(str(time.time()+dist.get_rank())))[-8:]
        self._save_process = None
        self._cleanup_process = None

    def __getstate__(self):
        # capture what is normally pickled
        state = self.__dict__.copy()

        # remove unpicklable/problematic variables 
        state['_save_process'] = None
        state['_cleanup_process'] = None
        return state

    def get_existing_saves(self) -> List[Path]:
        folders = [
            p
            for p in Path(self.path).iterdir()
            if p.is_dir() and re.match(RE_FOLDER, p.name)
        ]
        folders.sort(key=lambda p: _get_key_step(p.name))
        return folders

    def remove_folders(self, folders_to_remove: Tuple[str], rank: int):
        """Process target for folder removal"""
        if self.thread_debug:
            print(f"Rank {rank} starting cleanup of {len(folders_to_remove)} folders")
        for folder_str in folders_to_remove:
            folder = Path(folder_str)
            for file in folder.iterdir():
                if file.is_file():
                    file.unlink()
                elif file.is_dir():
                    assert file.name in [CONSOLIDATE_FOLDER]
                    for f in file.iterdir():
                        f.unlink()
                    file.rmdir()
            folder.rmdir()
        if self.thread_debug:
            print(f"Completed cleanup")

    def clean_up(self):
        # Wait for any previous cleanup to complete
        if self.async_cleanup and self._cleanup_process is not None:
            logger.info(f"Rank {dist.get_rank()} waiting for previous cleanup to complete...")
            self._cleanup_process.join()
            self._cleanup_process = None
        dist.barrier()

        logger.info("Cleaning up checkpoints...")
        dump_folders = []
        eval_complete_folders = []
        eval_incomplete_folders = []
        all_eval_folders = []
        other_folders = []
        for p in self.existing_saves:
            is_dump = _get_key_step(p.name) % self.dump_every.every == 0
            is_eval = _get_key_step(p.name) % self.eval_every.every == 0
            if is_dump:
                dump_folders.append(p)
            if is_eval:  # wait for evals to complete before removing them!
                if (p / "eval.complete").exists():
                    eval_complete_folders.append(p)
                else:
                    eval_incomplete_folders.append(p)
                all_eval_folders.append(p)
            if not (is_dump or is_eval):
                other_folders.append(p)

        logger.info(f"Dump folders: {dump_folders}")
        logger.info(f"Eval complete folders: {eval_complete_folders}")
        logger.info(f"Eval incomplete folders: {eval_incomplete_folders}")
        logger.info(f"All eval folders: {all_eval_folders}")
        logger.info(f"Other folders: {other_folders}")

        keep_with_incompletes = set()
        if self.dump_every.keep > 0:
            dump_folders = dump_folders[-self.dump_every.keep :]
        if self.eval_every.keep > 0:
            eval_folders_to_keep = set(all_eval_folders[-self.eval_every.keep :])
            keep_with_incompletes = set(eval_folders_to_keep) | set(
                eval_incomplete_folders
            )
            if not eval_folders_to_keep.issubset(keep_with_incompletes):
                diff = eval_folders_to_keep - keep_with_incompletes
                logger.warning(
                    f"WARNING: Attempted to clean up eval folders, but some are not yet complete. Disk usage may be higher than expected. Incomplete folders: {diff}"
                )

        folder_to_keep = set(other_folders + dump_folders) | keep_with_incompletes
        folder_to_remove = set(self.existing_saves) - folder_to_keep

        logger.info(f"Removing folders: {folder_to_remove}")

        if dist.get_rank() == 0 and len(folder_to_remove) > 0:
            str_folders_to_remove = tuple([str(folder) for folder in folder_to_remove])
            print(f"Removing folders: {str_folders_to_remove}")
            if self.async_cleanup:
                self._cleanup_process = Process(
                    target=self.remove_folders,
                    args=(str_folders_to_remove, dist.get_rank())
                )
                self._cleanup_process.start()
            else:
                self.remove_folders(str_folders_to_remove, dist.get_rank())

        dist.barrier()

        self.existing_saves = list(folder_to_keep)
        self.existing_saves.sort(key=lambda p: _get_key_step(p.name))

    def get_last_step_path(self, dp_rank: int = 0) -> Optional[Path]:
        path = None
        for p in reversed(self.existing_saves):
            if (p / TRAIN_STATE_NAME.format(dp_rank)).is_file():
                path = p
                break
        return path

    def _create_folder(self, base_path: Path, folder_name: str) -> Path:
        folder = base_path / folder_name
        if get_is_master():
            folder.mkdir(parents=False, exist_ok=True)
        if dist.is_initialized():
            dist.barrier()
        return folder

    def _get_dp_tp_mesh(
        self, device_mesh: Optional[DeviceMesh] = None
    ) -> Tuple[int, int]:
        dp_rank = 0
        tp_rank = 0
        if device_mesh is not None:
            if "dp_replicate" in device_mesh.mesh_dim_names:
                dp_rank = device_mesh.get_local_rank("dp_replicate")
                if "dp_shard" in device_mesh.mesh_dim_names:
                    dp_rank = dp_rank * device_mesh[
                        "dp_replicate"
                    ].size() + device_mesh.get_local_rank("dp_shard")
            if "tp" in device_mesh.mesh_dim_names:
                tp_rank = device_mesh.get_local_rank("tp")
        return dp_rank, tp_rank

    @torch.no_grad()
    def get_state_dict(self, model, optimizer):
        model_sd, optim_sd = get_state_dict(model, optimizer)
        return {"model": model_sd, "optim": optim_sd}

    def _async_shm_save(self, state_dict_path, rank, save_dir):
        """Process target for async state dict saving"""
        if self.thread_debug:
            print(f"Rank {rank} saving state dict to {save_dir}")
        # Copy checkpoint files from temp location to final save dir
        for file in Path(state_dict_path).glob("*"):
            shutil.move(file, Path(save_dir) / file.name)
        # Clean up temp directory
        shutil.rmtree(state_dict_path)
        # Touch save_dir / ckpt_{rank}.complete to mark done
        (Path(save_dir) / f"ckpt_{rank}.complete").touch()
        if self.thread_debug:
            print(f"Rank {rank} saved state dict to {save_dir}")

    def save(self, model, optimizer, train_state, config, device_mesh: Optional[DeviceMesh] = None, force_sync_save: bool = False) -> bool:
        # Wait for any previous save to complete if async
        if self.async_save_mode is not None: 
            if self._save_process is not None:
                logger.info(f"Rank {dist.get_rank()} waiting for previous save to complete...")
                self._save_process.join()
        dist.barrier()  # Ensure all ranks are ready to start new save
        
        # Create save directory and get state dict
        path = Path(self.path)
        curr_save_dir = self._create_folder(path, FOLDER_NAME.format(train_state.step))
        logger.info(f"Saving to: {str(curr_save_dir)}")

        state_dict = self.get_state_dict(model, optimizer)

        if self.async_save_mode == "shm" and (not force_sync_save):
            # Launch async save
            ckpt_path = Path('/dev/shm/')
            tmp_folder = f"{self._shm_save_hash}"
            ckpt_tmp_dir = self._create_folder(ckpt_path, tmp_folder)
            dcp.save(state_dict, checkpoint_id=ckpt_tmp_dir)
            self._save_process = Process(
                target=self._async_shm_save,
                args=(ckpt_tmp_dir, dist.get_rank(), curr_save_dir)
            )
            self._save_process.start()
        else:
            dcp.save(state_dict, checkpoint_id=curr_save_dir)
            # Touch save_dir / ckpt_{rank}.complete to mark done
            (Path(curr_save_dir) / f"ckpt_{dist.get_rank()}.complete").touch()

        # Handle the rest synchronously since they're quick
        if get_is_master():
            with open(curr_save_dir / CONFIG_NAME, "w") as f:
                json.dump(
                    OmegaConf.to_container(OmegaConf.structured(config), resolve=True),
                    f,
                )

        dp_rank, tp_rank = self._get_dp_tp_mesh(device_mesh)
        if tp_rank == 0:
            train_state_name = TRAIN_STATE_NAME.format(dp_rank)
            logger.info(f"Saving train state to: {str(curr_save_dir / train_state_name)}")
            with open(curr_save_dir / train_state_name, "w") as f:
                json.dump(train_state.state_dict(), f)
            logger.info("Train state saved!")

        self.existing_saves.append(curr_save_dir)
        self.clean_up()

        if dist.is_initialized():
            dist.barrier()
        return True

    @torch.no_grad()
    def load(
        self,
        model: nn.Module,
        optimizer,
        train_state,
        device_mesh: DeviceMesh,
        path: Optional[Path] = None,
    ):
        dp_rank, tp_rank = self._get_dp_tp_mesh(device_mesh)
        # Loading tries to load the provided path, if not available the last saved step and finally from the init path
        path = path or self.get_last_step_path(dp_rank=dp_rank)
        # If none of those are available don't do anything
        if path is None:
            # If no checkpoints exist do nothing
            return

        # Only load train state if it's provided, the files exist and we're not loading from init path
        train_state_name = TRAIN_STATE_NAME.format(dp_rank)
        logger.info("Reloading train state")
        with open(path / train_state_name, "r") as f:
            train_state_dict = json.load(f)
        train_state.load_state_dict(train_state_dict)
        logger.info("Train state reloaded")

        logger.info(f"Loading from: {str(path)}")
        state_dict = self.get_state_dict(
            model=model,
            optimizer=optimizer,
        )
        dcp.load(state_dict, checkpoint_id=path)
        logger.info("State dict loaded.")

        logger.info("Reloading model and optim")

        set_state_dict(
            model,
            optimizer,
            model_state_dict=state_dict["model"],
            optim_state_dict=state_dict["optim"],
        )
        logger.info("Model and optim reloaded")

    @classmethod
    def instantiate_and_make_dir(cls, args: CheckpointArgs):
        if get_is_master():
            os.makedirs(args.path, exist_ok=True)
        dist.barrier()

        return cls(args)
    

    def wipe_shm(self):
        if self.async_save_mode == "shm":
            if self._save_process is not None:
                self._save_process.terminate()
            shm_base = '/dev/shm'
            if os.path.exists(shm_base):
                for item in os.listdir(shm_base):
                    if item.startswith(self._shm_save_hash):
                        full_path = os.path.join(shm_base, item)
                        if os.path.isdir(full_path):
                            shutil.rmtree(full_path)
                        else:
                            os.remove(full_path)

    
    def __del__(self):
        if self._save_process is not None:
            self._save_process.join()
        # clean up shm
        if self.async_save_mode == "shm":
            self.wipe_shm()
        # Update destructor to also wait for cleanup
        if self._cleanup_process is not None:
            self._cleanup_process.join()
        
