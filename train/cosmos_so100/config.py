"""Hydra configuration overlay for SO-100 Cosmos-Predict2.5 post-training."""
from __future__ import annotations

import copy
import os

from hydra.core.config_store import ConfigStore
from megatron.core import parallel_state
from torch.utils.data import DataLoader, DistributedSampler

from cosmos_predict2._src.imaginaire.lazy_config import LazyCall as L
from cosmos_predict2._src.imaginaire.lazy_config import LazyDict
from cosmos_predict2._src.predict2.action.configs.action_conditioned.config import (
    make_config as make_cosmos_config,
)
from cosmos_predict2._src.predict2.action.configs.action_conditioned.net import (
    COSMOS_V1_2B_NET_MININET_ACTION_CHUNK,
)
from cosmos_predict2.config import MODEL_CHECKPOINTS, ModelKey, ModelVariant
from cosmos_so100.callback import LocalVideoEveryNDrawSample
from cosmos_so100.checkpointer import LoadOnlyDistributedCheckpointer
from cosmos_so100.dataset import CosmosSO100Dataset
from cosmos_so100.model import SO100ActionRankingModel
from cosmos_so100.network import SO100AdapterActionChunkDiT
from cosmos_so100.optimizer import get_joint_adapter_optimizer

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DATA_ROOT = os.environ.get("INHA_DATA_ROOT", os.path.join(REPO, "open", "data", "train"))
HEIGHT = int(os.environ.get("COSMOS_HEIGHT", "256"))
WIDTH = int(os.environ.get("COSMOS_WIDTH", "320"))
ACTION_MODE = os.environ.get("COSMOS_ACTION_MODE", "delta")
V2_ACTION_MODE = os.environ.get("COSMOS_ACTION_MODE_V2", "hybrid_step")
V2_ACTION_DIM = 12 if V2_ACTION_MODE == "hybrid_step" else 6
ACTION_SHIFT = int(os.environ.get("COSMOS_ACTION_SHIFT", "0"))
NUM_FRAMES = int(os.environ.get("COSMOS_NUM_FRAMES", "17"))
ACTION_VARIANT = os.environ.get("COSMOS_ACTION_VARIANT", "normal")
ACTION_LAYOUT = os.environ.get("COSMOS_ACTION_LAYOUT", "cosmos7")
HOLDOUT_COUNT = int(os.environ.get("COSMOS_HOLDOUT_COUNT", "6"))
SEED = int(os.environ.get("COSMOS_SEED", "0"))
PROBE_TAG = os.environ.get("COSMOS_PROBE_TAG", "step500_17f_normal")
PROBE_GUIDANCE = [
    int(value)
    for value in os.environ.get("COSMOS_PROBE_GUIDANCE", "0,3,7").split(",")
    if value.strip()
]


def get_sampler(dataset):
    return DistributedSampler(
        dataset,
        num_replicas=parallel_state.get_data_parallel_world_size(),
        rank=parallel_state.get_data_parallel_rank(),
        shuffle=True,
        seed=SEED,
    )


def _dataset(split: str, action_mode: str = ACTION_MODE):
    return L(CosmosSO100Dataset)(
        root=DATA_ROOT,
        split=split,
        height=HEIGHT,
        width=WIDTH,
        num_frames=NUM_FRAMES,
        holdout_count=HOLDOUT_COUNT,
        seed=SEED,
        action_mode=action_mode,
        action_shift=ACTION_SHIFT,
        action_variant=ACTION_VARIANT,
        action_layout=ACTION_LAYOUT,
    )


SO100_TRAIN_DATASET = _dataset("train")
SO100_HOLDOUT_DATASET = _dataset("holdout")
SO100_TRAIN_DATALOADER = L(DataLoader)(
    dataset=SO100_TRAIN_DATASET,
    sampler=L(get_sampler)(dataset=SO100_TRAIN_DATASET),
    batch_size=1,
    num_workers=4,
    pin_memory=True,
    persistent_workers=True,
    drop_last=True,
)
SO100_HOLDOUT_DATALOADER = L(DataLoader)(
    dataset=SO100_HOLDOUT_DATASET,
    sampler=L(get_sampler)(dataset=SO100_HOLDOUT_DATASET),
    batch_size=1,
    num_workers=2,
    pin_memory=True,
    persistent_workers=True,
    drop_last=True,
)

SO100_V2_TRAIN_DATASET = _dataset("train", action_mode=V2_ACTION_MODE)
SO100_V2_HOLDOUT_DATASET = _dataset("holdout", action_mode=V2_ACTION_MODE)
SO100_V2_TRAIN_DATALOADER = L(DataLoader)(
    dataset=SO100_V2_TRAIN_DATASET,
    sampler=L(get_sampler)(dataset=SO100_V2_TRAIN_DATASET),
    batch_size=1,
    num_workers=4,
    pin_memory=True,
    persistent_workers=True,
    drop_last=True,
)
SO100_V2_HOLDOUT_DATALOADER = L(DataLoader)(
    dataset=SO100_V2_HOLDOUT_DATASET,
    sampler=L(get_sampler)(dataset=SO100_V2_HOLDOUT_DATASET),
    batch_size=1,
    num_workers=2,
    pin_memory=True,
    persistent_workers=True,
    drop_last=True,
)

ACTION_CHECKPOINT = MODEL_CHECKPOINTS[
    ModelKey(variant=ModelVariant.ROBOT_ACTION_COND)
]
ACTION_CHECKPOINT_HF = (
    f"hf://{ACTION_CHECKPOINT.hf.repository}/{ACTION_CHECKPOINT.hf.filename}"
)
TRAIN_JOB_NAME = "cosmos_predict2p5_so100_17f"
TRAIN_OUTPUT = os.path.join(
    os.environ.get(
        "IMAGINAIRE_OUTPUT_ROOT",
        os.path.join(REPO, "open", "baseline", "outputs", "cosmos_action"),
    ),
    "inha_cosmos",
    "so100_action",
    TRAIN_JOB_NAME,
)
PROBE_CHECKPOINT = os.environ.get(
    "COSMOS_PROBE_CHECKPOINT",
    os.path.join(TRAIN_OUTPUT, "checkpoints", "iter_000000500"),
)
ADAPTER_PROBE_CHECKPOINT = os.environ.get(
    "COSMOS_ADAPTER_PROBE_CHECKPOINT",
    os.path.join(
        os.environ.get(
            "IMAGINAIRE_OUTPUT_ROOT",
            os.path.join(REPO, "open", "baseline", "outputs", "cosmos_action"),
        ),
        "inha_cosmos",
        "so100_joint_adapter",
        "cosmos_predict2p5_so100_joint_adapter_17f",
        "checkpoints",
        "iter_000000500",
    ),
)
ADAPTER_V2_PROBE_CHECKPOINT = os.environ.get(
    "COSMOS_ADAPTER_V2_PROBE_CHECKPOINT",
    os.path.join(
        os.environ.get(
            "IMAGINAIRE_OUTPUT_ROOT",
            os.path.join(REPO, "open", "baseline", "outputs", "cosmos_action"),
        ),
        "inha_cosmos",
        "so100_joint_adapter_v2",
        "cosmos_predict2p5_so100_joint_adapter_v2_17f",
        "checkpoints",
        "iter_000000250",
    ),
)

SO100_EXPERIMENT = LazyDict(
    dict(
        defaults=[
            "/experiment/cosmos_predict2p5_2B_reason_embeddings_action_conditioned_rectified_flow_bridge_13frame_480_640_",
            {"override /net": "cosmos_v1_2B_action_chunk_conditioned"},
            {"override /data_train": "so100_17frame_train"},
            {"override /data_val": "so100_17frame_holdout"},
            "_self_",
        ],
        job=dict(
            project="inha_cosmos",
            group="so100_action",
            name=TRAIN_JOB_NAME,
            wandb_mode="disabled",
        ),
        optimizer=dict(
            lr=3.0e-5,
            weight_decay=0.1,
        ),
        checkpoint=dict(
            save_iter=250,
            # The registry's S3 URI is NVIDIA-internal. Use the public HF
            # artifact explicitly so external training never waits on S3.
            load_path=ACTION_CHECKPOINT_HF,
            load_training_state=False,
            strict_resume=False,
            load_from_object_store=dict(enabled=False),
            save_to_object_store=dict(enabled=False),
        ),
        trainer=dict(
            max_iter=500,
            logging_iter=10,
            validation_iter=250,
            run_validation=False,
            straggler_detection=dict(enabled=False),
            callbacks=dict(
                every_n_sample_reg=dict(every_n=250, fps=6, save_s3=False),
                every_n_sample_ema=dict(every_n=250, fps=6, save_s3=False),
                heart_beat=dict(save_s3=False),
                iter_speed=dict(save_s3=False),
                device_monitor=dict(save_s3=False),
                wandb=dict(save_s3=False),
                wandb_10x=dict(save_s3=False),
                dataloader_speed=dict(save_s3=False),
            ),
        ),
        model_parallel=dict(context_parallel_size=1),
        model=dict(
            config=dict(
                min_num_conditional_frames=1,
                max_num_conditional_frames=1,
                conditional_frames_probs=None,
                state_t=1 + (NUM_FRAMES - 1) // 4,
                net=dict(
                    action_dim=7,
                    temporal_compression_ratio=4,
                ),
            ),
        ),
        dataloader_train=dict(batch_size=1),
        dataloader_val=dict(batch_size=1),
    ),
    flags={"allow_objects": True},
)

SO100_PROBE_EXPERIMENT = LazyDict(
    dict(
        defaults=[
            "/experiment/cosmos_predict2p5_2B_reason_embeddings_action_conditioned_rectified_flow_bridge_13frame_480_640_",
            {"override /net": "cosmos_v1_2B_action_chunk_conditioned"},
            {"override /data_train": "so100_17frame_probe"},
            {"override /data_val": "so100_17frame_holdout"},
            "_self_",
        ],
        job=dict(
            project="inha_cosmos",
            group="so100_action_probe",
            name=f"cosmos_predict2p5_so100_probe_{PROBE_TAG}",
            wandb_mode="disabled",
        ),
        optimizer=dict(lr=0.0, weight_decay=0.0),
        checkpoint=dict(
            type=L(LoadOnlyDistributedCheckpointer)(),
            save_iter=999_999_999,
            load_path=PROBE_CHECKPOINT,
            load_training_state=False,
            strict_resume=True,
            keys_not_to_resume=["optim", "scheduler", "trainer"],
            load_from_object_store=dict(enabled=False),
            save_to_object_store=dict(enabled=False),
        ),
        trainer=dict(
            max_iter=1,
            logging_iter=1,
            run_validation=False,
            straggler_detection=dict(enabled=False),
            callbacks=dict(
                every_n_sample_reg=dict(
                    every_n=1,
                    do_x0_prediction=False,
                    num_sampling_step=35,
                    guidance=PROBE_GUIDANCE,
                    fps=6,
                    save_s3=False,
                ),
                every_n_sample_ema=dict(every_n=999_999_999, save_s3=False),
                heart_beat=dict(save_s3=False),
                iter_speed=dict(save_s3=False),
                device_monitor=dict(save_s3=False),
                wandb=dict(save_s3=False),
                wandb_10x=dict(save_s3=False),
                dataloader_speed=dict(save_s3=False),
            ),
        ),
        model_parallel=dict(context_parallel_size=1),
        model=dict(
            config=dict(
                min_num_conditional_frames=1,
                max_num_conditional_frames=1,
                conditional_frames_probs=None,
                state_t=1 + (NUM_FRAMES - 1) // 4,
                net=dict(
                    action_dim=7,
                    temporal_compression_ratio=4,
                ),
            ),
        ),
        dataloader_train=dict(batch_size=1),
        dataloader_val=dict(batch_size=1),
    ),
    flags={"allow_objects": True},
)

SO100_ADAPTER_NET = copy.deepcopy(COSMOS_V1_2B_NET_MININET_ACTION_CHUNK)
SO100_ADAPTER_NET["_target_"] = SO100AdapterActionChunkDiT
SO100_ADAPTER_NET["joint_action_dim"] = 6
SO100_ADAPTER_NET["joint_adapter_hidden_dim"] = 64
SO100_ADAPTER_NET["joint_adapter_output_scale"] = 0.5

SO100_ADAPTER_V2_NET = copy.deepcopy(SO100_ADAPTER_NET)
SO100_ADAPTER_V2_NET["joint_adapter_bias"] = False
SO100_ADAPTER_V2_NET["joint_action_dim"] = V2_ACTION_DIM

SO100_ADAPTER_EXPERIMENT = LazyDict(
    dict(
        defaults=[
            "/experiment/cosmos_predict2p5_2B_reason_embeddings_action_conditioned_rectified_flow_bridge_13frame_480_640_",
            {"override /net": "cosmos_v1_2B_so100_joint_adapter"},
            {"override /data_train": "so100_17frame_train"},
            {"override /data_val": "so100_17frame_holdout"},
            "_self_",
        ],
        job=dict(
            project="inha_cosmos",
            group="so100_joint_adapter",
            name="cosmos_predict2p5_so100_joint_adapter_17f",
            wandb_mode="disabled",
        ),
        optimizer=dict(
            _target_=get_joint_adapter_optimizer,
            lr=1.0e-3,
            weight_decay=0.0,
        ),
        scheduler=dict(
            warm_up_steps=[10],
            cycle_lengths=[500],
        ),
        checkpoint=dict(
            save_iter=250,
            load_path=ACTION_CHECKPOINT_HF,
            load_training_state=False,
            strict_resume=False,
            load_from_object_store=dict(enabled=False),
            save_to_object_store=dict(enabled=False),
        ),
        trainer=dict(
            max_iter=500,
            logging_iter=10,
            validation_iter=250,
            run_validation=False,
            straggler_detection=dict(enabled=False),
            callbacks=dict(
                every_n_sample_reg=dict(
                    every_n=250,
                    do_x0_prediction=False,
                    num_sampling_step=35,
                    guidance=[0, 7],
                    fps=6,
                    save_s3=False,
                ),
                every_n_sample_ema=dict(every_n=999_999_999, save_s3=False),
                heart_beat=dict(save_s3=False),
                iter_speed=dict(save_s3=False),
                device_monitor=dict(save_s3=False),
                wandb=dict(save_s3=False),
                wandb_10x=dict(save_s3=False),
                dataloader_speed=dict(save_s3=False),
            ),
        ),
        model_parallel=dict(context_parallel_size=1),
        model=dict(
            config=dict(
                min_num_conditional_frames=1,
                max_num_conditional_frames=1,
                conditional_frames_probs=None,
                state_t=1 + 16 // 4,
                ema=dict(enabled=False),
                net=dict(
                    action_dim=7,
                    temporal_compression_ratio=4,
                    joint_action_dim=6,
                    joint_adapter_hidden_dim=64,
                    joint_adapter_output_scale=0.5,
                ),
            ),
        ),
        dataloader_train=dict(batch_size=1),
        dataloader_val=dict(batch_size=1),
    ),
    flags={"allow_objects": True},
)

# V2 starts from the public Cosmos checkpoint, not the failed biased V1
# adapter. It is deliberately a separate job so no DCP state can leak across.
SO100_ADAPTER_V2_EXPERIMENT = copy.deepcopy(SO100_ADAPTER_EXPERIMENT)
SO100_ADAPTER_V2_EXPERIMENT["defaults"][1] = {
    "override /net": "cosmos_v1_2B_so100_joint_adapter_v2"
}
SO100_ADAPTER_V2_EXPERIMENT["defaults"][2] = {
    "override /data_train": "so100_17frame_v2_train"
}
SO100_ADAPTER_V2_EXPERIMENT["defaults"][3] = {
    "override /data_val": "so100_17frame_v2_holdout"
}
SO100_ADAPTER_V2_EXPERIMENT["job"]["group"] = "so100_joint_adapter_v2"
SO100_ADAPTER_V2_EXPERIMENT["job"]["name"] = (
    "cosmos_predict2p5_so100_joint_adapter_v2_17f"
)
SO100_ADAPTER_V2_EXPERIMENT["trainer"]["max_iter"] = 250
SO100_ADAPTER_V2_EXPERIMENT["trainer"]["validation_iter"] = 125
SO100_ADAPTER_V2_EXPERIMENT["trainer"]["callbacks"]["every_n_sample_reg"][
    "every_n"
] = 125
SO100_ADAPTER_V2_EXPERIMENT["checkpoint"]["save_iter"] = 125
SO100_ADAPTER_V2_EXPERIMENT["scheduler"]["cycle_lengths"] = [250]
SO100_ADAPTER_V2_EXPERIMENT["model"]["_target_"] = SO100ActionRankingModel
SO100_ADAPTER_V2_EXPERIMENT["model"]["action_rank_margin"] = 0.01
SO100_ADAPTER_V2_EXPERIMENT["model"]["action_rank_weight"] = 1.0
SO100_ADAPTER_V2_EXPERIMENT["model"]["wrong_action_mode"] = "dp_roll"
SO100_ADAPTER_V2_EXPERIMENT["model"]["action_log_every"] = 10
SO100_ADAPTER_V2_EXPERIMENT["model"]["config"]["net"][
    "joint_adapter_bias"
] = False
SO100_ADAPTER_V2_EXPERIMENT["model"]["config"]["net"][
    "joint_action_dim"
] = V2_ACTION_DIM

SO100_ADAPTER_VIDEO_EXPERIMENT = LazyDict(
    dict(
        defaults=[
            "/experiment/cosmos_predict2p5_2B_reason_embeddings_action_conditioned_rectified_flow_bridge_13frame_480_640_",
            {"override /net": "cosmos_v1_2B_so100_joint_adapter"},
            {"override /data_train": "so100_17frame_probe"},
            {"override /data_val": "so100_17frame_holdout"},
            "_self_",
        ],
        job=dict(
            project="inha_cosmos",
            group="so100_joint_adapter_video",
            name=f"cosmos_predict2p5_so100_adapter_video_{PROBE_TAG}",
            wandb_mode="disabled",
        ),
        optimizer=dict(
            _target_=get_joint_adapter_optimizer,
            lr=0.0,
            weight_decay=0.0,
        ),
        checkpoint=dict(
            type=L(LoadOnlyDistributedCheckpointer)(),
            save_iter=999_999_999,
            load_path=ADAPTER_PROBE_CHECKPOINT,
            load_training_state=False,
            strict_resume=True,
            keys_not_to_resume=["optim", "scheduler", "trainer"],
            load_from_object_store=dict(enabled=False),
            save_to_object_store=dict(enabled=False),
        ),
        trainer=dict(
            max_iter=1,
            logging_iter=1,
            run_validation=False,
            straggler_detection=dict(enabled=False),
            callbacks=dict(
                every_n_sample_reg=dict(
                    _target_=LocalVideoEveryNDrawSample,
                    every_n=1,
                    do_x0_prediction=False,
                    num_sampling_step=35,
                    guidance=PROBE_GUIDANCE,
                    n_viz_sample=1,
                    fps=6,
                    save_s3=False,
                ),
                every_n_sample_ema=dict(every_n=999_999_999, save_s3=False),
                heart_beat=dict(save_s3=False),
                iter_speed=dict(save_s3=False),
                device_monitor=dict(save_s3=False),
                wandb=dict(save_s3=False),
                wandb_10x=dict(save_s3=False),
                dataloader_speed=dict(save_s3=False),
            ),
        ),
        model_parallel=dict(context_parallel_size=1),
        model=dict(
            config=dict(
                min_num_conditional_frames=1,
                max_num_conditional_frames=1,
                conditional_frames_probs=None,
                state_t=1 + 16 // 4,
                ema=dict(enabled=False),
                net=dict(
                    action_dim=7,
                    temporal_compression_ratio=4,
                    joint_action_dim=6,
                    joint_adapter_hidden_dim=64,
                    joint_adapter_output_scale=0.5,
                ),
            ),
        ),
        dataloader_train=dict(batch_size=1),
        dataloader_val=dict(batch_size=1),
    ),
    flags={"allow_objects": True},
)

SO100_ADAPTER_V2_VIDEO_EXPERIMENT = copy.deepcopy(SO100_ADAPTER_VIDEO_EXPERIMENT)
SO100_ADAPTER_V2_VIDEO_EXPERIMENT["defaults"][1] = {
    "override /net": "cosmos_v1_2B_so100_joint_adapter_v2"
}
SO100_ADAPTER_V2_VIDEO_EXPERIMENT["defaults"][2] = {
    "override /data_train": "so100_17frame_v2_probe"
}
SO100_ADAPTER_V2_VIDEO_EXPERIMENT["defaults"][3] = {
    "override /data_val": "so100_17frame_v2_holdout"
}
SO100_ADAPTER_V2_VIDEO_EXPERIMENT["job"]["group"] = "so100_joint_adapter_v2_video"
SO100_ADAPTER_V2_VIDEO_EXPERIMENT["job"]["name"] = (
    f"cosmos_predict2p5_so100_adapter_v2_video_{PROBE_TAG}"
)
SO100_ADAPTER_V2_VIDEO_EXPERIMENT["checkpoint"]["load_path"] = (
    ADAPTER_V2_PROBE_CHECKPOINT
)
SO100_ADAPTER_V2_VIDEO_EXPERIMENT["model"]["config"]["net"][
    "joint_adapter_bias"
] = False
SO100_ADAPTER_V2_VIDEO_EXPERIMENT["model"]["config"]["net"][
    "joint_action_dim"
] = V2_ACTION_DIM


def _register_so100() -> None:
    cs = ConfigStore.instance()
    cs.store(
        group="net",
        package="model.config.net",
        name="cosmos_v1_2B_so100_joint_adapter",
        node=SO100_ADAPTER_NET,
    )
    cs.store(
        group="net",
        package="model.config.net",
        name="cosmos_v1_2B_so100_joint_adapter_v2",
        node=SO100_ADAPTER_V2_NET,
    )
    cs.store(
        group="data_train",
        package="dataloader_train",
        name="so100_17frame_train",
        node=SO100_TRAIN_DATALOADER,
    )
    cs.store(
        group="data_val",
        package="dataloader_val",
        name="so100_17frame_holdout",
        node=SO100_HOLDOUT_DATALOADER,
    )
    cs.store(
        group="data_train",
        package="dataloader_train",
        name="so100_17frame_v2_train",
        node=SO100_V2_TRAIN_DATALOADER,
    )
    cs.store(
        group="data_val",
        package="dataloader_val",
        name="so100_17frame_v2_holdout",
        node=SO100_V2_HOLDOUT_DATALOADER,
    )
    cs.store(
        group="data_train",
        package="dataloader_train",
        name="so100_17frame_v2_probe",
        node=SO100_V2_HOLDOUT_DATALOADER,
    )
    cs.store(
        group="data_train",
        package="dataloader_train",
        name="so100_17frame_probe",
        node=SO100_HOLDOUT_DATALOADER,
    )
    cs.store(
        group="experiment",
        package="_global_",
        name="inha_cosmos_so100_500",
        node=SO100_EXPERIMENT,
    )
    cs.store(
        group="experiment",
        package="_global_",
        name="inha_cosmos_so100_probe",
        node=SO100_PROBE_EXPERIMENT,
    )
    cs.store(
        group="experiment",
        package="_global_",
        name="inha_cosmos_so100_joint_adapter_500",
        node=SO100_ADAPTER_EXPERIMENT,
    )
    cs.store(
        group="experiment",
        package="_global_",
        name="inha_cosmos_so100_joint_adapter_v2_250",
        node=SO100_ADAPTER_V2_EXPERIMENT,
    )
    cs.store(
        group="experiment",
        package="_global_",
        name="inha_cosmos_so100_adapter_video",
        node=SO100_ADAPTER_VIDEO_EXPERIMENT,
    )
    cs.store(
        group="experiment",
        package="_global_",
        name="inha_cosmos_so100_adapter_v2_video",
        node=SO100_ADAPTER_V2_VIDEO_EXPERIMENT,
    )


def make_config():
    config = make_cosmos_config()
    _register_so100()
    return config
