# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Cosmos3-Edge full-SFT for SO-100 forward dynamics.

This follows the released DROID forward-dynamics recipe, changing only the
base tier (Edge), SO-100 dataset adapter, fixed-camera viewpoint, and the short
screening horizon.  It is deliberately not a LoRA or standalone adapter run.
"""
from __future__ import annotations

import copy

from hydra.core.config_store import ConfigStore

from cosmos_framework.configs.base.experiment.sft.models.edge_model_config import EDGE_MODEL_CONFIG
from cosmos_framework.data.generator.joint_dataloader import PackingDataLoader, RankPartitionedDataLoader
from cosmos_framework.data.generator.processors import build_processor_lazy
from cosmos_framework.utils.lazy_config import LazyCall as L
from cosmos_framework.utils.lazy_config import LazyDict
from cosmos3_edge_so100_dataset import get_action_so100_edge_sft_dataset


cs = ConfigStore.instance()


def _make_model_config() -> dict:
    cfg = copy.deepcopy(EDGE_MODEL_CONFIG)
    cfg["sound_gen"] = False
    cfg["resolution"] = "480"
    cfg["max_num_tokens_after_packing"] = 45056
    cfg["activation_checkpointing"]["mode"] = "full"
    cfg["tokenizer"]["encode_exact_durations"] = [17]
    # Resolve the processor from the already-downloaded local Edge snapshot.
    cfg["vlm_config"]["tokenizer"] = L(build_processor_lazy)(
        tokenizer_type="${oc.env:COSMOS3_EDGE_HF_PATH}",
    )
    return cfg


action_fd_so100_edge = LazyDict(
    dict(
        defaults=[
            {"override /data_train": None},
            {"override /data_val": None},
            {"override /model": "mot_fsdp"},
            {"override /optimizer": "fusedadamw"},
            {"override /scheduler": "lambdalinear"},
            {"override /tokenizer": "wan2pt2_tokenizer"},
            {"override /sound_tokenizer": None},
            {"override /vlm_config": None},
            {"override /checkpoint": "gcp"},
            {"override /callbacks": ["basic", "optimization", "job_monitor", "training_stats"]},
            {"override /ema": "power"},
            {"override /ckpt_type": "dcp"},
            "_self_",
        ],
        job=dict(
            project="cosmos3_action_fd",
            group="action_sft",
            name="cosmos3_edge_so100_native",
            wandb_mode="disabled",
        ),
        model=dict(config=_make_model_config()),
        optimizer=dict(
            betas=[0.9, 0.99],
            eps=1.0e-08,
            fused=True,
            # Official full-SFT generation pathway plus Edge's und-K norm and
            # all three released action bridge modules.
            keys_to_select=[
                "moe_gen",
                "time_embedder",
                "vae2llm",
                "llm2vae",
                "k_norm_und_for_gen",
                "action2llm",
                "llm2action",
                "action_modality_embed",
            ],
            lr=1.0e-04,
            lr_multipliers={
                "action2llm": 5.0,
                "llm2action": 5.0,
                "action_modality_embed": 5.0,
            },
            optimizer_type="FusedAdam",
            weight_decay=0.05,
        ),
        scheduler=dict(
            # Keep the official 20k horizon even for 250/500-step screens so
            # the run can resume without having decayed its LR to zero.
            cycle_lengths=[20000],
            f_max=[0.4],
            f_min=[0.0],
            f_start=[0.0],
            lr_scheduler_type="LambdaLinear",
            verbosity_interval=0,
            warm_up_steps=[0],
        ),
        trainer=dict(
            distributed_parallelism="fsdp",
            grad_accum_iter=1,
            logging_iter=10,
            max_iter=500,
            max_val_iter=None,
            run_validation=False,
            run_validation_on_start=False,
            save_zero_checkpoint=False,
            seed=42,
            timeout_period=999999999,
            validation_iter=100,
            compile_config=dict(recompile_limit=100, use_duck_shape=False),
            cudnn=dict(benchmark=True, deterministic=False),
            ddp=dict(broadcast_buffers=True, find_unused_parameters=False, static_graph=True),
            grad_scaler_args=dict(enabled=False),
            # The optional internal straggler package is not distributed with
            # the OSS framework.  Keep monitoring disabled so trainer startup
            # does not fail before model construction.
            straggler_detection=dict(enabled=False, report_freq=50),
            callbacks=dict(
                dataloader_speed=dict(every_n=50, save_s3=False, step_size=1),
                device_monitor=dict(every_n=100, log_memory_detail=True, save_s3=False, step_size=1),
                grad_clip=dict(clip_norm=1.0, force_finite=True),
                heart_beat=dict(every_n=100, save_s3=False, step_size=1, update_interval_in_minute=20),
                iter_speed=dict(every_n=10, hit_thres=10, save_s3=False, save_s3_every_log_n=500),
                low_precision=dict(update_iter=1),
                manual_gc=dict(every_n=100, gc_level=1, warm_up=1),
                norm_monitor=dict(every_n=100),
                param_count=dict(save_s3=False),
                sigma_loss_analysis=dict(every_n=250, every_n_viz=250, save_s3=False),
                skip_nan_step=dict(max_consecutive_nan=100),
                training_stats=dict(log_freq=50),
                compile_tokenizer=dict(enabled=True, warmup_resolutions=["480"]),
            ),
        ),
        checkpoint=dict(
            dcp_async_mode_enabled=False,
            enable_gcs_patch_in_boto3=True,
            keys_not_to_resume=[],
            keys_to_skip_loading=["net_ema."],
            load_ema_to_reg=False,
            load_from_object_store=dict(bucket="", credentials="", enabled=False),
            save_to_object_store=dict(bucket="", credentials="", enabled=False),
            load_path="???",
            load_training_state=False,
            only_load_scheduler_state=False,
            save_iter=250,
            strict_resume=True,
            verbose=True,
        ),
        dataloader_train=L(PackingDataLoader)(
            audio_sample_rate=48000,
            dataset_name="action_data",
            max_samples_per_batch=1,
            max_sequence_length=None,
            patch_spatial="${model.config.diffusion_expert_config.patch_spatial}",
            sound_latent_fps="${model.config.sound_latent_fps}",
            tokenizer_spatial_compression_factor="${model.config.tokenizer.spatial_compression_factor}",
            tokenizer_temporal_compression_factor="${model.config.tokenizer.temporal_compression_factor}",
            dataloader=L(RankPartitionedDataLoader)(
                batch_size=1,
                in_order=False,
                num_workers=2,
                persistent_workers=True,
                pin_memory=True,
                prefetch_factor=2,
                sampler=None,
                datasets=dict(
                    action_data=dict(
                        ratio=1,
                        dataset=L(get_action_so100_edge_sft_dataset)(
                            root="${oc.env:SO100_TRAIN_ROOT}",
                            index_path="${oc.env:SO100_INDEX_PATH}",
                            holdout_manifest="${oc.env:SO100_HOLDOUT_MANIFEST}",
                            fps=6.0,
                            chunk_length=16,
                            sample_stride=4,
                            embodiment_type="bridge_orig_lerobot",
                            mode="forward_dynamics",
                            viewpoint="third_person_view",
                            resolution="480",
                            max_action_dim="${model.config.max_action_dim}",
                            cfg_dropout_rate=0.1,
                            tokenizer_config="${model.config.vlm_config.tokenizer}",
                            iterable_shuffle=True,
                            episode_shuffle_seed=42,
                        ),
                    ),
                ),
            ),
        ),
        dataloader_val=None,
        upload_reproducible_setup=False,
    ),
    flags={"allow_objects": True},
)


cs.store(
    group="experiment",
    package="_global_",
    name="action_fd_so100_edge",
    node=action_fd_so100_edge,
)
