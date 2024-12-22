import random
import warnings
import numpy as np
import torch
import torch.distributed as dist
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import (
    HOOKS,
    DistSamplerSeedHook,
    EpochBasedRunner,
    Fp16OptimizerHook,
    OptimizerHook,
    build_optimizer,
    build_runner,
    get_dist_info,
)
from mmcv.utils import build_from_cfg
from mmdet.core import EvalHook
from mmdet.datasets import build_dataset, replace_ImageToTensor
from mmdet.utils import get_root_logger
import time
import os.path as osp
from projects.mmdet3d_plugin.datasets.builder import build_dataloader
from projects.mmdet3d_plugin.core.evaluation.eval_hooks import (
    CustomDistEvalHook,
)
from projects.mmdet3d_plugin.datasets import custom_build_dataset
import torch.cuda.nvtx as nvtx  # Import NVTX for profiling

def custom_train_detector(
    model,
    dataset,
    cfg,
    distributed=False,
    validate=False,
    timestamp=None,
    meta=None,
):
    logger = get_root_logger(cfg.log_level)

    # Prepare data loaders
    nvtx.range_push("Data Loader Preparation")
    dataset = dataset if isinstance(dataset, (list, tuple)) else [dataset]
    if "imgs_per_gpu" in cfg.data:
        logger.warning(
            '"imgs_per_gpu" is deprecated in MMDet V2.0. '
            'Please use "samples_per_gpu" instead'
        )
        if "samples_per_gpu" in cfg.data:
            logger.warning(
                f'Got "imgs_per_gpu"={cfg.data.imgs_per_gpu} and '
                f'"samples_per_gpu"={cfg.data.samples_per_gpu}, "imgs_per_gpu"'
                f"={cfg.data.imgs_per_gpu} is used in this experiments"
            )
        else:
            logger.warning(
                'Automatically set "samples_per_gpu"="imgs_per_gpu"='
                f"{cfg.data.imgs_per_gpu} in this experiments"
            )
        cfg.data.samples_per_gpu = cfg.data.imgs_per_gpu

    if "runner" in cfg:
        runner_type = cfg.runner["type"]
    else:
        runner_type = "EpochBasedRunner"

    data_loaders = [
        build_dataloader(
            ds,
            cfg.data.samples_per_gpu,
            cfg.data.workers_per_gpu,
            len(cfg.gpu_ids),
            dist=distributed,
            seed=cfg.seed,
            nonshuffler_sampler=dict(type="DistributedSampler"),
            runner_type=runner_type,
        )
        for ds in dataset
    ]
    nvtx.range_pop()

    # Put model on GPUs
    nvtx.range_push("Model Initialization")
    if distributed:
        find_unused_parameters = cfg.get("find_unused_parameters", False)
        model = MMDistributedDataParallel(
            model.cuda(),
            device_ids=[torch.cuda.current_device()],
            broadcast_buffers=False,
            find_unused_parameters=find_unused_parameters,
        )
    else:
        model = MMDataParallel(
            model.cuda(cfg.gpu_ids[0]), device_ids=cfg.gpu_ids
        )
    nvtx.range_pop()

    # Build runner
    nvtx.range_push("Runner Setup")
    optimizer = build_optimizer(model, cfg.optimizer)
    if "runner" not in cfg:
        cfg.runner = {
            "type": "EpochBasedRunner",
            "max_epochs": cfg.total_epochs,
        }
        warnings.warn(
            "config is now expected to have a `runner` section, "
            "please set `runner` in your config.",
            UserWarning,
        )
    else:
        if "total_epochs" in cfg:
            assert cfg.total_epochs == cfg.runner.max_epochs

    runner = build_runner(
        cfg.runner,
        default_args=dict(
            model=model,
            optimizer=optimizer,
            work_dir=cfg.work_dir,
            logger=logger,
            meta=meta,
        ),
    )
    runner.timestamp = timestamp
    nvtx.range_pop()

    # FP16 setting
    nvtx.range_push("FP16/Optimizer Configuration")
    fp16_cfg = cfg.get("fp16", None)
    if fp16_cfg is not None:
        optimizer_config = Fp16OptimizerHook(
            **cfg.optimizer_config, **fp16_cfg, distributed=distributed
        )
    elif distributed and "type" not in cfg.optimizer_config:
        optimizer_config = OptimizerHook(**cfg.optimizer_config)
    else:
        optimizer_config = cfg.optimizer_config
    nvtx.range_pop()

    # Register hooks
    nvtx.range_push("Hook Registration")
    runner.register_training_hooks(
        cfg.lr_config,
        optimizer_config,
        cfg.checkpoint_config,
        cfg.log_config,
        cfg.get("momentum_config", None),
    )
    if distributed and isinstance(runner, EpochBasedRunner):
        runner.register_hook(DistSamplerSeedHook())
    nvtx.range_pop()

    # Register evaluation hooks
    if validate:
        nvtx.range_push("Validation Setup")
        val_samples_per_gpu = cfg.data.val.pop("samples_per_gpu", 1)
        if val_samples_per_gpu > 1:
            cfg.data.val.pipeline = replace_ImageToTensor(
                cfg.data.val.pipeline
            )
        val_dataset = custom_build_dataset(cfg.data.val, dict(test_mode=True))
        val_dataloader = build_dataloader(
            val_dataset,
            samples_per_gpu=val_samples_per_gpu,
            workers_per_gpu=cfg.data.workers_per_gpu,
            dist=distributed,
            shuffle=False,
            nonshuffler_sampler=dict(type="DistributedSampler"),
        )
        eval_cfg = cfg.get("evaluation", {})
        eval_cfg["by_epoch"] = cfg.runner["type"] != "IterBasedRunner"
        eval_cfg["jsonfile_prefix"] = osp.join(
            "val",
            cfg.work_dir,
            time.ctime().replace(" ", "_").replace(":", "_"),
        )
        eval_hook = CustomDistEvalHook if distributed else EvalHook
        runner.register_hook(eval_hook(val_dataloader, **eval_cfg))
        nvtx.range_pop()

    # Run the training process
    nvtx.range_push("Training Process")
    if cfg.resume_from:
        runner.resume(cfg.resume_from)
    elif cfg.load_from:
        runner.load_checkpoint(cfg.load_from)
    runner.run(data_loaders, cfg.workflow)
    nvtx.range_pop()
