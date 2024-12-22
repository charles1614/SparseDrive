# ---------------------------------------------
# Copyright (c) OpenMMLab. All rights reserved.
# ---------------------------------------------
#  Modified by Zhiqi Li
# ---------------------------------------------

from .mmdet_train import custom_train_detector
# from mmseg.apis import train_segmentor
from mmdet.apis import train_detector

import torch.cuda.nvtx as nvtx  # Import NVTX for profiling

nvtx.range_push("Custom Train Model")  # Start NVTX marker
def custom_train_model(
    model,
    dataset,
    cfg,
    distributed=False,
    validate=False,
    timestamp=None,
    meta=None,
):
    """A function wrapper for launching model training according to cfg.

    Because we need different eval_hook in runner. Should be deprecated in the
    future.
    """
    if cfg.model.type in ["EncoderDecoder3D"]:
        assert False
    else:
        custom_train_detector(
            model,
            dataset,
            cfg,
            distributed=distributed,
            validate=validate,
            timestamp=timestamp,
            meta=meta,
        )
nvtx.range_pop()  # End NVTX marker


def train_model(
    model,
    dataset,
    cfg,
    distributed=False,
    validate=False,
    timestamp=None,
    meta=None,
):
    """A function wrapper for launching model training according to cfg.

    Because we need different eval_hook in runner. Should be deprecated in the
    future.
    """
    train_detector(
        model,
        dataset,
        cfg,
        distributed=distributed,
        validate=validate,
        timestamp=timestamp,
        meta=meta,
    )
