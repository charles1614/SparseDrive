from typing import List, Optional, Tuple, Union
import warnings

import numpy as np
import torch
import torch.nn as nn
import torch.cuda.nvtx as nvtx  # Import NVTX for profiling

from mmcv.runner import BaseModule
from mmdet.models import HEADS
from mmdet.models import build_head


@HEADS.register_module()
class SparseDriveHead(BaseModule):
    def __init__(
        self,
        task_config: dict,
        det_head = dict,
        map_head = dict,
        motion_plan_head = dict,
        init_cfg=None,
        **kwargs,
    ):
        super(SparseDriveHead, self).__init__(init_cfg)
        self.task_config = task_config
        if self.task_config['with_det']:
            self.det_head = build_head(det_head)
        if self.task_config['with_map']:
            self.map_head = build_head(map_head)
        if self.task_config['with_motion_plan']:
            self.motion_plan_head = build_head(motion_plan_head)

    def init_weights(self):
        if self.task_config['with_det']:
            self.det_head.init_weights()
        if self.task_config['with_map']:
            self.map_head.init_weights()
        if self.task_config['with_motion_plan']:
            self.motion_plan_head.init_weights()

    def forward(
        self,
        feature_maps: Union[torch.Tensor, List],
        metas: dict,
    ):
        nvtx.range_push("SparseDriveHead.forward")
        
        if self.task_config['with_det']:
            nvtx.range_push("Detection Head")
            det_output = self.det_head(feature_maps, metas)
            nvtx.range_pop()
        else:
            det_output = None

        if self.task_config['with_map']:
            nvtx.range_push("Map Head")
            map_output = self.map_head(feature_maps, metas)
            nvtx.range_pop()
        else:
            map_output = None
        
        if self.task_config['with_motion_plan']:
            nvtx.range_push("Motion Planning Head")
            motion_output, planning_output = self.motion_plan_head(
                det_output, 
                map_output, 
                feature_maps,
                metas,
                self.det_head.anchor_encoder,
                self.det_head.instance_bank.mask,
                self.det_head.anchor_handler,
            )
            nvtx.range_pop()
        else:
            motion_output, planning_output = None, None

        nvtx.range_pop()
        return det_output, map_output, motion_output, planning_output

    def loss(self, model_outs, data):
        nvtx.range_push("SparseDriveHead.loss")
        det_output, map_output, motion_output, planning_output = model_outs
        losses = dict()
        if self.task_config['with_det']:
            nvtx.range_push("Detection Loss")
            loss_det = self.det_head.loss(det_output, data)
            losses.update(loss_det)
            nvtx.range_pop()
        
        if self.task_config['with_map']:
            nvtx.range_push("Map Loss")
            loss_map = self.map_head.loss(map_output, data)
            losses.update(loss_map)
            nvtx.range_pop()

        if self.task_config['with_motion_plan']:
            nvtx.range_push("Motion Planning Loss")
            motion_loss_cache = dict(
                indices=self.det_head.sampler.indices, 
            )
            loss_motion = self.motion_plan_head.loss(
                motion_output, 
                planning_output, 
                data, 
                motion_loss_cache
            )
            losses.update(loss_motion)
            nvtx.range_pop()
        
        nvtx.range_pop()
        return losses

    def post_process(self, model_outs, data):
        nvtx.range_push("SparseDriveHead.post_process")
        det_output, map_output, motion_output, planning_output = model_outs
        if self.task_config['with_det']:
            nvtx.range_push("Detection Post-process")
            det_result = self.det_head.post_process(det_output)
            batch_size = len(det_result)
            nvtx.range_pop()
        
        if self.task_config['with_map']:
            nvtx.range_push("Map Post-process")
            map_result= self.map_head.post_process(map_output)
            batch_size = len(map_result)
            nvtx.range_pop()

        if self.task_config['with_motion_plan']:
            nvtx.range_push("Motion Planning Post-process")
            motion_result, planning_result = self.motion_plan_head.post_process(
                det_output,
                motion_output, 
                planning_output,
                data,
            )
            nvtx.range_pop()

        nvtx.range_push("Results Assembly")
        results = [dict()] * batch_size
        for i in range(batch_size):
            if self.task_config['with_det']:
                results[i].update(det_result[i])
            if self.task_config['with_map']:
                results[i].update(map_result[i])
            if self.task_config['with_motion_plan']:
                results[i].update(motion_result[i])
                results[i].update(planning_result[i])
        nvtx.range_pop()

        nvtx.range_pop()
        return results
