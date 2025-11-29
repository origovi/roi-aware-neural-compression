# Copyright (c) 2021-2024, InterDigital Communications, Inc
# All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted (subject to the limitations in the disclaimer
# below) provided that the following conditions are met:

# * Redistributions of source code must retain the above copyright notice,
#   this list of conditions and the following disclaimer.
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
# * Neither the name of InterDigital Communications, Inc nor the names of its
#   contributors may be used to endorse or promote products derived from this
#   software without specific prior written permission.

# NO EXPRESS OR IMPLIED LICENSES TO ANY PARTY'S PATENT RIGHTS ARE GRANTED BY
# THIS LICENSE. THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND
# CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT
# NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
# PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
# OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
# WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR
# OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
# ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import math

import torch
import torch.nn as nn
from utils.interface import *

from pytorch_msssim import ms_ssim

class RateDistortionLossWithBoundingBoxes(nn.Module):
    """Custom rate distortion loss with a Lagrangian parameter."""

    def __init__(self, alpha, lmbda=0.01, metric="mse", return_type="all"):
        super().__init__()
        if metric == "mse":
            self.metric = nn.MSELoss()
        elif metric == "ms-ssim":
            self.metric = ms_ssim
        else:
            raise NotImplementedError(f"{metric} is not implemented!")
        self.lmbda = lmbda
        self.alpha = alpha  # alpha controls how is the bounding box distortion taken into account in the computations of the loss
        self.return_type = return_type
    
    @staticmethod
    def _mask_tensor(tensor: torch.Tensor, objects: list[ROISet]):
        """
        Masks the input tensor with the correspondent bounding boxes.
        Returns:
            - masked tensor without bounding boxes with shape (B,C,H,W)
            - masked tensor with only bounding boxes with shape (B,C,H,W)
            - proportion of elements that lie inside a bounding box
        """
        B, C, H, W = tensor.shape
        mask_bb = torch.zeros((B, H, W), device=tensor.device, dtype=tensor.dtype)
        for b in range(B):
            for obj in objects[b].objects:
                bb = obj.bounding_box_2d.scale_to_shape((W, H))
                mask_bb[b, int(bb.ymin):int(bb.ymax), int(bb.xmin):int(bb.xmax)] = 1
        mask_bb = mask_bb.unsqueeze(1)  # Convert to shape (B, 1, H, W)
        mask_no_bb = 1 - mask_bb
        tensor_no_bb = tensor * mask_no_bb
        tensor_bb_only = tensor * mask_bb
        bb_proportion = mask_bb.sum().item() / mask_bb.numel()
        return tensor_no_bb, tensor_bb_only, bb_proportion
    
    def _compute_distortion(self, x_hat, x):
        metric_loss = None
        if self.metric == ms_ssim:
            ms_ssim_loss = self.metric(x_hat, x, data_range=1)
            distortion = 1 - ms_ssim_loss
            metric_loss = ms_ssim_loss
        else:
            mse_loss = self.metric(x_hat, x)
            distortion = mse_loss
            metric_loss = mse_loss
        return distortion, metric_loss

    def forward(self, output, target, gt_objects: list[ROISet]):
        N, _, H, W = target.size()
        out = {}
        num_pixels = N * H * W

        output_no_bb, output_bb_only, output_bb_prop = self._mask_tensor(output['x_hat'], gt_objects)
        target_no_bb, target_bb_only, target_bb_prop = self._mask_tensor(target, gt_objects)

        out["bpp_loss"] = sum(
            (torch.log(likelihoods).sum() / (-math.log(2) * num_pixels))
            for likelihoods in output["likelihoods"].values()
        )
        for name, likelihoods in output["likelihoods"].items():
            out[f"bpp_{name}"] = torch.log(likelihoods).sum() / (-math.log(2) * num_pixels)

        no_bb_distortion, no_bb_metric_loss = self._compute_distortion(output_no_bb, target_no_bb)
        only_bb_distortion, only_bb_metric_loss = self._compute_distortion(output_bb_only, target_bb_only)

        if self.metric == ms_ssim:
            out[f'no_bb_msssim_loss'] = no_bb_metric_loss
            out[f'only_bb_msssim_loss'] = only_bb_metric_loss
        else:
            out[f'no_bb_mse_loss'] = no_bb_metric_loss
            out[f'only_bb_mse_loss'] = only_bb_metric_loss

        out["rd_loss"] = out["bpp_loss"] + self.lmbda * (self.alpha * target_bb_prop * only_bb_distortion + (1-self.alpha) * (1-target_bb_prop) * no_bb_distortion)
        if self.return_type == "all":
            return out
        else:
            return out[self.return_type]
