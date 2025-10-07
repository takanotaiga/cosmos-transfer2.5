# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Callable, Optional

import attrs
import cv2
import numpy as np
import torch

from cosmos_transfer2._src.transfer2.datasets.augmentors.fast_blur import BilateralGaussian


@attrs.define
class BilateralFilterConfig:
    """Configuration for Bilateral filter"""

    use_random: bool = False
    # if use_random is False, then optionally define the param values
    d: int = 30
    sigma_color: int = 150
    sigma_space: int = 100
    iter: int = 1

    # if use_random is True, then optionally define the range
    d_min: int = 15
    d_max: int = 50
    sigma_color_min: int = 100
    sigma_color_max: int = 300
    sigma_space_min: int = 50
    sigma_space_max: int = 150
    iter_min: int = 1
    iter_max: int = 2

    # Whether to use GPU kernel (inference only)
    use_cuda: bool = False


def apply_bilateral_filter(
    frames: np.ndarray,
    d: int = 9,
    sigma_color: float = 75,
    sigma_space: float = 75,
    iter: int = 1,
    bilateral_cuda_module: Optional[Callable] = None,
) -> np.ndarray:
    blurred_image = []
    if bilateral_cuda_module is not None:
        frames = torch.from_numpy(frames).cuda()
        for _image in frames.permute(1, 2, 3, 0):
            # The hyper-parameters to NPP version of Bilateral Gaussian blur need to be adjusted.
            # The chosen values are chosen based on experiments, but it does not guarantee bitwise accuracy.
            blurred_image.append(bilateral_cuda_module(_image, d // 3, (sigma_color // 2) ** 2, sigma_space**2))
        blurred_image = torch.stack(blurred_image).permute(3, 0, 1, 2)
        blurred_image = blurred_image.cpu().numpy()
        return blurred_image

    for _image_np in frames.transpose((1, 2, 3, 0)):
        for _ in range(iter):
            _image_np = cv2.bilateralFilter(_image_np, d, sigma_color, sigma_space)
        blurred_image += [_image_np]

    blurred_image = np.stack(blurred_image).transpose((3, 0, 1, 2))
    return blurred_image


class BilateralFilter:
    def __init__(self, config: BilateralFilterConfig) -> None:
        self.use_random = config.use_random
        self.config = config
        assert not (self.use_random and self.config.use_cuda), "Cannot use GPU kernel for training."
        self.bilateral_cuda_module = BilateralGaussian() if self.config.use_cuda else None

    def __call__(self, frames: np.ndarray) -> np.ndarray:
        config = self.config
        if self.use_random:
            d = np.random.randint(config.d_min, config.d_max)
            sigma_color = np.random.randint(config.sigma_color_min, config.sigma_color_max)
            sigma_space = np.random.randint(config.sigma_space_min, config.sigma_space_max)
            iter = np.random.randint(config.iter_min, config.iter_max)
        else:
            d = config.d
            sigma_color = config.sigma_color
            sigma_space = config.sigma_space
            iter = config.iter
        return apply_bilateral_filter(frames, d, sigma_color, sigma_space, iter, self.bilateral_cuda_module)
