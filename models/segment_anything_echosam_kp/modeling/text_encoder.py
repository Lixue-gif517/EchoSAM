# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from tkinter import X
from unittest import skip
from unittest.mock import patch
import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Optional, Tuple, Type

from .common import LayerNorm2d, MLPBlock, Adapter, AugAdapter
import math
class FeatureEnhancer(nn.Module):
    def __init__(self,a = 1):
        super().__init__()
