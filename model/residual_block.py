'''
   Copyright 2024 Qiaojun Feng, Sai Jadhav, Tianyu Zhao, Zhirui Dai, K. M. Brian Lee, Nikolay Atanasov, UC San Diego. 

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
'''

import torch.nn as nn

from model.common import get_norm

import MinkowskiEngine as ME
import MinkowskiEngine.MinkowskiFunctional as MEF


class BasicBlockBase(nn.Module):
    expansion = 1
    NORM_TYPE = "BN"

    def __init__(
        self,
        inplanes,
        planes,
        stride=1,
        dilation=1,
        downsample=None,
        bn_momentum=0.1,
        D=3,
    ):
        super(BasicBlockBase, self).__init__()

        self.conv1 = ME.MinkowskiConvolution(
            inplanes, planes, kernel_size=3, stride=stride, dimension=D
        )
        self.norm1 = get_norm(self.NORM_TYPE, planes, bn_momentum=bn_momentum, D=D)
        self.conv2 = ME.MinkowskiConvolution(
            planes,
            planes,
            kernel_size=3,
            stride=1,
            dilation=dilation,
            **{"bias" if ME.__version__ >= "0.5.4" else "has_bias": False},
            dimension=D,
        )
        self.norm2 = get_norm(self.NORM_TYPE, planes, bn_momentum=bn_momentum, D=D)
        self.downsample = downsample

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.norm1(out)
        out = MEF.relu(out)

        out = self.conv2(out)
        out = self.norm2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = MEF.relu(out)

        return out


class BasicBlockBN(BasicBlockBase):
    NORM_TYPE = "BN"


class BasicBlockIN(BasicBlockBase):
    NORM_TYPE = "IN"


def get_block(
    norm_type,
    inplanes,
    planes,
    stride=1,
    dilation=1,
    downsample=None,
    bn_momentum=0.1,
    D=3,
):
    if norm_type == "BN":
        return BasicBlockBN(
            inplanes, planes, stride, dilation, downsample, bn_momentum, D
        )
    elif norm_type == "IN":
        return BasicBlockIN(
            inplanes, planes, stride, dilation, downsample, bn_momentum, D
        )
    else:
        raise ValueError(f"Type {norm_type}, not defined")
