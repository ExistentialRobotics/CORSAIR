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

import MinkowskiEngine as ME


def get_norm(norm_type, num_feats, bn_momentum=0.05, D=-1):
    if norm_type == "BN":
        return ME.MinkowskiBatchNorm(num_feats, momentum=bn_momentum)
    elif norm_type == "IN":
        return ME.MinkowskiInstanceNorm(num_feats, dimension=D)
    else:
        raise ValueError(f"Type {norm_type}, not defined")
