# Copyright 2024 Qiaojun Feng, Sai Jadhav, Tianyu Zhao, Zhirui Dai, K. M. Brian Lee, Nikolay Atanasov, UC San Diego. 

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

FROM nvidia/cuda:12.1.0-cudnn8-devel-ubuntu22.04

# https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/1.15.0/docker-specialized.html?highlight=nvidia_driver_capabilities
# With these variables you don't have to run the final image with the --gpus argument.
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=all

RUN DEBIAN_FRONTEND=noninteractive apt-get update && apt upgrade -y && \
    apt-get install -y \
    sudo \
    xauth \
    git \
    python3 \
    python3-pip \
    python3-dev \
    libopenblas-dev \
    build-essential \
    cmake \
    vim \
    xorg-dev \
    libxcb-shm0 \
    libglu1-mesa-dev \
    libc++-dev \
    libc++abi-dev \
    libsdl2-dev \
    ninja-build \
    libxi-dev \
    libtbb-dev \
    autoconf \
    libtool \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# ME installation requires this.
RUN ln -s /usr/bin/python3 /usr/bin/python # && ln -s /usr/bin/pip3 /usr/bin/pip

# Keeps Python from generating .pyc files in the container
ENV PYTHONDONTWRITEBYTECODE=1
# Turns off buffering for easier container logging
ENV PYTHONUNBUFFERED=1

RUN useradd -m -s /bin/bash -G sudo user && echo "user:user" | chpasswd && adduser user sudo
RUN echo "user ALL=(ALL) NOPASSWD: ALL" >> /etc/sudoers
RUN echo "export PATH=$PATH:/home/user/.local/bin" >> /home/user/.bashrc
RUN chown -R user:user /home/user

USER user
WORKDIR /home/user

RUN git clone --recursive https://github.com/pybind/pybind11 && cd pybind11 && \
    git checkout 50acb81b0a0f769dedd265727ade44e03778a977 && mkdir build && cd build && \
    cmake .. && make -j$(nproc) && sudo make install && \
    cd ../.. && rm -rf pybind11

RUN git clone --recursive https://github.com/ExistentialRobotics/CORSAIR

WORKDIR /home/user/CORSAIR
ENV PATH="/home/user/.local/bin:${PATH}"
RUN pip3 install --upgrade pip
RUN pip3 install -r requirements.txt --user --verbose
RUN cd deps/MinkowskiEngine && rm requirements.txt && export TORCH_CUDA_ARCH_LIST="6.1 7.0 7.5 8.0 8.6 8.9" && \
    python3 setup.py install --user --verbose --blas=openblas --force_cuda --cuda_home=/usr/local/cuda

RUN pip3 uninstall -y open3d && cd /home/user && git clone --recursive https://github.com/isl-org/Open3D && \
    cd Open3D && git checkout 160209d055214a199719559d7e65fcecd14960ce && \
    mkdir build && cd build && cmake .. -DUSE_SYSTEM_PYBIND11=ON && make install-pip-package -j$(nproc) && \
    cd /home/user && rm -rf Open3D

COPY --chown=user:user data/ShapeNetCore.v2.PC15k /home/user/CORSAIR/data/ShapeNetCore.v2.PC15k
COPY --chown=user:user entrypoint_chair.bash /home/user/CORSAIR/entrypoint_chair.bash
RUN chmod +x /home/user/CORSAIR/entrypoint_chair.bash
COPY --chown=user:user entrypoint_table.bash /home/user/CORSAIR/entrypoint_table.bash
RUN chmod +x /home/user/CORSAIR/entrypoint_table.bash
ENTRYPOINT ["/home/user/CORSAIR/entrypoint_chair.bash"]
