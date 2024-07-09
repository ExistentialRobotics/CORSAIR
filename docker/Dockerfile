FROM nvidia/cuda:12.1.0-cudnn8-devel-ubuntu22.04

# https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/user-guide.html#dockerfiles
# With these variables you don't have to run the final image with the --gpus argument.
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

RUN DEBIAN_FRONTEND=noninteractive apt-get update && apt upgrade -y && \
    apt-get install -y \
    sudo \
    git \
    python3 \
    python3-pip \
    python3-dev \
    libopenblas-dev \
    build-essential \
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

RUN git clone --recursive https://github.com/ExistentialRobotics/CORSAIR

WORKDIR /home/user/CORSAIR
ENV PATH="/home/user/.local/bin:${PATH}"
RUN pip3 install --upgrade pip
RUN pip3 install -r requirements.txt --user --verbose
RUN cd deps/MinkowskiEngine && rm requirements.txt && export TORCH_CUDA_ARCH_LIST="7.0 7.5 8.0 8.6 8.9" && \
    python3 setup.py install --user --verbose --blas=openblas --force_cuda --cuda_home=/usr/local/cuda

COPY --chown=user:user data/ShapeNetCore.v2.PC15k /home/user/CORSAIR/data/ShapeNetCore.v2.PC15k
COPY --chown=user:user entrypoint.bash /home/user/CORSAIR/entrypoint.bash
RUN chmod +x /home/user/CORSAIR/entrypoint.bash
ENTRYPOINT ["/home/user/CORSAIR/entrypoint.bash"]
