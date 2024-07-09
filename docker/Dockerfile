FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

RUN DEBIAN_FRONTEND=noninteractive apt-get update && apt upgrade -y && \
    apt-get install -y \
    sudo \
    git \
    python3 \
    python3-pip  && apt-get clean

RUN useradd -m -s /bin/bash -G sudo user && echo "user:user" | chpasswd && adduser user sudo
RUN echo "user ALL=(ALL) NOPASSWD: ALL" >> /etc/sudoers
RUN echo "export PATH=$PATH:/home/user/.local/bin" >> /home/user/.bashrc
RUN chown -R user:user /home/user

USER user
WORKDIR /home/user

RUN pip install pipenv
RUN git clone --recursive https://github.com/ExistentialRobotics/CORSAIR

WORKDIR /home/user/CORSAIR
ENV PATH="/home/user/.local/bin:${PATH}"
RUN pipenv install --verbose
#RUN pipenv shell && cd deps/MinkowskiEngine && pip install . --verbose

#COPY --chown=user:user data/ShapeNetCore.v2.PC15k /home/user/data/ShapeNetCore.v2.PC15k
#COPY entrypoint.bash /home/user/CORSAIR/entrypoint.bash

#ENTRYPOINT ["/home/user/CORSAIR/entrypoint.bash"]
