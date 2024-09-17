FROM nvcr.io/nvidia/cuda:12.5.1-cudnn-devel-ubuntu22.04

ARG TENSORRT_VERSION=10.2.0.19
ARG CUDA_USER_VERSION=12.5
ARG CUDNN_USER_VERSION=8.9
ARG OPERATING_SYSTEM=Linux

ENV DEBIAN_FRONTEND noninteractive

# Install package dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        autoconf \
        automake \
        libtool \
        pkg-config \
        ca-certificates \
        wget \
        git \
        curl \
        libjpeg-dev \
        libpng-dev \
        language-pack-en \
        locales \
        locales-all \
        python3 \
        python3-dev \
        python3-pip \
        python3-setuptools \
        libprotobuf-dev \
        protobuf-compiler \
        zlib1g-dev \
        swig \
        vim \
        gdb \
        valgrind \
        libsm6 \
        libxext6 \
        libxrender-dev \
        cmake && \
    apt-get clean
    
# create user
ARG USER
ARG USER_ID
ARG GROUP_ID
RUN useradd -m ${USER} --uid=${USER_ID} 

RUN cd /usr/local/bin && \
    ln -s /usr/bin/python3 python && \
    ln -s /usr/bin/pip3 pip && \
    pip install --upgrade pip setuptools wheel
    
# install pyhton libs
USER ${USER_ID}:${GROUP_ID}
ENV PATH="/home/${USER}/.local/bin:${PATH}"
WORKDIR /home/${USER}
COPY requirements.txt /home/${USER}/requirements.txt
RUN python3 -m pip install --upgrade pip
RUN pip3 install --user --no-warn-script-location -r requirements.txt
RUN python3 -m pip install onnx_graphsurgeon --extra-index-url https://pypi.ngc.nvidia.com

# System locale
# Important for UTF-8
ENV LC_ALL en_US.UTF-8
ENV LANG en_US.UTF-8
ENV LANGUAGE en_US.UTF-8
# Add UTF-8 support
#RUN apt-get update && \
#    apt-get install -y --no-install-recommends locales && \
#    locale-gen en_US.UTF-8

COPY ./TensorRT-${TENSORRT_VERSION}.${OPERATING_SYSTEM}.x86_64-gnu.cuda-${CUDA_USER_VERSION}.tar.gz /opt
RUN cd /opt && \
    tar -xzf TensorRT-${TENSORRT_VERSION}.${OPERATING_SYSTEM}.x86_64-gnu.cuda-${CUDA_USER_VERSION}.tar.gz && \
    rm TensorRT-${TENSORRT_VERSION}.${OPERATING_SYSTEM}.x86_64-gnu.cuda-${CUDA_USER_VERSION}.tar.gz && \
    export PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}' | cut -d. -f1,2 | tr -d .) && \
    python3 -m pip install TensorRT-${TENSORRT_VERSION}/python/tensorrt-*-cp${PYTHON_VERSION}-none-linux_x86_64.whl && \
    python3 -m pip install TensorRT-${TENSORRT_VERSION}/python/tensorrt_lean-*-cp${PYTHON_VERSION}-none-linux_x86_64.whl && \
    python3 -m pip install TensorRT-${TENSORRT_VERSION}/python/tensorrt_dispatch-*-cp${PYTHON_VERSION}-none-linux_x86_64.whl
    #python3 -m pip install TensorRT-${TENSORRT_VERSION}/onnx_graphsurgeon/onnx_graphsurgeon-0.5.1-py2.py3-none-any.whl

ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/TensorRT-${TENSORRT_VERSION}/lib
ENV PATH=$PATH:/opt/TensorRT-${TENSORRT_VERSION}/bin

