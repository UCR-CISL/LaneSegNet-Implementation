## Use the CUDA *devel* image instead of the smaller runtime image.  The devel
## variant includes nvcc and CUDA headers, which are required to compile
## mmcv‑full and other custom CUDA ops.  Without these, mmcv builds only CPU
## extensions and will fail at runtime when CUDA kernels such as
## `ms_deform_attn_impl_forward` are invoked.
FROM nvidia/cuda:11.7.1-cudnn8-devel-ubuntu20.04

## -----------------------------------------------------------------------------
## LaneSegNet Docker image
##
## This Dockerfile builds an environment capable of running the
## OpenDriveLab/LaneSegNet project without needing to train from scratch.  It
## installs the specific versions of PyTorch and the OpenMMLab libraries
## requested by the user: torch==2.0.0, torchvision==0.15.0, mmcls==0.25.0,
## mmcv‑full==1.6.0, mmdet==2.26.0, mmdet3d==1.0.0rc6 and
## mmsegmentation==0.29.1.  The image is based on NVIDIA’s CUDA 11.7 runtime
## image to make GPU acceleration available.  If you don’t have an NVIDIA GPU
## you can swap the base image for a non‑GPU Ubuntu image and install the
## CPU‑only versions of PyTorch.

## Use noninteractive apt to avoid tzdata prompts during build
ENV DEBIAN_FRONTEND=noninteractive

## Install basic development tools and Python 3.8 on Ubuntu 20.04.  Ubuntu 20.04
## includes python3.8 packages, which are required for mmdet3d’s older
## dependencies (e.g. numba==0.53.0).  We also install a full build toolchain
## (gcc, g++, make, ninja, cmake) and some image/linear algebra libraries used by
## OpenMMLab.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
       git \
       wget \
       curl \
       ca-certificates \
       python3.8 \
       python3.8-dev \
       python3.8-distutils \
       python3-pip \
       build-essential \
       ninja-build \
       cmake \
       libopenblas-dev \
       libeigen3-dev \
       libjpeg-dev \
       zlib1g-dev \
       libturbojpeg-dev \
       libpng-dev \
       libgl1 \
       libglib2.0-0 \
       libsm6 \
    && rm -rf /var/lib/apt/lists/*

## Use python3.8 as the default `python` command and upgrade pip.
RUN ln -sf /usr/bin/python3.8 /usr/bin/python && python -m pip install --upgrade pip

## -----------------------------------------------------------------------------
## Install Python packages
##
## The wheels below correspond to PyTorch 2.0.0 and torchvision 0.15.0 built
## against CUDA 11.7.  The extra‑index‑url flag tells pip to look at
## download.pytorch.org for the prebuilt CUDA wheels.  If you want CUDA 11.8
## simply change `cu117` to `cu118` in the URLs.

RUN python -m pip install --no-cache-dir \
        torch==2.0.0+cu117 \
        torchvision==0.15.0+cu117 \
        --extra-index-url https://download.pytorch.org/whl/cu117

## Install mmengine (required by the new MM series) and OpenMIM for managing
## OpenMMLab packages.  mmengine is pinned to a known working version.
RUN python -m pip install --no-cache-dir \
        mmengine==0.7.4 \
        openmim==0.3.7

## -----------------------------------------------------------------------------
## Build mmcv from source with CUDA ops
##
## The stock pip wheels for mmcv-full do not include CUDA kernels for
## PyTorch 2.0/11.7, which leads to runtime errors such as
## `ms_deform_attn_impl_forward: implementation for device cuda:0 not found`.
## To ensure the necessary CUDA kernels are built, we clone the mmcv
## repository at v1.6.0 and install it in editable mode while setting
## `MMCV_WITH_OPS=1` and `FORCE_CUDA=1`.  These flags trigger the
## compilation of CUDA ops using nvcc provided by the devel base image.
RUN git clone --depth=1 --branch v1.6.0 https://github.com/open-mmlab/mmcv.git /tmp/mmcv \
    && cd /tmp/mmcv \
    && python -m pip install --no-cache-dir -r requirements/runtime.txt \
    && TORCH_CUDA_ARCH_LIST="6.1;7.5;8.0;8.6" MMCV_WITH_OPS=1 FORCE_CUDA=1 python -m pip install --no-cache-dir . \
    && rm -rf /tmp/mmcv

## Install the remaining OpenMMLab libraries.  These versions match those
## recommended by the LaneSegNet authors.  We install them after mmcv so
## that they detect the already-built mmcv during their setup.  Note that
## mmdet3d has an rc tag and depends on numba==0.53.0, which is only
## available for Python <3.10, hence our use of Python 3.8.
RUN python -m pip install --no-cache-dir \
        mmcls==0.25.0 \
        mmdet==2.26.0 \
        mmdet3d==1.0.0rc6 \
        mmsegmentation==0.29.1

## Install project‑specific requirements from the repository’s requirements.txt.
## These versions come directly from the LaneSegNet repo and include
## similaritymeasures, numpy, scipy, ortools, setuptools and openlanev2.  Pinning
## them here ensures reproducibility.
RUN pip3 install \
    -i https://team-n8fgjuae.pypimirror.stablebuild.com/2025-06-19/ \
    ortools==9.2.9972
RUN python -m pip install --no-cache-dir \
        similaritymeasures==0.6.0 \
        numpy==1.22.4 \
        scipy==1.8.0 \
        setuptools==59.5.0 \
        openlanev2==2.1.0

## -----------------------------------------------------------------------------
## Clone the LaneSegNet repository
WORKDIR /workspace
RUN git clone --depth=1 https://github.com/OpenDriveLab/LaneSegNet.git
WORKDIR /workspace/LaneSegNet

## Set PYTHONPATH so that modules in LaneSegNet/projects can be imported.  We
## append our project directory to the existing PYTHONPATH rather than
## overriding it entirely.  This allows the interpreter to find
## system‑installed packages such as mmcv.
ENV PYTHONPATH=/workspace/LaneSegNet:${PYTHONPATH}

## -----------------------------------------------------------------------------
## Helper scripts
##
## download_ckpt.sh: downloads the pretrained LaneSegNet checkpoint from
## HuggingFace.  The user should execute this script manually after the image
## is built.  A fast multi‑threaded download via aria2c is attempted when
## available, falling back to wget otherwise.
RUN echo '#!/bin/bash' > /workspace/LaneSegNet/download_ckpt.sh \
    && echo 'set -e' >> /workspace/LaneSegNet/download_ckpt.sh \
    && echo 'echo "Downloading LaneSegNet pretrained checkpoint..."' >> /workspace/LaneSegNet/download_ckpt.sh \
    && echo 'CKPT_URL="https://huggingface.co/opendrivelab/lanesegnet/resolve/main/lanesegnet_r50_8x1_24e_olv2_subset_A.pth"' >> /workspace/LaneSegNet/download_ckpt.sh \
    && echo 'OUT_FILE="lanesegnet_pretrained.pth"' >> /workspace/LaneSegNet/download_ckpt.sh \
    && echo 'if command -v aria2c > /dev/null; then' >> /workspace/LaneSegNet/download_ckpt.sh \
    && echo '  aria2c -x 16 -s 16 -o "$OUT_FILE" "$CKPT_URL"' >> /workspace/LaneSegNet/download_ckpt.sh \
    && echo 'else' >> /workspace/LaneSegNet/download_ckpt.sh \
    && echo '  wget -O "$OUT_FILE" "$CKPT_URL"' >> /workspace/LaneSegNet/download_ckpt.sh \
    && echo 'fi' >> /workspace/LaneSegNet/download_ckpt.sh \
    && chmod +x /workspace/LaneSegNet/download_ckpt.sh

## run_eval.sh: runs distributed testing on the pretrained model and writes
## visualization results to the `results` directory.  The first argument
## specifies the number of GPUs to use (default: 1).  The script checks for
## the presence of the checkpoint and instructs the user to download it if
## missing.  Visualization is enabled via the `--show` flag.
RUN echo '#!/bin/bash' > /workspace/LaneSegNet/run_eval.sh \
    && echo 'set -e' >> /workspace/LaneSegNet/run_eval.sh \
    && echo 'GPUS="${1:-1}"' >> /workspace/LaneSegNet/run_eval.sh \
    && echo 'CONFIG="projects/configs/lanesegnet_r50_8x1_24e_olv2_subset_A.py"' >> /workspace/LaneSegNet/run_eval.sh \
    && echo 'CKPT="lanesegnet_pretrained.pth"' >> /workspace/LaneSegNet/run_eval.sh \
    && echo 'OUTDIR="results"' >> /workspace/LaneSegNet/run_eval.sh \
    && echo 'mkdir -p "$OUTDIR"' >> /workspace/LaneSegNet/run_eval.sh \
    && echo 'if [ ! -f "$CKPT" ]; then' >> /workspace/LaneSegNet/run_eval.sh \
    && echo '  echo "Checkpoint $CKPT not found. Please run ./download_ckpt.sh to fetch it."' >> /workspace/LaneSegNet/run_eval.sh \
    && echo '  exit 1' >> /workspace/LaneSegNet/run_eval.sh \
    && echo 'fi' >> /workspace/LaneSegNet/run_eval.sh \
    && echo './tools/dist_test.sh "$GPUS" "$CONFIG" "$CKPT" --show --out-dir "$OUTDIR"' >> /workspace/LaneSegNet/run_eval.sh \
    && chmod +x /workspace/LaneSegNet/run_eval.sh

## The default command simply launches a Bash shell.  After starting a
## container from this image, the user can call ./download_ckpt.sh and
## ./run_eval.sh as needed.
CMD ["/bin/bash"]