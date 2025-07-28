# syntax=docker/dockerfile:1

#FROM nvidia/cuda:11.8.0-base-ubuntu22.04
FROM nvidia/cuda:11.6.1-cudnn8-runtime-ubuntu20.04

# Install system dependencies
RUN apt-get update && apt-get install -y \
    wget \
    gcc \
    python3-dev \
    build-essential \
    curl \
    bzip2 \
    && rm -rf /var/lib/apt/lists/*

# Install Miniconda to default location ~/miniconda3 and add it to PATH
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /miniconda.sh && \
    bash /miniconda.sh -b -p /root/miniconda3 && \
    rm /miniconda.sh

WORKDIR /reproducibility

# Copy the codebase into the container
COPY . /reproducibility

# Conda setup
RUN /bin/bash -c "/root/miniconda3/bin/conda init bash"
RUN /bin/bash -c "\
    source /root/miniconda3/etc/profile.d/conda.sh && \
    conda config --add channels defaults && \
    conda config --set always_yes yes && \
    conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main && \
    conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r && \
    conda update -n base -c defaults conda && \
    conda env create -f environment.yml && \
    conda activate SIGIRReproducibility && \
    python3 run_compile_all_cython.py && \
    conda clean -a -y"

# Start a login shell to ensure conda is usable
CMD ["/bin/bash", "--login"]