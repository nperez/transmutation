# syntax=docker/dockerfile:1
FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y \
    python3 python3-pip python3-venv git \
    && rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python3 /usr/bin/python

WORKDIR /app

# Torch first — large download, cached as its own layer.
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install torch==2.5.1

# CUDA kernel packages — pre-built wheels, bind-mounted from host.
# Downloaded by run.sh on first use; cached in training/wheels/ for subsequent builds.
# Pin transformers<5 because mamba-ssm 2.2.4 imports removed classes from transformers 5.x.
RUN --mount=type=bind,source=wheels,target=/tmp/wheels \
    pip install "transformers>=4.39.0,<5" /tmp/wheels/causal_conv1d-*.whl /tmp/wheels/mamba_ssm-*.whl

# Remaining deps — lightweight, change frequently.
COPY requirements.txt .
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r requirements.txt

COPY . /app/training/

ENTRYPOINT ["python"]
