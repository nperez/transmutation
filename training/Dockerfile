# syntax=docker/dockerfile:1
FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y \
    python3 python3-pip python3-venv git \
    && rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python3 /usr/bin/python

WORKDIR /app

# mamba-ssm from git. Compiles CUDA kernels at install time.
# Pin transformers<5 because mamba-ssm imports removed classes from transformers 5.x.
RUN --mount=type=cache,target=/root/.cache/pip \
    --mount=type=bind,source=wheels,target=/tmp/wheels \
    pip install "transformers>=4.39.0,<5" /tmp/wheels/causal_conv1d-*.whl einops && \
    pip install --no-cache-dir --no-build-isolation \
        "mamba-ssm @ git+https://github.com/state-spaces/mamba.git@70f7eaa45db949379417716c0d947789b68f95cc"

# The mamba-ssm wheel doesn't include mamba3 modules yet (packaging bug).
# Copy them from the repo into the installed package.
# Copy ALL mamba3-related files from the repo into the installed package.
# The wheel doesn't include them (packaging bug). We overwrite the entire
# modules/ and ops/ trees to catch any new files mamba3 depends on.
RUN SITE=$(python -c "import mamba_ssm; import os; print(os.path.dirname(mamba_ssm.__file__))") && \
    git clone --filter=blob:none https://github.com/state-spaces/mamba.git /tmp/mamba3-src && \
    cd /tmp/mamba3-src && git checkout 70f7eaa45db949379417716c0d947789b68f95cc && \
    cp -r /tmp/mamba3-src/mamba_ssm/modules/* "$SITE/modules/" && \
    cp -r /tmp/mamba3-src/mamba_ssm/ops/* "$SITE/ops/" && \
    rm -rf /tmp/mamba3-src
COPY patch_mamba3_fp16.py /tmp/patch_mamba3_fp16.py
RUN SITE=$(python -c "import mamba_ssm; import os; print(os.path.dirname(mamba_ssm.__file__))") && \
    python /tmp/patch_mamba3_fp16.py "$SITE"

# Remaining deps — lightweight, change frequently.
COPY requirements.txt .
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r requirements.txt

COPY . /app/training/

RUN useradd -m -u 1000 trainer
USER trainer

ENTRYPOINT ["python"]
