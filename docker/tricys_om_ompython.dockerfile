FROM openmodelica/openmodelica:v1.24.5-ompython AS base_om

USER root

RUN apt-get update && apt-get install -y \
    vim \
    curl \
    git \
    sudo \
    && rm -rf /var/lib/apt/lists/*

RUN useradd --create-home --shell /bin/bash appuser && \
    usermod -aG sudo appuser && \
    echo "appuser ALL=(ALL) NOPASSWD: ALL" >> /etc/sudoers

FROM base_om AS tricys

WORKDIR /tricys

COPY . /tricys

RUN chown -R appuser:appuser /tricys

USER appuser

RUN pip install --upgrade pip setuptools wheel

RUN make dev-install

ENV PATH="/home/appuser/.local/bin:${PATH}"
