# syntax=docker/dockerfile:1.7

ARG NODE_BASE_IMAGE=node:20-alpine

FROM ${NODE_BASE_IMAGE} AS tricys_visual_build

ARG VITE_API_URL=/api/v1
ARG VITE_API_V2_URL=/api/v2/goview
ARG VITE_GOVIEW_URL=/goview/

ENV VITE_API_URL=${VITE_API_URL} \
    VITE_API_V2_URL=${VITE_API_V2_URL} \
    VITE_GOVIEW_URL=${VITE_GOVIEW_URL}

WORKDIR /app/tricys_visual

COPY tricys_visual/package.json ./package.json

RUN npm install

COPY tricys_visual/ ./

RUN npm run build

FROM tricys_visual_build AS tricys_goview_build

ARG VITE_PRO_PATH=

ENV HUSKY=0 \
    VITE_PRO_PATH=${VITE_PRO_PATH}

WORKDIR /app/tricys_goview

COPY tricys_goview/package.json ./package.json

RUN npm install

COPY tricys_goview/ ./

RUN npm run build

FROM openmodelica/openmodelica:v1.24.5-ompython AS base_om

USER root

RUN apt-get update && apt-get install -y \
    vim \
    curl \
    git \
    sudo \
    ca-certificates \
    nginx \
    && rm -rf /var/lib/apt/lists/*

RUN useradd --create-home --shell /bin/bash appuser && \
    usermod -aG sudo appuser && \
    echo "appuser ALL=(ALL) NOPASSWD: ALL" >> /etc/sudoers

FROM base_om AS tricys_runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /tricys

COPY . /tricys

RUN chown -R appuser:appuser /tricys

USER appuser

RUN pip install --upgrade pip setuptools wheel

RUN pip install -e ".[dev]" && \
    omc ./script/modelica_install/install.mos

RUN pip install -r tricys_backend/requirements.txt

ENV PATH="/home/appuser/.local/bin:${PATH}"

USER root

COPY docker/scripts/tricys_backend_entrypoint.sh /usr/local/bin/tricys_backend_entrypoint.sh
COPY docker/scripts/tricys_runtime_entrypoint.sh /usr/local/bin/tricys_runtime_entrypoint.sh
COPY docker/nginx/nginx_visual.conf /etc/nginx/tricys/visual.conf
COPY docker/nginx/nginx_goview.conf /etc/nginx/tricys/goview.conf
COPY --from=tricys_visual_build /app/tricys_visual/dist /opt/tricys-web/visual
COPY --from=tricys_goview_build /app/tricys_goview/dist /opt/tricys-web/goview

RUN chmod +x /usr/local/bin/tricys_backend_entrypoint.sh /usr/local/bin/tricys_runtime_entrypoint.sh && \
    mkdir -p /run/nginx /var/lib/nginx /var/log/nginx /usr/share/nginx/html /etc/nginx/tricys && \
    rm -f /etc/nginx/conf.d/default.conf

WORKDIR /tricys

EXPOSE 80 8000 8050

ENTRYPOINT ["/usr/local/bin/tricys_runtime_entrypoint.sh"]

CMD ["backend"]