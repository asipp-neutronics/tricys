# syntax=docker/dockerfile:1.7
# ==============================================================================
# TRICYS V2.0 Unified All-in-One Multi-Physics & Fullstack Docker Image
# Multi-fidelity Tritium Fuel Cycle Simulation & Digital Twin Platform:
#   1. Physics Kernels:
#      - FESTIM v2.0 + DOLFINx 0.9.0 + PETSc + SLEPc (Component-level FEM)
#      - OpenModelica 1.27.1 + OMSimulator + OMPython (System DAE & SSP/FMI)
#      - Solidified Modelica Standard Library (MSL 3.2.3, 4.0.0, 4.1.0)
#      - CasADi >=3.6 (H2ISO Cryogenic Distillation MESH Solver)
#      - FMPy >=0.3.26 (Multi-rate Co-Simulation & FMU Runner)
#   2. Fullstack Web & Digital Twin Services:
#      - FastAPI + Uvicorn + SQLite + WebSockets (Industrial Backend API)
#      - Vue 3 + Vite (tricys_visual Industrial Web Console)
#      - GoView (tricys_goview Digital Twin Big Screen)
#      - Dash + Plotly + PyTables (HDF5 Scientific Interactive Visualizer)
#      - Nginx Reverse Proxy & Static Asset Hosting
# ==============================================================================

ARG NODE_BASE_IMAGE=node:20-alpine
ARG BASE_IMAGE=dolfinx/dolfinx:v0.9.0

# ==============================================================================
# Stage 1: Build tricys_visual (Vue 3 + Vite Web Console)
# ==============================================================================
FROM ${NODE_BASE_IMAGE} AS tricys_visual_build

ARG VITE_API_URL=/api/v1
ARG VITE_API_V2_URL=/api/v2/goview
ARG VITE_GOVIEW_URL=/goview/

ENV VITE_API_URL=${VITE_API_URL} \
    VITE_API_V2_URL=${VITE_API_V2_URL} \
    VITE_GOVIEW_URL=${VITE_GOVIEW_URL}

WORKDIR /app/tricys_visual
COPY tricys_visual/ ./
RUN if [ ! -f "dist/index.html" ]; then \
        npm config set registry https://registry.npmmirror.com || true && \
        npm install && \
        npm run build; \
    fi

# ==============================================================================
# Stage 2: Build tricys_goview (Digital Twin Screen)
# ==============================================================================
FROM tricys_visual_build AS tricys_goview_build

ARG VITE_PRO_PATH=
ENV HUSKY=0 \
    VITE_PRO_PATH=${VITE_PRO_PATH}

WORKDIR /app/tricys_goview
COPY tricys_goview/ ./
RUN if [ ! -f "dist/index.html" ]; then \
        npm config set registry https://registry.npmmirror.com || true && \
        npm install && \
        npm run build; \
    fi

# ==============================================================================
# Stage 3: TRICYS Unified Scientific, Simulation & Fullstack Runtime Base
# ==============================================================================
FROM ${BASE_IMAGE} AS tricys

USER root

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    DEBIAN_FRONTEND=noninteractive \
    PATH="/home/appuser/.local/bin:/dolfinx-env/bin:${PATH}" \
    MODELICAPATH="/home/appuser/.openmodelica/libraries:/usr/lib/omlibrary" \
    PYTHONPATH="/workspace/h2iso/src:/workspace/tricys:/workspace/tricys_backend:/workspace/tricys/tricys_backend:${PYTHONPATH:-}" \
    HDF5_VISUALIZER_PORT=8050 \
    HDF5_VISUALIZER_BASE_URL=/hdf5/ \
    HDF5_CONTEXTS_DIR=/data/hdf5_contexts

# 1. System utilities + OpenModelica 1.27 + OMSimulator + Nginx
RUN sed -i 's@http://.*archive.ubuntu.com@https://mirrors.tuna.tsinghua.edu.cn@g' /etc/apt/sources.list.d/ubuntu.sources 2>/dev/null || true && \
    sed -i 's@http://security.ubuntu.com@https://mirrors.tuna.tsinghua.edu.cn@g' /etc/apt/sources.list.d/ubuntu.sources 2>/dev/null || true && \
    echo 'deb [trusted=yes] http://build.openmodelica.org/apt noble release' > /etc/apt/sources.list.d/openmodelica.list && \
    apt-get update && apt-get install -y --no-install-recommends \
        curl \
        git \
        sudo \
        vim \
        ca-certificates \
        build-essential \
        cmake \
        libhdf5-dev \
        gosu \
        omc \
        omsimulator \
        nginx \
    && rm -rf /var/lib/apt/lists/*

# 2. Standardize non-root appuser (UID 1000) for OpenModelica security compliance
RUN if id -u appuser >/dev/null 2>&1; then \
        echo "appuser already exists"; \
    elif id -u ubuntu >/dev/null 2>&1; then \
        usermod -l appuser ubuntu && \
        groupmod -n appuser ubuntu && \
        usermod -d /home/appuser -m appuser; \
    else \
        useradd -u 1000 -m -s /bin/bash appuser; \
    fi && \
    usermod -aG sudo appuser && \
    grep -q "appuser ALL=(ALL) NOPASSWD: ALL" /etc/sudoers || echo "appuser ALL=(ALL) NOPASSWD: ALL" >> /etc/sudoers

# 3. Configure pip mirror and install fullstack Scientific + Backend + Visualization stack
RUN pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple && \
    pip config set global.trusted-host pypi.tuna.tsinghua.edu.cn && \
    pip install \
        "scifem==0.5.0" \
        "FESTIM==2.0" \
        OMPython \
        OMSimulator \
        "casadi>=3.6" \
        "fmpy>=0.3.26" \
        "scipy>=1.8" \
        "matplotlib>=3.5" \
        seaborn \
        pandas \
        tables \
        pyyaml \
        pytest \
        SALib \
        openai \
        python-dotenv \
        tabulate \
        python-json-logger \
        python-docx \
        pymupdf \
        fastapi \
        uvicorn \
        websockets \
        wsproto \
        sqlmodel \
        psutil \
        aiofiles \
        python-multipart \
        pydantic-settings \
        requests \
        h5py \
        "python-jose[cryptography]" \
        "passlib[bcrypt]" \
        "bcrypt==4.0.1" \
        "dash>=2.14.0" \
        "dash-bootstrap-components>=1.5.0" \
        "plotly>=5.18.0"

# 4. Solidify Modelica Standard Library (MSL) into container filesystem (Self-healing from upstream repo)
RUN (curl -fSL --retry 3 --connect-timeout 15 https://build.openmodelica.org/apt/pool/contrib-noble/omlibrary_1.27.1-1_all.deb -o /tmp/omlibrary.deb || \
     curl -fSL --retry 3 --connect-timeout 15 http://build.openmodelica.org/apt/pool/contrib-noble/omlibrary_1.27.1-1_all.deb -o /tmp/omlibrary.deb) && \
    dpkg -i /tmp/omlibrary.deb && \
    rm -f /tmp/omlibrary.deb && \
    mkdir -p /home/appuser/.openmodelica/libraries /usr/lib/omlibrary && \
    cd /usr/lib/omlibrary && \
    unzip -q /usr/share/omlibrary/cache/7a4bf7de77a3986e8eb1e88cbb515d646f78f834.zip && \
    mv OpenModelica-ModelicaStandardLibrary-*/Modelica 'Modelica 4.1.0+maint.om' && \
    mv OpenModelica-ModelicaStandardLibrary-*/Complex.mo 'Complex 4.1.0+maint.om.mo' && \
    mv OpenModelica-ModelicaStandardLibrary-*/ModelicaServices 'ModelicaServices 4.1.0+maint.om' && \
    (mv OpenModelica-ModelicaStandardLibrary-*/ModelicaReference 'ModelicaReference 4.1.0+maint.om' 2>/dev/null || true) && \
    rm -rf OpenModelica-ModelicaStandardLibrary-* && \
    unzip -q /usr/share/omlibrary/cache/96032134c36668898e1693e69bd9f81aa38de3dd.zip && \
    mv OpenModelica-ModelicaStandardLibrary-*/Modelica 'Modelica 4.0.0+maint.om' && \
    mv OpenModelica-ModelicaStandardLibrary-*/Complex.mo 'Complex 4.0.0+maint.om.mo' && \
    mv OpenModelica-ModelicaStandardLibrary-*/ModelicaServices 'ModelicaServices 4.0.0+maint.om' && \
    rm -rf OpenModelica-ModelicaStandardLibrary-* && \
    unzip -q /usr/share/omlibrary/cache/efd981a1176f124938d6d6759f7c09e0fbf55ddf.zip && \
    mv OpenModelica-ModelicaStandardLibrary-*/Modelica 'Modelica 3.2.3+maint.om' && \
    mv OpenModelica-ModelicaStandardLibrary-*/Complex.mo 'Complex 3.2.3+maint.om.mo' && \
    mv OpenModelica-ModelicaStandardLibrary-*/ModelicaServices 'ModelicaServices 3.2.3+maint.om' && \
    rm -rf OpenModelica-ModelicaStandardLibrary-* && \
    ln -sf 'Modelica 4.1.0+maint.om' 'Modelica' && \
    ln -sf 'Complex 4.1.0+maint.om.mo' 'Complex.mo' && \
    cp /usr/share/omlibrary/cache/index.json ./ && \
    cp -ru /usr/lib/omlibrary/* /home/appuser/.openmodelica/libraries/ && \
    rm -rf /root/.openmodelica && \
    ln -s /home/appuser/.openmodelica /root/.openmodelica && \
    chown -R appuser:appuser /home/appuser/.openmodelica /usr/lib/omlibrary && \
    rm -rf /usr/share/omlibrary

# 5. Web Frontend Artifacts & Nginx Configuration
COPY docker/nginx/nginx_visual.conf /etc/nginx/tricys/visual.conf
COPY docker/nginx/nginx_goview.conf /etc/nginx/tricys/goview.conf
COPY docker/nginx/nginx_allinone.conf /etc/nginx/tricys/allinone.conf
COPY --from=tricys_visual_build /app/tricys_visual/dist /opt/tricys-web/visual
COPY --from=tricys_goview_build /app/tricys_goview/dist /opt/tricys-web/goview
RUN mkdir -p /run/nginx /var/lib/nginx /var/log/nginx /usr/share/nginx/html /etc/nginx/tricys && \
    rm -f /etc/nginx/conf.d/default.conf

# 6. Prepare workspace directories and copy project source
WORKDIR /workspace
RUN mkdir -p /workspace/tricys /workspace/tricys_backend /workspace/h2iso /workspace/output /data /data/workspaces /data/assets /data/hdf5_contexts && \
    chown -R appuser:appuser /workspace /data /tmp /opt/tricys-web

COPY . /workspace/tricys
RUN cd /workspace/tricys && (pip install --no-deps -e . || true)

# 7. Global CLI wrapper and entrypoints
COPY docker/scripts/tricys_hdf5_cli.sh /usr/local/bin/tricys
COPY docker/scripts/tricys_backend_entrypoint.sh /usr/local/bin/tricys_backend_entrypoint.sh
COPY docker/scripts/tricys_entrypoint.sh /usr/local/bin/tricys_entrypoint.sh
RUN chmod +x /usr/local/bin/tricys \
             /usr/local/bin/tricys_backend_entrypoint.sh \
             /usr/local/bin/tricys_entrypoint.sh

EXPOSE 80 8000 3020 8050

ENTRYPOINT ["/usr/local/bin/tricys_entrypoint.sh"]
CMD ["all-in-one"]