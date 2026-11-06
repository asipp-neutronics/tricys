#!/usr/bin/env bash
# ==============================================================================
# TRICYS V2.0 Unified All-in-One Docker Entrypoint
# Supports:
#   1. Web Services & Microservices:
#      - backend: FastAPI backend service (uvicorn)
#      - visual: Nginx hosting Vue3 web console (port 80)
#      - goview: Nginx hosting GoView digital twin screen (port 80)
#      - hdf5: Dash/Plotly HDF5 interactive server (port 8050)
#      - fullstack / web: Runs backend + hdf5 + nginx all-in-one in single container
#   2. Testing & Workflow Execution:
#      - pytest / test: Runs pytest test suite
#      - run / workflow: Runs tricys CLI workflow
#   3. Interactive CLI & Shell:
#      - bash / sh / * : Runs the given command or drops to shell
# ==============================================================================

set -euo pipefail

APP_USER="appuser"
APP_GROUP="appuser"
WORKSPACE="/workspace"
TRICYS_DIR="${WORKSPACE}/tricys"
H2ISO_DIR="${WORKSPACE}/h2iso"
OUTPUT_DIR="${WORKSPACE}/output"
WEB_ROOT="/usr/share/nginx/html"
WEB_BUNDLE_ROOT="/opt/tricys-web"
NGINX_CONFIG_ROOT="/etc/nginx/tricys"

ensure_owned_dir() {
  local dir_path="$1"
  if [[ -n "${dir_path}" ]]; then
    mkdir -p "${dir_path}"
    chown -R "${APP_USER}:${APP_GROUP}" "${dir_path}" 2>/dev/null || true
  fi
}

ensure_owned_file_parent() {
  local file_path="$1"
  if [[ -n "${file_path}" ]]; then
    mkdir -p "$(dirname "${file_path}")"
    touch "${file_path}" 2>/dev/null || true
    chown "${APP_USER}:${APP_GROUP}" "${file_path}" 2>/dev/null || true
    chown -R "${APP_USER}:${APP_GROUP}" "$(dirname "${file_path}")" 2>/dev/null || true
  fi
}

setup_x11_display() {
  mkdir -p /tmp/.X11-unix
  chmod 1777 /tmp/.X11-unix 2>/dev/null || true

  if [[ -S /tmp/.X11-unix-wslg/X0 ]]; then
    # Windows 11 WSLg environment
    ln -sf /tmp/.X11-unix-wslg/X0 /tmp/.X11-unix/X0 2>/dev/null || true
    export WAYLAND_DISPLAY="${WAYLAND_DISPLAY:-wayland-0}"
    export XDG_RUNTIME_DIR="${XDG_RUNTIME_DIR:-/mnt/wslg/runtime-dir}"
    export PULSE_SERVER="${PULSE_SERVER:-/mnt/wslg/PulseServer}"
  elif [[ -S /tmp/.X11-unix-host/X0 ]]; then
    # Native Linux X11 environment
    ln -sf /tmp/.X11-unix-host/X0 /tmp/.X11-unix/X0 2>/dev/null || true
  fi
  export DISPLAY="${DISPLAY:-:0}"
}
setup_x11_display


start_nginx_app() {
  local app_name="$1"
  local source_dir="${WEB_BUNDLE_ROOT}/${app_name}"
  local source_conf="${NGINX_CONFIG_ROOT}/${app_name}.conf"

  rm -f /etc/nginx/conf.d/default.conf
  rm -f /etc/nginx/sites-enabled/default
  rm -f /etc/nginx/sites-available/default
  cp "${source_conf}" /etc/nginx/conf.d/default.conf

  mkdir -p "${WEB_ROOT}"
  find "${WEB_ROOT}" -mindepth 1 -maxdepth 1 -exec rm -rf {} +
  if [[ -d "${source_dir}" ]]; then
    cp -a "${source_dir}/." "${WEB_ROOT}/"
  fi

  exec nginx -g 'daemon off;'
}

# Configure default Locale, Python & Modelica library paths
export LANG="${LANG:-zh_CN.UTF-8}"
export LC_ALL="${LC_ALL:-zh_CN.UTF-8}"
export LANGUAGE="${LANGUAGE:-zh_CN:zh}"
export PYTHONPATH="${H2ISO_DIR}/src:${WORKSPACE}/tricys_backend:${TRICYS_DIR}/tricys_backend:${TRICYS_DIR}:${PYTHONPATH:-}"
export MODELICAPATH="/home/appuser/.openmodelica/libraries:/usr/lib/omlibrary:${MODELICAPATH:-}"

run_as_appuser() {
  exec gosu "${APP_USER}" env LANG="${LANG}" LC_ALL="${LC_ALL}" LANGUAGE="${LANGUAGE}" PYTHONPATH="${PYTHONPATH}" MODELICAPATH="${MODELICAPATH}" PATH="/home/appuser/.local/bin:${PATH}" "$@"
}

cmd="${1:-backend}"
if [[ $# -gt 0 ]]; then
  shift
fi

case "${cmd}" in
  # --------------------------------------------------------------------------
  # 1. Fullstack Web & Microservices
  # --------------------------------------------------------------------------
  backend)
    sqlite_file_path=""
    database_url="${DATABASE_URL:-}"
    case "${database_url}" in
      sqlite:////*) sqlite_file_path="/${database_url#sqlite:////}" ;;
      sqlite:///*)  sqlite_file_path="${database_url#sqlite://}" ;;
    esac
    ensure_owned_dir "/data"
    ensure_owned_dir "${WORKSPACES_DIR:-/data/workspaces}"
    ensure_owned_dir "${ASSETS_DIR:-/data/assets}"
    ensure_owned_dir "${HDF5_CONTEXTS_DIR:-/data/hdf5_contexts}"
    ensure_owned_file_parent "${sqlite_file_path}"

    for bdir in "${TRICYS_DIR}/tricys_backend" "${WORKSPACE}/tricys_backend"; do
      if [[ -d "${bdir}" ]]; then
        touch "${bdir}/backend.log" 2>/dev/null || true
        chown "${APP_USER}:${APP_GROUP}" "${bdir}/backend.log" 2>/dev/null || true
        chmod 666 "${bdir}/backend.log" 2>/dev/null || true
      fi
    done

    cd "${TRICYS_DIR}" 2>/dev/null || cd /workspace
    if [[ $# -eq 0 ]]; then
      set -- uvicorn tricys_backend.main:app --host 0.0.0.0 --port "${TRICYS_BACKEND_PORT:-8000}"
    elif [[ "$1" == --* ]]; then
      set -- uvicorn tricys_backend.main:app "$@"
    fi
    run_as_appuser "$@"
    ;;

  visual)
    start_nginx_app visual
    ;;

  goview)
    start_nginx_app goview
    ;;

  hdf5)
    context_dir="${HDF5_CONTEXTS_DIR:-/data/hdf5_contexts}"
    port="${HDF5_VISUALIZER_PORT:-8050}"
    base_url="${HDF5_VISUALIZER_BASE_URL:-/hdf5/}"
    secret="${HDF5_VISUALIZER_SECRET:-change-me-before-production}"
    ensure_owned_dir "/data"
    ensure_owned_dir "${context_dir}"

    cd "${TRICYS_DIR}" 2>/dev/null || cd /workspace
    if [[ $# -eq 0 ]]; then
      set -- \
        python3 -m tricys.visualizer.main --server-mode --host 0.0.0.0 --port "${port}" \
        --base-pathname "${base_url}" \
        --secret "${secret}" \
        --context-dir "${context_dir}" \
        --no-browser
    elif [[ "$1" == --* ]]; then
      set -- python3 -m tricys.visualizer.main "$@"
    fi
    run_as_appuser "$@"
    ;;

  all-in-one|fullstack|web)
    echo "================================================================================"
    echo "  Starting TRICYS All-in-One Fullstack Runtime (Backend + HDF5 + Web UI)        "
    echo "================================================================================"
    sqlite_file_path=""
    database_url="${DATABASE_URL:-}"
    case "${database_url}" in
      sqlite:////*) sqlite_file_path="/${database_url#sqlite:////}" ;;
      sqlite:///*)  sqlite_file_path="${database_url#sqlite://}" ;;
    esac
    ensure_owned_dir "/data"
    ensure_owned_dir "${WORKSPACES_DIR:-/data/workspaces}"
    ensure_owned_dir "${ASSETS_DIR:-/data/assets}"
    ensure_owned_dir "${HDF5_CONTEXTS_DIR:-/data/hdf5_contexts}"
    ensure_owned_file_parent "${sqlite_file_path}"

    for bdir in "${TRICYS_DIR}/tricys_backend" "${WORKSPACE}/tricys_backend"; do
      if [[ -d "${bdir}" ]]; then
        touch "${bdir}/backend.log" 2>/dev/null || true
        chown "${APP_USER}:${APP_GROUP}" "${bdir}/backend.log" 2>/dev/null || true
        chmod 666 "${bdir}/backend.log" 2>/dev/null || true
      fi
    done

    cd "${TRICYS_DIR}" 2>/dev/null || cd /workspace

    # 1. Start Backend in background
    gosu "${APP_USER}" env PYTHONPATH="${PYTHONPATH}" MODELICAPATH="${MODELICAPATH}" PATH="/home/appuser/.local/bin:${PATH}" \
      uvicorn tricys_backend.main:app --host 0.0.0.0 --port "${TRICYS_BACKEND_PORT:-8000}" &
    BACKEND_PID=$!

    # 2. Start HDF5 Server in background
    context_dir="${HDF5_CONTEXTS_DIR:-/data/hdf5_contexts}"
    base_url="${HDF5_VISUALIZER_BASE_URL:-/hdf5/}"
    secret="${HDF5_VISUALIZER_SECRET:-change-me-before-production}"
    gosu "${APP_USER}" env PYTHONPATH="${PYTHONPATH}" MODELICAPATH="${MODELICAPATH}" PATH="/home/appuser/.local/bin:${PATH}" \
      python3 -m tricys.visualizer.main --server-mode --host 0.0.0.0 --port "${HDF5_VISUALIZER_PORT:-8050}" \
      --base-pathname "${base_url}" --secret "${secret}" \
      --context-dir "${context_dir}" --no-browser &
    HDF5_PID=$!

    # 3. Configure and start Nginx with allinone.conf
    rm -f /etc/nginx/conf.d/default.conf /etc/nginx/sites-enabled/default /etc/nginx/sites-available/default
    if [[ -f "${NGINX_CONFIG_ROOT}/allinone.conf" ]]; then
      cp "${NGINX_CONFIG_ROOT}/allinone.conf" /etc/nginx/conf.d/default.conf
    elif [[ -f "${TRICYS_DIR}/docker/nginx/nginx_allinone.conf" ]]; then
      cp "${TRICYS_DIR}/docker/nginx/nginx_allinone.conf" /etc/nginx/conf.d/default.conf
    else
      cp "${NGINX_CONFIG_ROOT}/visual.conf" /etc/nginx/conf.d/default.conf
    fi

    nginx -g 'daemon off;' &
    NGINX_PID=$!

    cleanup() {
      echo "[TRICYS] Shutting down services..."
      kill -TERM "${BACKEND_PID}" "${HDF5_PID}" "${NGINX_PID}" 2>/dev/null || true
      wait "${BACKEND_PID}" "${HDF5_PID}" "${NGINX_PID}" 2>/dev/null || true
      exit 0
    }

    trap cleanup SIGTERM SIGINT

    echo "================================================================================"
    echo "  TRICYS All-in-One Ready!                                                      "
    echo "  - Web Console:       http://0.0.0.0:80 (forwarded to host 8080)               "
    echo "  - GoView Big Screen: http://0.0.0.0:3020 (and /goview/)                       "
    echo "  - FastAPI Backend:   http://0.0.0.0:8000 (and /api/)                          "
    echo "  - HDF5 Visualizer:   http://0.0.0.0:8050 (and /hdf5/)                         "
    echo "================================================================================"

    wait -n "${NGINX_PID}" "${BACKEND_PID}" "${HDF5_PID}" || true
    cleanup
    ;;

  # --------------------------------------------------------------------------
  # 2. Testing & Workflow Execution
  # --------------------------------------------------------------------------
  pytest|test)
    echo "================================================================================"
    echo "  TRICYS Test Suite Execution                                                   "
    echo "================================================================================"
    cd "${TRICYS_DIR}"
    run_as_appuser pytest test/ -v "$@"
    ;;

  run|workflow)
    cd "${TRICYS_DIR}"
    run_as_appuser tricys run "$@"
    ;;

  # --------------------------------------------------------------------------
  # 3. Modelica Graphical Modeling (OMEdit GUI)
  # --------------------------------------------------------------------------
  omedit|OMEdit)
    echo "================================================================================"
    echo "  Starting OpenModelica Connection Editor (OMEdit) GUI                          "
    echo "================================================================================"
    cd "${TRICYS_DIR}" 2>/dev/null || cd /workspace
    default_model="${TRICYS_DIR}/tricys/example/example_data/example_model/package.mo"
    if [[ $# -eq 0 && -f "${default_model}" ]]; then
      set -- "${default_model}"
    fi
    run_as_appuser OMEdit "$@"
    ;;

  bash|sh)
    cd "${TRICYS_DIR}" 2>/dev/null || cd /workspace
    exec /bin/bash "$@"
    ;;

  -*)
    exec tricys "${cmd}" "$@"
    ;;

  *)
    exec "${cmd}" "$@"
    ;;
esac
