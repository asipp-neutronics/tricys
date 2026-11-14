#!/usr/bin/env bash
# ==============================================================================
# TRICYS OMEdit CLI Wrapper
# Automatically injects the default example_model package when invoked without args
# ==============================================================================

set -euo pipefail

# Ensure UTF-8 locale for proper Chinese comment and text rendering
export LANG="${LANG:-zh_CN.UTF-8}"
export LC_ALL="${LC_ALL:-zh_CN.UTF-8}"
export LANGUAGE="${LANGUAGE:-zh_CN:zh}"
export DISPLAY="${DISPLAY:-:0}"

# Auto-bridge X11 socket if needed
if [[ ! -S /tmp/.X11-unix/X0 ]]; then
  mkdir -p /tmp/.X11-unix 2>/dev/null || true
  if [[ -S /tmp/.X11-unix-wslg/X0 ]]; then
    ln -sf /tmp/.X11-unix-wslg/X0 /tmp/.X11-unix/X0 2>/dev/null || true
  elif [[ -S /tmp/.X11-unix-host/X0 ]]; then
    ln -sf /tmp/.X11-unix-host/X0 /tmp/.X11-unix/X0 2>/dev/null || true
  fi
fi

DEFAULT_MODEL="/workspace/tricys/tricys/example/example_data/example_model/package.mo"
FALLBACK_MODEL="/workspace/example_data/example_model/package.mo"

if [[ $# -eq 0 ]]; then
  if [[ -f "${DEFAULT_MODEL}" ]]; then
    set -- "${DEFAULT_MODEL}"
  elif [[ -f "${FALLBACK_MODEL}" ]]; then
    set -- "${FALLBACK_MODEL}"
  fi
fi

exec /usr/bin/OMEdit "$@"
