#!/usr/bin/env bash
set -euo pipefail

# Wait for ANY GPU to become free (low memory usage + no compute apps),
# then run a job script on that GPU.
#
# Usage:
#   bash tools/train/wait_any_gpu_then_run.sh <mem_used_threshold_mb> <job_script> [lock_dir] [gpu_ids_csv]
#
# Examples:
#   bash tools/train/wait_any_gpu_then_run.sh 2000 /path/to/job.sh
#   bash tools/train/wait_any_gpu_then_run.sh 2000 /path/to/job.sh /tmp/run_once.lock "0,1,2,3"
#
# Notes:
# - Uses optional per-GPU lock root: set GPU_LOCK_ROOT=/path/to/locks
# - Uses optional run-once lock directory (LOCK_DIR) to avoid duplicate launches

THRESH_MB="${1:-}"
JOB_SCRIPT="${2:-}"
LOCK_DIR="${3:-}"
GPU_IDS_CSV="${4:-}"
GPU_LOCK_ROOT="${GPU_LOCK_ROOT:-}"

if [[ -z "${THRESH_MB}" || -z "${JOB_SCRIPT}" ]]; then
  echo "Usage: $0 <mem_used_threshold_mb> <job_script> [lock_dir] [gpu_ids_csv]" >&2
  exit 2
fi
if [[ ! -f "${JOB_SCRIPT}" ]]; then
  echo "[wait_any_gpu_then_run] job script not found: ${JOB_SCRIPT}" >&2
  exit 2
fi

echo "[wait_any_gpu_then_run] thresh_mb=${THRESH_MB} job=${JOB_SCRIPT} gpu_ids_csv=${GPU_IDS_CSV:-ALL}"

acquire_gpu_lock () {
  local gpu="$1"
  local root="$2"
  if [[ -z "$root" ]]; then
    echo ""
    return 0
  fi
  mkdir -p "$root"
  local lockdir="${root}/gpu${gpu}.lock"
  local pidfile="${lockdir}/pid"
  # Clean stale lock if owner pid is gone
  if [[ -d "${lockdir}" ]]; then
    if [[ -f "${pidfile}" ]]; then
      local oldpid
      oldpid="$(cat "${pidfile}" 2>/dev/null || true)"
      if [[ -n "${oldpid}" ]] && kill -0 "${oldpid}" 2>/dev/null; then
        return 1
      fi
    fi
    rm -rf "${lockdir}" 2>/dev/null || true
  fi
  if mkdir "${lockdir}" 2>/dev/null; then
    echo "$$" > "${pidfile}" || true
    echo "$(date -Is)" > "${lockdir}/ts" || true
    echo "${lockdir}"
    return 0
  fi
  return 1
}

release_gpu_lock () {
  if [[ -n "${GPU_LOCKDIR:-}" ]] && [[ -d "${GPU_LOCKDIR}" ]]; then
    rm -rf "${GPU_LOCKDIR}" 2>/dev/null || true
  fi
}

get_gpu_ids () {
  local csv="${1:-}"
  if [[ -n "${csv}" ]]; then
    echo "${csv}" | tr ',' ' ' | tr -s ' '
    return 0
  fi
  # IMPORTANT:
  # Do NOT strip newlines here. We need one GPU index per line so that:
  #   for gpu in $(get_gpu_ids); do ...
  # iterates correctly.
  nvidia-smi --query-gpu=index --format=csv,noheader,nounits 2>/dev/null \
    | sed 's/[[:space:]]//g' \
    | sed '/^$/d' || true
}

while true; do
  # If another waiter already launched this job (run-once lock exists), exit ASAP.
  if [[ -n "${LOCK_DIR}" ]] && [[ -d "${LOCK_DIR}" ]]; then
    echo "[wait_any_gpu_then_run] lock already exists (${LOCK_DIR}), exiting."
    exit 0
  fi

  chosen_gpu=""
  for gpu in $(get_gpu_ids "${GPU_IDS_CSV:-}"); do
    used="$(nvidia-smi --id="${gpu}" --query-gpu=memory.used --format=csv,noheader,nounits | head -n1 | tr -d '[:space:]' || true)"
    pids="$(nvidia-smi --id="${gpu}" --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null | tr -d '[:space:]' | sed '/^$/d' || true)"
    if [[ -n "${used}" ]] && [[ "${used}" =~ ^[0-9]+$ ]] && (( used <= THRESH_MB )); then
      if [[ -z "${pids}" ]]; then
        if GPU_LOCKDIR="$(acquire_gpu_lock "${gpu}" "${GPU_LOCK_ROOT}")"; then
          if [[ -n "${GPU_LOCKDIR}" ]]; then
            echo "[wait_any_gpu_then_run] acquired gpu lock: ${GPU_LOCKDIR}"
            trap release_gpu_lock EXIT INT TERM
          fi
          chosen_gpu="${gpu}"
          echo "[wait_any_gpu_then_run] gpu=${chosen_gpu} is free (mem_used=${used}MiB <= ${THRESH_MB}MiB, no compute apps). launching..."
          break
        fi
      fi
    fi
  done

  if [[ -n "${chosen_gpu}" ]]; then
    break
  fi
  sleep 60
done

chmod +x "${JOB_SCRIPT}" || true
export CUDA_VISIBLE_DEVICES="${chosen_gpu}"
export NVIDIA_VISIBLE_DEVICES="${chosen_gpu}"

# Optional: lock to avoid launching the same job multiple times from multiple watchers.
if [[ -n "${LOCK_DIR}" ]]; then
  if mkdir "${LOCK_DIR}" 2>/dev/null; then
    echo "[wait_any_gpu_then_run] acquired lock: ${LOCK_DIR}"
  else
    echo "[wait_any_gpu_then_run] lock already exists (${LOCK_DIR}), exiting."
    exit 0
  fi
fi

# Final sanity check after acquiring lock (avoid last-moment races).
used2="$(nvidia-smi --id="${chosen_gpu}" --query-gpu=memory.used --format=csv,noheader,nounits | head -n1 | tr -d '[:space:]' || true)"
pids2="$(nvidia-smi --id="${chosen_gpu}" --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null | tr -d '[:space:]' | sed '/^$/d' || true)"
if [[ -n "${used2}" ]] && [[ "${used2}" =~ ^[0-9]+$ ]] && (( used2 > THRESH_MB )); then
  echo "[wait_any_gpu_then_run] post-lock check failed: mem_used=${used2}MiB > ${THRESH_MB}MiB, exiting."
  exit 1
fi
if [[ -n "${pids2}" ]]; then
  echo "[wait_any_gpu_then_run] post-lock check failed: compute apps exist (pids=${pids2}), exiting."
  exit 1
fi

set +e
bash "${JOB_SCRIPT}"
exit_code=$?
set -e
exit "${exit_code}"


