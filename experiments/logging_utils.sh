#!/usr/bin/env bash

log_slug() {
    local value="${1:-unknown}"
    value="$(printf '%s' "$value" | tr '[:upper:]' '[:lower:]')"
    value="$(printf '%s' "$value" | sed -E 's#[^a-z0-9._-]+#_#g; s#^_+##; s#_+$##')"
    if [[ -z "$value" ]]; then
        value="unknown"
    fi
    printf '%s' "$value"
}

log_gpu_bucket() {
    local gpu_count="${1:-unknown}"
    case "$gpu_count" in
        1) printf 'single_gpu' ;;
        2) printf 'dual_gpu' ;;
        ''|unknown|Unknown|UNKNOWN) printf 'unknown_gpu' ;;
        *) printf 'multi_gpu_%s' "$(log_slug "$gpu_count")" ;;
    esac
}

make_log_path() {
    local env_name
    local gpu_count
    local run_name
    env_name="$(log_slug "${1:-unknown}")"
    gpu_count="${2:-unknown}"
    run_name="$(log_slug "${3:-run}")"

    local log_root="${EXPERIMENT_LOG_ROOT:-experiments/logs}"
    local timestamp="${EXPERIMENT_LOG_TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}"
    local gpu_bucket
    gpu_bucket="$(log_gpu_bucket "$gpu_count")"

    local log_dir="${log_root}/${env_name}/${gpu_bucket}"
    mkdir -p "$log_dir"
    printf '%s/%s_%s.log\n' "$log_dir" "$run_name" "$timestamp"
}

announce_log_path() {
    local log_file="$1"
    echo "[logging] writing stdout/stderr to ${log_file}"
}
