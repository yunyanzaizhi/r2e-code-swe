# Code/SWE No-Docker Environment

This adds a repository-level code/SWE environment to the existing verl-agent multi-turn environment stack.

## What It Supports

- R2E-Gym-Lite, R2E-Gym-Subset, SWE-Bench-Lite, and local JSONL task records through a normalized `CodeSWETask` schema.
- Per-episode workspace directories under `env.code_swe.runtime.workspace_root`.
- Tool calls in an R2E-Gym/SWE-agent compatible XML-like format:
  - `bash`
  - `str_replace_editor`
  - `submit`
- User-mode no-Docker execution with command timeout, output clipping, cwd/path validation, restricted environment variables, and blocked high-risk commands.
- Terminal reward on `submit`: export patch, optionally apply `test_patch`, derive/run a local test command, and return 1.0 on exit code 0.

## Important No-Docker Limitations

This runtime does not call Docker, Podman, or the official SWE-Bench Docker harness. If a task only works inside its dataset Docker image or needs unavailable local dependencies, the environment returns reward 0 and records `fail_reason` such as `setup_failed`, `no_test_command`, `install_failed`, or `tests_failed`.

The fallback sandbox is a user-mode workspace plus subprocess policy. It blocks common high-risk commands and sensitive path patterns, but it is not equivalent to kernel namespace/chroot isolation unless such support is added by deployment-specific configuration.

## Smoke Test

```bash
cd /home/caiting/verl-agent-exp-copy-from-lab-server-20260505
.venv/bin/python -m examples.code_swe.smoke_code_swe_env \
  --dataset_name R2E-Gym/R2E-Gym-Lite \
  --split dev_10pr_v1 \
  --max_samples 1
```

This loads 1 task, resets the env, views `/testbed`, submits, and writes a JSONL log. Without a repo cache it should still finish without crashing, usually with `setup_failed`.

## Local Fixture Eval

```bash
.venv/bin/python -m examples.code_swe.run_code_swe_eval \
  --dataset_path /path/to/tasks.jsonl \
  --max_samples 3 \
  --default_test_command "python -m pytest -q"
```

## Small V100 LoRA Training

The launch script intentionally requires `MODEL_PATH`; it does not default to downloading a model.

```bash
export MODEL_PATH=/path/to/local/instruct-model
export CODE_SWE_REPO_CACHE_DIR=/path/to/precloned/repos
bash examples/gigpo_trainer/run_code_swe_lora_v100.sh
```

Default values are tiny: train batch 2, val batch 2, group size 2, max env steps 6, LoRA rank 16, fp16-compatible vLLM/FSDP settings, and one epoch.
