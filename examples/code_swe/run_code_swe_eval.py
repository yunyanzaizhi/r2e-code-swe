import argparse
import json
from datetime import datetime
from pathlib import Path

from agent_system.environments.env_package.code_swe.envs import CodeSWEEnv
from agent_system.environments.env_package.code_swe.runtime import RuntimeConfig
from agent_system.environments.env_package.code_swe.tasks import load_tasks_from_hf, load_tasks_from_jsonl


def load_tasks(args):
    if args.dataset_path:
        return load_tasks_from_jsonl(args.dataset_path, dataset_name="local", split=args.split)[: args.max_samples]
    return load_tasks_from_hf(args.dataset_name, args.split, max_samples=args.max_samples, streaming=args.streaming)


def main():
    parser = argparse.ArgumentParser(description="Run a tiny no-model code/SWE environment eval rollout.")
    parser.add_argument("--dataset_name", default="R2E-Gym/R2E-Gym-Lite")
    parser.add_argument("--split", default="dev_10pr_v1")
    parser.add_argument("--dataset_path", default=None)
    parser.add_argument("--max_samples", type=int, default=3)
    parser.add_argument("--workspace_root", default="/tmp/verl_agent_code_swe_eval/workspaces")
    parser.add_argument("--repo_cache_dir", default=None)
    parser.add_argument("--patches_dir", default=None)
    parser.add_argument("--log_path", default=None)
    parser.add_argument("--gpu_count", default="unknown")
    parser.add_argument("--allow_network_clone", action="store_true")
    parser.add_argument("--default_test_command", default=None)
    parser.add_argument("--streaming", action="store_true")
    args = parser.parse_args()

    if args.log_path is None:
        gpu_bucket = {"1": "single_gpu", "2": "dual_gpu"}.get(str(args.gpu_count), f"multi_gpu_{args.gpu_count}" if str(args.gpu_count).isdigit() else "unknown_gpu")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.log_path = f"experiments/logs/code_swe/{gpu_bucket}/eval_{timestamp}.jsonl"
    if args.patches_dir is None:
        args.patches_dir = str(Path(args.log_path).with_suffix("") / "patches")

    tasks = load_tasks(args)
    runtime_config = RuntimeConfig(
        workspace_root=args.workspace_root,
        repo_cache_dir=args.repo_cache_dir,
        patches_dir=args.patches_dir,
        allow_network_clone=args.allow_network_clone,
        default_test_command=args.default_test_command,
        cleanup_workspaces=False,
        command_timeout=30,
        reward_timeout=180,
    )
    env = CodeSWEEnv(tasks=tasks, runtime_config=runtime_config, env_num=len(tasks), group_n=1, max_steps=3)

    Path(args.log_path).parent.mkdir(parents=True, exist_ok=True)
    with open(args.log_path, "w", encoding="utf-8") as log_f:
        obs, infos = env.reset()
        for idx, info in enumerate(infos):
            log_f.write(json.dumps({"step": 0, "env": idx, "action": "reset", "observation": obs[idx], "info": info}, ensure_ascii=False) + "\n")

        actions = [
            {
                "tool_name": "bash",
                "parameters": {
                    "cmd": "pwd && git status --short 2>/dev/null || true && find . -maxdepth 2 -type f | sed -n '1,80p'",
                },
            }
            for _ in tasks
        ]
        obs, rewards, dones, infos = env.step(actions)
        for idx, info in enumerate(infos):
            log_f.write(
                json.dumps(
                    {"step": 1, "env": idx, "action": actions[idx], "reward": rewards[idx], "done": dones[idx], "observation": obs[idx], "info": info},
                    ensure_ascii=False,
                )
                + "\n"
            )

        submit = [{"tool_name": "submit", "parameters": {}} for _ in tasks]
        obs, rewards, dones, infos = env.step(submit)
        for idx, info in enumerate(infos):
            log_f.write(
                json.dumps(
                    {"step": 2, "env": idx, "action": submit[idx], "reward": rewards[idx], "done": dones[idx], "observation": obs[idx], "info": info},
                    ensure_ascii=False,
                )
                + "\n"
            )

    env.close()
    success = sum(1 for info in infos if info.get("won"))
    print(f"Eval rollout finished. success={success}/{len(infos)} logs={args.log_path} patches={args.patches_dir}")


if __name__ == "__main__":
    main()
