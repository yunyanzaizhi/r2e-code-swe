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
    return load_tasks_from_hf(
        dataset_name=args.dataset_name,
        split=args.split,
        max_samples=args.max_samples,
        streaming=args.streaming,
    )


def main():
    parser = argparse.ArgumentParser(description="Smoke test the no-Docker code/SWE environment.")
    parser.add_argument("--dataset_name", default="R2E-Gym/R2E-Gym-Lite")
    parser.add_argument("--split", default="dev_10pr_v1")
    parser.add_argument("--dataset_path", default=None, help="Optional local JSONL task file.")
    parser.add_argument("--max_samples", type=int, default=1)
    parser.add_argument("--workspace_root", default="/tmp/verl_agent_code_swe_smoke")
    parser.add_argument("--repo_cache_dir", default=None)
    parser.add_argument("--allow_network_clone", action="store_true")
    parser.add_argument("--default_test_command", default=None)
    parser.add_argument("--log_path", default=None)
    parser.add_argument("--gpu_count", default="unknown")
    parser.add_argument("--streaming", action="store_true")
    args = parser.parse_args()

    if args.log_path is None:
        gpu_bucket = {"1": "single_gpu", "2": "dual_gpu"}.get(str(args.gpu_count), f"multi_gpu_{args.gpu_count}" if str(args.gpu_count).isdigit() else "unknown_gpu")
        args.log_path = f"experiments/logs/code_swe/{gpu_bucket}/smoke_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"

    tasks = load_tasks(args)
    runtime_config = RuntimeConfig(
        workspace_root=args.workspace_root,
        repo_cache_dir=args.repo_cache_dir,
        allow_network_clone=args.allow_network_clone,
        default_test_command=args.default_test_command,
        cleanup_workspaces=False,
        command_timeout=20,
        reward_timeout=120,
    )
    env = CodeSWEEnv(tasks=tasks, runtime_config=runtime_config, env_num=len(tasks), group_n=1, max_steps=2)

    Path(args.log_path).parent.mkdir(parents=True, exist_ok=True)
    with open(args.log_path, "w", encoding="utf-8") as log_f:
        obs, infos = env.reset()
        for i, info in enumerate(infos):
            log_f.write(json.dumps({"event": "reset", "env": i, "info": info, "observation": obs[i][:2000]}, ensure_ascii=False) + "\n")

        view_action = {"tool_name": "str_replace_editor", "parameters": {"command": "view", "path": "/testbed"}}
        obs, rewards, dones, infos = env.step([view_action for _ in tasks])
        for i, info in enumerate(infos):
            log_f.write(
                json.dumps(
                    {
                        "event": "view",
                        "env": i,
                        "reward": rewards[i],
                        "done": dones[i],
                        "info": info,
                        "observation": obs[i][:4000],
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

        submit_action = {"tool_name": "submit", "parameters": {}}
        obs, rewards, dones, infos = env.step([submit_action for _ in tasks])
        for i, info in enumerate(infos):
            log_f.write(
                json.dumps(
                    {
                        "event": "submit",
                        "env": i,
                        "reward": rewards[i],
                        "done": dones[i],
                        "info": info,
                        "observation": obs[i][:4000],
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    env.close()
    print(f"Smoke finished without crashing. Wrote log to {args.log_path}")
    for i, info in enumerate(infos):
        print(f"[{i}] task_id={info.get('task_id')} reward={rewards[i]} won={info.get('won')} fail_reason={info.get('fail_reason')}")


if __name__ == "__main__":
    main()
