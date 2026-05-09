import argparse
import json
from datetime import datetime
from pathlib import Path

from agent_system.environments.env_package.r2e_code_swe.envs import R2ECodeSWEEnv
from agent_system.environments.env_package.r2e_code_swe.runtime import R2ERuntimeConfig
from agent_system.environments.env_package.r2e_code_swe.tasks import load_r2e_tasks_from_hf


def main():
    parser = argparse.ArgumentParser(description="Smoke test the R2E-only Docker code/SWE environment.")
    parser.add_argument("--dataset_name", default="R2E-Gym/R2E-Gym-Lite")
    parser.add_argument("--split", default="dev_10pr_v1")
    parser.add_argument("--max_samples", type=int, default=1)
    parser.add_argument("--max_steps", type=int, default=2)
    parser.add_argument("--streaming", action="store_true")
    parser.add_argument("--r2e_repo_root", default="/home/caiting/R2E-Gym")
    parser.add_argument("--trajectory_dir", default="experiments/logs/r2e_code_swe/smoke")
    parser.add_argument("--patches_dir", default=None)
    parser.add_argument("--command_timeout", type=int, default=60)
    parser.add_argument("--reward_timeout", type=int, default=300)
    parser.add_argument("--max_output_chars", type=int, default=12000)
    parser.add_argument("--log_path", default=None)
    args = parser.parse_args()

    if args.log_path is None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.log_path = f"experiments/logs/r2e_code_swe/smoke_{args.max_samples}_{stamp}.jsonl"

    tasks = load_r2e_tasks_from_hf(
        dataset_name=args.dataset_name,
        split=args.split,
        max_samples=args.max_samples,
        streaming=args.streaming,
        mode="eval",
    )
    runtime_config = R2ERuntimeConfig(
        r2e_repo_root=args.r2e_repo_root,
        command_timeout=args.command_timeout,
        reward_timeout=args.reward_timeout,
        max_output_chars=args.max_output_chars,
        trajectory_dir=args.trajectory_dir,
        patches_dir=args.patches_dir,
    )
    env = R2ECodeSWEEnv(
        tasks=tasks,
        runtime_config=runtime_config,
        env_num=len(tasks),
        group_n=1,
        max_steps=args.max_steps,
        is_train=False,
    )

    Path(args.log_path).parent.mkdir(parents=True, exist_ok=True)
    with open(args.log_path, "w", encoding="utf-8") as log_f:
        obs, infos = env.reset()
        for idx, info in enumerate(infos):
            log_f.write(json.dumps({"event": "reset", "env": idx, "observation": obs[idx], "info": info}, ensure_ascii=False) + "\n")

        bash_actions = [{"tool_name": "bash", "parameters": {"cmd": "pwd && ls -la | sed -n '1,60p'"}} for _ in tasks]
        obs, rewards, dones, infos = env.step(bash_actions)
        for idx, info in enumerate(infos):
            log_f.write(json.dumps({"event": "bash", "env": idx, "reward": rewards[idx], "done": dones[idx], "observation": obs[idx], "info": info}, ensure_ascii=False) + "\n")

        submit_actions = [{"tool_name": "submit", "parameters": {}} for _ in tasks]
        obs, rewards, dones, infos = env.step(submit_actions)
        for idx, info in enumerate(infos):
            log_f.write(json.dumps({"event": "submit", "env": idx, "reward": rewards[idx], "done": dones[idx], "observation": obs[idx], "info": info}, ensure_ascii=False) + "\n")

    env.close()
    print(f"R2E smoke finished. logs={args.log_path}")
    for idx, info in enumerate(infos):
        print(f"[{idx}] task_id={info.get('task_id')} image={info.get('docker_image')} reward={rewards[idx]} won={info.get('won')} fail_reason={info.get('fail_reason')} patch={info.get('patch_path')}")


if __name__ == "__main__":
    main()
