import argparse
import json
from datetime import datetime
from pathlib import Path

from agent_system.environments.env_package.r2e_code_swe.envs import R2ECodeSWEEnv
from agent_system.environments.env_package.r2e_code_swe.runtime import R2ERuntimeConfig
from agent_system.environments.env_package.r2e_code_swe.tasks import load_r2e_tasks_from_hf


def main():
    parser = argparse.ArgumentParser(description="Run a tiny no-model R2E code/SWE rollout/eval.")
    parser.add_argument("--dataset_name", default="R2E-Gym/R2E-Gym-Lite")
    parser.add_argument("--split", default="dev_10pr_v1")
    parser.add_argument("--max_samples", type=int, default=3)
    parser.add_argument("--max_steps", type=int, default=5)
    parser.add_argument("--rollout_n", type=int, default=1)
    parser.add_argument("--streaming", action="store_true")
    parser.add_argument("--trajectory_dir", default="experiments/logs/r2e_code_swe/rollout_eval")
    parser.add_argument("--patches_dir", default=None)
    parser.add_argument("--log_path", default=None)
    args = parser.parse_args()

    if args.log_path is None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.log_path = f"experiments/logs/r2e_code_swe/rollout_eval_{args.max_samples}_{stamp}.jsonl"
    if args.patches_dir is None:
        args.patches_dir = str(Path(args.log_path).with_suffix("") / "patches")

    tasks = load_r2e_tasks_from_hf(args.dataset_name, args.split, args.max_samples, args.streaming, mode="eval")
    env = R2ECodeSWEEnv(
        tasks=tasks,
        runtime_config=R2ERuntimeConfig(trajectory_dir=args.trajectory_dir, patches_dir=args.patches_dir),
        env_num=len(tasks),
        group_n=args.rollout_n,
        max_steps=args.max_steps,
        is_train=False,
    )

    Path(args.log_path).parent.mkdir(parents=True, exist_ok=True)
    with open(args.log_path, "w", encoding="utf-8") as log_f:
        obs, infos = env.reset()
        for idx, info in enumerate(infos):
            log_f.write(json.dumps({"step": 0, "env": idx, "action": "reset", "observation": obs[idx], "info": info}, ensure_ascii=False) + "\n")

        inspect = [{"tool_name": "bash", "parameters": {"cmd": "pwd && git status --short | sed -n '1,80p' && find . -maxdepth 2 -type f | sed -n '1,80p'"}} for _ in env.current_tasks]
        obs, rewards, dones, infos = env.step(inspect)
        for idx, info in enumerate(infos):
            log_f.write(json.dumps({"step": 1, "env": idx, "action": inspect[idx], "reward": rewards[idx], "done": dones[idx], "observation": obs[idx], "info": info}, ensure_ascii=False) + "\n")

        submit = [{"tool_name": "submit", "parameters": {}} for _ in env.current_tasks]
        obs, rewards, dones, infos = env.step(submit)
        for idx, info in enumerate(infos):
            log_f.write(json.dumps({"step": 2, "env": idx, "action": submit[idx], "reward": rewards[idx], "done": dones[idx], "observation": obs[idx], "info": info}, ensure_ascii=False) + "\n")

    env.close()
    success = sum(1 for info in infos if info.get("won"))
    print(f"R2E rollout/eval finished. success={success}/{len(infos)} logs={args.log_path} patches={args.patches_dir}")


if __name__ == "__main__":
    main()
