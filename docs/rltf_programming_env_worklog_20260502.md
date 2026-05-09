# RLTF Reward 与 ProgrammingEnv 批量环境适配工作文档

- 生成时间：2026-05-02
- 远程项目：`/home/caiting/verl-agent-exp`
- 说明：远程目录当前不是 git 仓库，因此本文中的“更改时间”取自服务器文件 mtime，并结合当前文件内容与训练日志整理。

## 1. 总体目标

本轮工作围绕两个目标展开：

1. 将 RLTF 论文中的 coarse reward + adaptive reward 接入 `ProgrammingEnv`，让代码修复环境不再只有 0/1 奖励。
2. 建立更复杂的双卡 50-step 启动脚本，并让 `ProgrammingEnv` 支持 `env_num/group_n` 批量 rollout，避免训练/验证阶段 batch size 与 observation size 不匹配。

## 2. 文件更改概览

| 文件 | 更改时间 | 主要变更 |
| --- | --- | --- |
| `agent_system/environments/env_package/programming/envs.py` | 2026-05-01 23:01:53 | 接入 RLTF reward；改用当前 Python 启动 pytest；支持批量 env；增加 workspace 隔离与校验 |
| `agent_system/environments/env_manager.py` | 2026-05-01 13:20:07 | 构建 ProgrammingEnv 时传入 `reward_mode` |
| `tests/test_programming_rltf_reward.py` | 2026-05-01 23:01:20 | 新增 RLTF reward、pytest 调用、timeout、批量 env、workspace 隔离等回归测试 |
| `scripts/run_quixbugs_last3_lora.sh` | 2026-05-01 13:20:07 | 原单卡脚本启用 `+env.programming.reward_mode=rltf_scalar` |
| `scripts/run_quixbugs_last3_rltf_2gpu_50.sh` | 2026-05-01 22:41:54 | 新增双卡 50-step 更强参数启动脚本 |

## 3. 详细变更

### 3.1 `agent_system/environments/env_package/programming/envs.py`

#### 更改时间

- 2026-05-01 23:01:53

#### 关键代码变更

1. 新增依赖：

```python
import shutil
import tempfile
```

2. `ProgrammingEnv.__init__` 增加 reward 与批量配置：

```python
class ProgrammingEnv(gym.Env):
    def __init__(self, data_root, max_steps=10, reward_mode="binary", env_num=1, group_n=1):
        self.data_root = data_root
        self.max_steps = max_steps
        self.reward_mode = reward_mode
        if not isinstance(env_num, int) or not isinstance(group_n, int) or env_num < 1 or group_n < 1:
            raise ValueError("env_num and group_n must be positive integers")
        self.env_num = env_num
        self.group_n = group_n
        self.num_processes = env_num * group_n
        self._workspace_root = tempfile.mkdtemp(prefix="programming_env_")
```

3. `reset()` 从单条 observation 改成批量 observation，并为每个 slot 拷贝独立 workspace：

```python
selected_tasks = self.tasks[:self.env_num]
self.cur_files = [
    task
    for task in selected_tasks
    for _ in range(self.group_n)
]

for index, cur_file in enumerate(self.cur_files):
    workspace = os.path.join(self._workspace_root, str(index))
    shutil.copytree(self.data_root, workspace)
    file_path = os.path.join(workspace, "python_programs", cur_file)
```

4. `step()` 增加 action 数量校验，并对每个 batch slot 独立执行 pytest：

```python
if len(actions) != self.num_processes:
    raise ValueError(
        f"Expected {self.num_processes} actions, got {len(actions)}"
    )

for index, new_code in enumerate(actions):
    cur_file = self.cur_files[index]
    workspace = self.workspaces[index]
    file_path = os.path.join(workspace, "python_programs", cur_file)
```

5. pytest 调用从依赖 shell PATH 改为当前 Python 解释器，并显式设置 `PYTHONPATH`：

```python
python_paths = [
    workspace,
    os.path.join(workspace, "python_testcases"),
    env.get("PYTHONPATH", ""),
]
result = subprocess.run(
    [sys.executable, "-m", "pytest", "-q", test_path],
    cwd=workspace,
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
    timeout=10,
    text=True,
    env=env,
)
```

6. 接入 RLTF coarse + adaptive reward：

```python
pass_ratio = n_pass / total if total else (1.0 if success else 0.0)
adaptive = -0.3 + 1.3 * pass_ratio
```

```python
def _compute_reward(self, feedback):
    if self.reward_mode == "binary":
        return 1.0 if feedback["feedback_category"] == "pass" else 0.0
    if self.reward_mode == "rltf_scalar":
        return feedback["reward_components"]["final"]
    raise ValueError(f"Unsupported programming reward_mode: {self.reward_mode}")
```

```python
def _coarse_reward(self, category):
    return {
        "pass": 1.0,
        "failure": -0.3,
        "error": -0.6,
        "timeout": -0.6,
        "syntax_error": -1.0,
    }.get(category, -0.6)
```

7. 增加 workspace 清理：

```python
def close(self):
    if os.path.exists(self._workspace_root):
        shutil.rmtree(self._workspace_root)
```

8. factory 传入 `env_num/group_n/reward_mode`：

```python
return ProgrammingEnv(
    data_root=data_root,
    max_steps=max_steps,
    reward_mode=reward_mode,
    env_num=env_num,
    group_n=group_n,
)
```

#### 为什么这么改

- 原环境只返回 1 条 observation，双卡脚本中 `val_batch_size=2` 时，rollout loop 报错：

```text
AssertionError: gen_batch size 2 does not match obs size 1
```

- 初始批量修复只复制列表长度，但不同 action 共用同一个 `python_programs/*.py`，会造成跨样本污染。
- 因此最终采用每个 batch slot 独立 workspace 的方式，既保留原来的 ProgrammingEnv 逻辑，又满足 `env_num/group_n` 的批量语义。

#### 更改前后对比

| 项目 | 更改前 | 更改后 |
| --- | --- | --- |
| reward | 只有 pass=1/fail=0 | 支持 `binary` 与 `rltf_scalar` 两种模式 |
| pytest 启动 | `pytest -q ...`，依赖 PATH | `sys.executable -m pytest`，使用当前虚拟环境 |
| PYTHONPATH | 未显式设置，临时测试可能导入失败 | 显式包含 workspace 与 `python_testcases` |
| timeout/error | 普通异常容易混在失败 reward 中 | timeout=-0.6；pytest 启动失败 fail-fast |
| reset 输出 | 固定 1 条 obs | `env_num * group_n` 条 obs |
| step 输入 | 默认只看 `actions[0]` | 逐 action 独立评测 |
| 文件隔离 | 所有 action 写同一个源码目录 | 每个 batch slot 独立临时 workspace |

---

### 3.2 `agent_system/environments/env_manager.py`

#### 更改时间

- 2026-05-01 13:20:07

#### 关键代码变更

在 Programming 环境构建时，将配置里的 reward mode 传入环境：

```python
_envs = build_programming_envs(
    seed=config.env.seed,
    env_num=config.data.train_batch_size,
    group_n=group_n,
    data_root=config.env.programming.data_root,
    max_steps=config.env.max_steps,
    reward_mode=getattr(config.env.programming, "reward_mode", "binary"),
    resources_per_worker=resources_per_worker,
    is_train=True,
)
```

验证环境同样传入：

```python
_val_envs = build_programming_envs(
    seed=config.env.seed + 1000,
    env_num=config.data.val_batch_size,
    group_n=1,
    data_root=config.env.programming.data_root,
    max_steps=config.env.max_steps,
    reward_mode=getattr(config.env.programming, "reward_mode", "binary"),
    resources_per_worker=resources_per_worker,
    is_train=False,
)
```

#### 为什么这么改

脚本里虽然可以写：

```bash
+env.programming.reward_mode=rltf_scalar
```

但如果 `env_manager.py` 不读取并传入这个字段，`ProgrammingEnv` 会继续使用默认 binary reward，RLTF reward 不会真正生效。

#### 更改前后对比

| 项目 | 更改前 | 更改后 |
| --- | --- | --- |
| 脚本配置 `reward_mode` | 配了也不会传到 env | train/val env 都会接收 |
| 默认兼容 | 固定旧 reward | 未配置时仍默认 `binary` |
| RLTF 开关 | 无法从脚本切换 | 可用 `+env.programming.reward_mode=rltf_scalar` 切换 |

---

### 3.3 `tests/test_programming_rltf_reward.py`

#### 更改时间

- 2026-05-01 23:01:20

#### 关键代码变更

新增 14 个测试，覆盖以下行为：

```python
def test_rltf_scalar_reward_uses_partial_test_pass_rate(tmp_path): ...
def test_rltf_scalar_reward_penalizes_syntax_errors(tmp_path): ...
def test_binary_reward_mode_preserves_existing_pass_fail_behavior(tmp_path): ...
def test_build_programming_envs_passes_reward_mode(tmp_path): ...
def test_rltf_reward_uses_current_python_for_pytest_when_path_lacks_pytest(tmp_path, monkeypatch): ...
def test_rltf_reward_classifies_pytest_timeout(tmp_path, monkeypatch): ...
def test_pytest_start_oserror_fails_fast(tmp_path, monkeypatch): ...
def test_programming_env_reset_honors_env_num_and_group_n(tmp_path): ...
def test_programming_env_step_handles_batched_actions(tmp_path): ...
def test_programming_env_selects_env_num_tasks_and_repeats_by_group_n(tmp_path): ...
def test_programming_env_batched_non_empty_actions_are_isolated(tmp_path): ...
def test_programming_env_step_rejects_wrong_action_count(tmp_path): ...
def test_programming_env_rejects_non_positive_env_sizes(tmp_path): ...
def test_programming_env_close_removes_workspace_root(tmp_path): ...
```

其中批量环境关键测试包括：

```python
assert [info["file"] for info in infos] == [
    "a_bug.py",
    "a_bug.py",
    "b_bug.py",
    "b_bug.py",
]
```

以及独立 action 评测：

```python
_, rewards, dones, infos = env.step([
    "def fix_a():\n    return 1\n",
    "def fix_b():\n    return 0\n",
])

assert rewards == [1.0, -0.3]
assert dones == [True, False]
assert [info["file"] for info in infos] == ["a_bug.py", "b_bug.py"]
```

#### 为什么这么改

- RLTF reward 会直接影响训练信号，必须有测试保证 pass/fail/syntax/timeout 等类别映射正确。
- 后续双卡启动时出现过两个实际错误：
  - `ppo_mini_batch_size 0 should be larger than 0 after normalization`
  - `gen_batch size 2 does not match obs size 1`
- 因此测试中增加了 batch 长度、任务重复、action 数量校验和 workspace 清理的覆盖。

#### 更改前后对比

| 项目 | 更改前 | 更改后 |
| --- | --- | --- |
| reward 测试 | 无专门测试 | 覆盖 partial pass、syntax、binary 保持兼容 |
| pytest 调用测试 | 无 | 覆盖 PATH 缺 pytest 时仍可运行 |
| timeout 测试 | 无 | timeout 分类为 `timeout` 且 reward=-0.6 |
| 批量环境测试 | 无 | 覆盖 `env_num/group_n`、非空 action、隔离、错误 action 数量 |
| 清理测试 | 无 | 覆盖 `close()` 删除临时 workspace |

验证结果：

```text
.venv/bin/python -m pytest -q tests/test_programming_rltf_reward.py
14 passed
```

---

### 3.4 `scripts/run_quixbugs_last3_lora.sh`

#### 更改时间

- 2026-05-01 13:20:07

#### 关键代码变更

原单卡 recent-window k=3 LoRA 脚本增加 RLTF reward 开关：

```bash
+env.programming.reward_mode=rltf_scalar \
```

上下文：

```bash
env.env_name=Programming \
env.seed=0 \
env.max_steps=10 \
env.history_length=$history_length \
env.rollout.n=$group_size \
+env.programming.data_root=/home/caiting/verl-agent-exp/third_party/QuixBugs \
+env.programming.memory_type=simple \
+env.programming.reward_mode=rltf_scalar \
```

#### 为什么这么改

该脚本是最先用于验证 RLTF reward 接入的启动脚本。加上 `reward_mode=rltf_scalar` 后，可以直接在原有训练链路中验证新 reward 是否被环境使用。

#### 更改前后对比

| 项目 | 更改前 | 更改后 |
| --- | --- | --- |
| reward | 默认 binary | 使用 RLTF scalar |
| 启动规模 | 单卡，原 k=3/history 设置 | 保持不变，仅替换 reward 信号 |
| 作用 | baseline/原始验证 | 验证 RLTF 接入是否可跑通 |

相关日志现象：

- 配置中出现：`reward_mode: rltf_scalar`
- 成功样本仍能给出 `[text][score] 1.0`
- 原先 timeout/invalid 等情况不再只有 0/1，而会进入 coarse/adaptive 分类。

---

### 3.5 `scripts/run_quixbugs_last3_rltf_2gpu_50.sh`

#### 更改时间

- 2026-05-01 22:41:54

#### 关键代码变更

新增双卡 50-step 脚本，核心配置如下：

```bash
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1}
export VLLM_ATTENTION_BACKEND=FLASH_ATTN
export VLLM_USE_V1=0
export WANDB_MODE=disabled
export TORCHDYNAMO_DISABLE=1
```

训练规模：

```bash
train_data_size=2
val_data_size=2
group_size=2
history_length=3
total_training_steps=50
```

模型与 LoRA 参数：

```bash
actor_rollout_ref.model.lora_rank=16 \
actor_rollout_ref.model.lora_alpha=32 \
actor_rollout_ref.actor.optim.lr=1e-6 \
actor_rollout_ref.model.use_remove_padding=True \
```

双卡相关修正：

```bash
actor_rollout_ref.actor.ppo_mini_batch_size=2 \
trainer.n_gpus_per_node=2 \
```

rollout 与 reward：

```bash
env.max_steps=15 \
env.history_length=$history_length \
env.rollout.n=$group_size \
+env.programming.reward_mode=rltf_scalar \
```

日志和 checkpoint 隔离：

```bash
RUN_ID=${RUN_ID:-$(date +%Y%m%d_%H%M%S)}
trainer.default_local_dir=$PWD/checkpoints/programming_history_exp/recent_window_k3_rltf_2gpu_50_$RUN_ID \
trainer.val_before_train=True 2>&1 | tee experiments/logs/quixbugs_recent_window_k3_rltf_2gpu_50_$RUN_ID.log
```

#### 为什么这么改

用户目标是“使用双卡、把 steps/epochs 从原来的短跑拉高到 50，并提高部分参数，在更复杂的代码环境中继续验证当前 reward 函数”。因此脚本从原单卡短跑扩展为：

- 双卡：`trainer.n_gpus_per_node=2`
- 更长训练：`trainer.total_training_steps=50`、`trainer.total_epochs=50`
- 更大验证/训练 batch：`train_data_size=2`、`val_data_size=2`
- 更复杂 rollout：`env.max_steps=15`、`env.rollout.n=2`
- 更强 LoRA：`lora_rank=16`、`lora_alpha=32`

#### 更改前后对比

| 项目 | 原脚本/初版 | 当前脚本 |
| --- | --- | --- |
| GPU | 1 卡 | 2 卡 |
| 训练长度 | 短 run/约 16 个 batch 量级 | 目标 50 steps |
| env max steps | 10 | 15 |
| LoRA | 较小设置 | rank=16, alpha=32 |
| batch | 单样本为主 | train=2, val=2 |
| reward | binary 或未显式传入 | `rltf_scalar` |
| 日志目录 | 固定文件名，容易覆盖 | 每次 `RUN_ID` 独立日志/checkpoint |

#### 调试过的问题

1. 双卡 mini-batch 归一化错误：

```text
AssertionError: ppo_mini_batch_size 0 should be larger than 0 after normalization
```

原因是双卡下：

```text
ppo_mini_batch_size=1 -> 1 // 2 = 0
```

修正为：

```bash
actor_rollout_ref.actor.ppo_mini_batch_size=2
```

2. 验证阶段 batch 与 env obs 不匹配：

```text
AssertionError: gen_batch size 2 does not match obs size 1
```

根因是旧 `ProgrammingEnv.reset()` 只返回 1 条 obs。后续通过 `env_num/group_n` 批量支持和 per-slot workspace 隔离修复。

#### 当前启动状态

最近一次启动：

```text
RUN_ID=20260501_230309
TRAIN_LOG=/home/caiting/verl-agent-exp/experiments/logs/quixbugs_recent_window_k3_rltf_2gpu_50_20260501_230309.log
LAUNCH_LOG=/home/caiting/verl-agent-exp/experiments/logs/quixbugs_recent_window_k3_rltf_2gpu_50_launch_20260501_230309.log
```

启动后确认进程存在：

```text
python3 -m verl.trainer.main_ppo ... trainer.n_gpus_per_node=2 ... +env.programming.reward_mode=rltf_scalar ...
```

早期日志显示 Ray 已启动：

```text
Started a local Ray instance.
```

## 4. Reward 逻辑前后对比

### 更改前

环境成功时：

```python
reward = 1.0 if success else 0.0
```

特点：

- 只知道是否全部通过。
- 1 个测试通过、1 个失败和完全崩溃都可能被压成失败。
- 对代码修复这种多测试反馈任务来说，训练信号过粗。

### 更改后

RLTF scalar 模式下：

```text
pass:         1.0
failure:      -0.3 + 1.3 * pass_ratio
error:        -0.6
timeout:      -0.6
syntax_error: -1.0
```

举例：

| 测试结果 | pass_ratio | reward |
| --- | --- | --- |
| 全部通过 | 1.0 | 1.0 |
| 1 pass / 1 fail | 0.5 | 0.35 |
| 0 pass / 有 failure | 0.0 | -0.3 |
| runtime/error | - | -0.6 |
| syntax error | - | -1.0 |

这让模型能区分“部分修好”“完全错误”“语法错误”“运行超时”，比原始 binary reward 更适合代码工程环境。

## 5. 验证记录

已执行并通过：

```text
.venv/bin/python -m pytest -q tests/test_programming_rltf_reward.py
14 passed
```

已执行并通过：

```text
.venv/bin/python -m py_compile agent_system/environments/env_package/programming/envs.py
bash -n scripts/run_quixbugs_last3_rltf_2gpu_50.sh
```

代码审查结果：

- 修复前审查发现：共享源码目录导致 batch slot 污染、action 数量未校验。
- 修复后审查：无 CRITICAL/HIGH 阻塞问题。

## 6. 总结

本轮改动没有重写整个 Programming 环境，而是围绕当前训练链路做了最小闭环：

1. reward 从 binary 扩展为 RLTF coarse + adaptive。
2. `env_manager.py` 支持从脚本配置切换 reward mode。
3. `ProgrammingEnv` 支持双卡验证所需的批量 observation/action。
4. 每个 batch slot 独立 workspace，避免代码文件互相污染。
5. 新增 14 个测试覆盖 reward、pytest、timeout、batch、隔离与清理。
6. 新增双卡 50-step 脚本，用于更复杂配置下验证 RLTF reward。

# 2026.5.2

## 新增agent_system/memory/programming_summary.py
用于提取结构化摘要

## agent_system/environments/env_manager.py 
1.保存修改前副本为env_manager.py.bak_summary;
2.新增```from agent_system.memory.programming_summary import ProgrammingSummaryMemory```;
3.然后找到 ProgrammingEnvironmentManager.__init__ 这一段：
```
class ProgrammingEnvironmentManager(EnvironmentManagerBase):
    def __init__(self, envs, projection_f, config):
        memory_type = getattr(config.env.programming, "memory_type", "simple")
        if memory_type == "structured_summary":
            self.memory = StructuredSummaryMemory()
            self.use_structured_summary = True
        else:
            self.memory = SimpleMemory()
            self.use_structured_summary = False
        super().__init__(envs, projection_f, config)
```
改成：
```
class ProgrammingEnvironmentManager(EnvironmentManagerBase):
    def __init__(self, envs, projection_f, config):
        memory_type = getattr(config.env.programming, "memory_type", "simple")
        if memory_type == "structured_summary":
            self.memory = ProgrammingSummaryMemory()
            self.use_structured_summary = True
        else:
            self.memory = SimpleMemory()
            self.use_structured_summary = False
        super().__init__(envs, projection_f, config)
```

## 新增scripts/run_quixbugs_structured_summary_rltf_2gpu_50.sh

