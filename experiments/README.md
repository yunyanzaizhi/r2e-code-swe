# Task 2: 多轮交互历史管理策略对比实验

基于 [verl-agent](https://github.com/langfengQ/verl-agent) 框架，使用 GRPO 强化学习算法训练 LLM 智能体，研究不同历史管理策略对多轮交互任务的影响。

## 研究目标

比较 3 种历史管理策略在 Sokoban 和 ALFWorld 环境下的表现：

| 策略 | 说明 |
|------|------|
| **Full History** | 保留全部交互历史（`history_length=100`，远大于 `max_steps`） |
| **Recent Window** | 仅保留最近 K 步历史（K=3 / K=5），滑动窗口截断 |
| **Structured Summary** | 将历史压缩为结构化摘要（进度、关键事件、循环检测等） |

## 项目结构

```
experiments/
├── run_sokoban_full_history.sh         # Sokoban Full History 实验
├── run_sokoban_recent_window.sh        # Sokoban Recent Window 实验（K 通过参数传入）
├── run_sokoban_structured_summary.sh   # Sokoban Structured Summary 实验
├── run_alfworld_full_history.sh        # ALFWorld Full History 实验
├── analyze_results.py                  # 日志解析与结果分析
├── README.md                           # 本文件
└── logs/                               # 从远程服务器同步的训练日志
    ├── lab-server/                      # 10.14.18.28 (2× V100S-32GB)
    │   ├── full_history.log
    │   ├── recent_window_k3.log
    │   ├── recent_window_k5.log
    │   ├── structured_summary.log
    │   ├── alfworld_full_history.log
    │   └── alfworld_full_history_ray_worker.log
    └── server-22/                       # 10.14.18.22 (1× V100S-32GB)
        └── structured_summary.log

agent_system/
├── memory/
│   ├── memory.py                       # SimpleMemory（Full History / Recent Window）
│   └── structured_summary.py           # StructuredSummaryMemory（新增）
├── environments/
│   ├── env_manager.py                  # SokobanEnvironmentManager / AlfWorldEnvironmentManager
│   └── prompts/                        # 各环境 prompt 模板
└── reward_manager/
    └── episode.py                      # Episode reward 计算

verl/workers/
└── fsdp_workers.py                     # V100 兼容性补丁（已修改）
```

## 对框架的修改

### 1. 新增文件

- **`agent_system/memory/structured_summary.py`**  
  实现 `StructuredSummaryMemory` 类，继承 `BaseMemory`。维护结构化摘要而非原始历史，包含进度跟踪、关键事件记录、循环检测、无效动作统计等功能。

- **`experiments/` 下的 4 个实验脚本 + 1 个分析脚本**

### 2. 修改文件

- **`verl/workers/fsdp_workers.py`**  
  - 新增 `_get_attn_implementation()` 函数：自动检测 GPU 算力，V100（sm < 80）使用 `sdpa` 替代 `flash_attention_2`
  - Actor CPUOffload 单 GPU 修复：`world_size == 1` 时允许 CPUOffload，解决单卡 OOM
  - 模型加载处统一调用 `_get_attn_implementation()`

- **`agent_system/environments/env_manager.py`**  
  - `SokobanEnvironmentManager` 新增 `memory_type` 配置项，支持选择 `structured_summary` 或默认的 `SimpleMemory`

## 环境要求

- Python 3.10+
- PyTorch 2.6+
- vLLM 0.8.5
- GPU: NVIDIA V100S-32GB（已适配） 或 A100/H100（原生支持）
- 模型: `Qwen/Qwen2.5-1.5B-Instruct`

## 运行方式

### 前置准备

```bash
# 克隆代码
git clone https://github.com/langfengQ/verl-agent.git
cd verl-agent

# 创建虚拟环境并安装依赖
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
pip install vllm==0.8.5

# 设置环境变量
export PYTHONPATH="$(pwd)/agent_system/environments/env_package/sokoban:$(pwd)/agent_system/environments/env_package/alfworld:$(pwd)/agent_system/environments/env_package/gym_cards:${PYTHONPATH}"
```

### Sokoban 实验

```bash
# Full History（保留全部交互历史）
bash experiments/run_sokoban_full_history.sh vllm

# Recent Window K=3（保留最近 3 步）
bash experiments/run_sokoban_recent_window.sh vllm 3

# Recent Window K=5（保留最近 5 步）
bash experiments/run_sokoban_recent_window.sh vllm 5

# Structured Summary（结构化摘要）
bash experiments/run_sokoban_structured_summary.sh vllm
```

### ALFWorld 实验

```bash
# Full History
bash experiments/run_alfworld_full_history.sh vllm
```

> ALFWorld 数据会自动下载到 `~/.cache/alfworld/`，也可提前手动放置。

### 结果分析

```bash
# 解析日志并生成对比表格（需要先将日志放到 experiments/logs/ 下）
python3 experiments/analyze_results.py
```

## 核心训练参数

| 参数 | 值 | 说明 |
|------|-----|------|
| 模型 | Qwen2.5-1.5B-Instruct | 基座模型 |
| 算法 | GRPO | Group Relative Policy Optimization |
| 学习率 | 1e-6 | Actor 学习率 |
| train_batch_size | 16 | 训练 batch |
| val_batch_size | 32 | 验证 batch |
| group_size | 2 | GRPO 采样组大小 |
| max_prompt_length | 4096 | 最大 prompt 长度 |
| max_response_length | 128 | 最大生成长度 |
| total_epochs | 50 | 总训练步数 |
| test_freq | 5 | 每 5 步验证一次 |
| KL loss | low_var_kl, coef=0.01 | KL 正则化 |
| invalid_action_penalty | 0.1 | 无效动作惩罚 |

### V100 适配参数

| 参数 | 值 | 说明 |
|------|-----|------|
| ppo_micro_batch_size_per_gpu | 1 | 减少显存占用 |
| param_offload | True | 参数 CPU 卸载 |
| optimizer_offload | True | 优化器 CPU 卸载 |
| mixed_precision | fp16 | 半精度训练（V100 不支持 bf16） |
| gpu_memory_utilization | 0.4 | vLLM KV cache 限制 |
| tensor_model_parallel_size | 2 | 2 卡张量并行 |
| enforce_eager | True | 禁用 CUDA Graph（兼容性） |
| gradient_checkpointing | True | 梯度检查点 |

### 环境参数

| 参数 | Sokoban | ALFWorld |
|------|---------|----------|
| max_steps | 15 | 50 |
| env_name | Sokoban | alfworld/AlfredTWEnv |
| dim_room | [6,6] | - |
| num_boxes | 1 | - |
| eval_dataset | - | eval_in_distribution |

## 日志说明

训练日志中的关键指标：

- `step:N` — 当前训练步
- `episode/success_rate` — 训练集成功率
- `val/success_rate` — 验证集成功率
- `episode/length/mean` — 平均 episode 长度
- `prompt_length/mean` — 平均 prompt token 数（反映历史长度）
- `timing_s/step` — 每步耗时（秒）
- `actor/pg_loss` — 策略梯度损失
- `actor/grad_norm` — 梯度范数
- `perf/max_memory_allocated_gb` — GPU 峰值显存

日志以 10% 概率采样输出 `[text][prompt]`、`[text][response]`、`[text][score]` 详细交互内容。
