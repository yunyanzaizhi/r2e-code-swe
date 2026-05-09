# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Note that we don't combine the main with ray_trainer as ray_trainer is used by other main.
"""
# 注意：不将main函数与ray_trainer合并，因为ray_trainer被其他主程序使用

import os

import hydra
import ray
from omegaconf import OmegaConf

from verl.trainer.ppo.ray_trainer import RayPPOTrainer
from verl.trainer.ppo.reward import load_reward_manager
from verl.trainer.constants_ppo import get_ppo_ray_runtime_env


# 使用Hydra进行配置管理，指定配置路径和配置文件
@hydra.main(config_path="config", config_name="ppo_trainer", version_base=None)
def main(config):
    """主函数入口，Hydra会自动加载配置文件并作为config参数传入"""
    run_ppo(config)  # 调用PPO训练函数


def run_ppo(config) -> None:
    """运行PPO训练的主要函数"""
    
    # 检查Ray是否已初始化
    if not ray.is_initialized():
        # 如果Ray未初始化，则进行初始化
        
        # 从配置中获取Ray初始化参数
        ray_init_kwargs = config.get("ray_init", {})
        
        # 获取PPO训练的环境配置
        default_runtime_env = get_ppo_ray_runtime_env()
        
        # 合并默认环境配置和用户配置
        runtime_env_kwargs = ray_init_kwargs.get("runtime_env", {})
        runtime_env = OmegaConf.merge(default_runtime_env, runtime_env_kwargs)
        
        # 创建最终的Ray初始化参数
        ray_init_kwargs = OmegaConf.create({**ray_init_kwargs, "runtime_env": runtime_env})
        
        print(f"ray init kwargs: {ray_init_kwargs}")
        
        # 初始化Ray分布式计算框架
        ray.init(**OmegaConf.to_container(ray_init_kwargs))

    # 创建远程任务运行器
    runner = TaskRunner.remote()
    
    # 远程执行run方法
    ray.get(runner.run.remote(config))


# 定义远程任务运行器类
@ray.remote(num_cpus=1)  # 指定分配1个CPU核心
class TaskRunner:
    """远程任务运行器，避免主任务被调度到头节点"""
    
    def run(self, config):
        """主要的训练执行逻辑"""
        
        # 打印配置信息
        from pprint import pprint
        from omegaconf import OmegaConf
        from verl.utils.fs import copy_to_local

        # 解析并打印配置
        pprint(OmegaConf.to_container(config, resolve=True))
        OmegaConf.resolve(config)  # 解析配置中的变量引用

        # 下载模型检查点（从HDFS到本地）
        local_path = copy_to_local(
            config.actor_rollout_ref.model.path,  # 模型路径
            use_shm=config.actor_rollout_ref.model.get("use_shm", False)  # 是否使用共享内存
        )

        # 创建训练和验证环境
        from agent_system.environments import make_envs
        envs, val_envs = make_envs(config)

        # 初始化tokenizer和processor
        from verl.utils import hf_processor, hf_tokenizer

        trust_remote_code = config.data.get("trust_remote_code", False)
        
        # 加载tokenizer（用于文本编码）
        tokenizer = hf_tokenizer(local_path, trust_remote_code=trust_remote_code)
        
        # 加载processor（用于多模态数据处理，如图像等）
        processor = hf_processor(
            local_path, 
            trust_remote_code=trust_remote_code, 
            use_fast=True
        )  # 对于多模态LLM，可能为None

        # 检查vllm版本兼容性
        if config.actor_rollout_ref.rollout.name in ["vllm"]:
            from verl.utils.vllm_utils import is_version_ge

            # 如果使用LoRA，需要vllm 0.7.3+版本
            if config.actor_rollout_ref.model.get("lora_rank", 0) > 0:
                if not is_version_ge(pkg="vllm", minver="0.7.3"):
                    raise NotImplementedError("PPO LoRA is not supported before vllm 0.7.3")

        # 根据策略类型选择Worker类
        if config.actor_rollout_ref.actor.strategy in ["fsdp", "fsdp2"]:
            # FSDP策略：全分片数据并行
            assert config.critic.strategy in ["fsdp", "fsdp2"]
            from verl.single_controller.ray import RayWorkerGroup
            from verl.workers.fsdp_workers import ActorRolloutRefWorker, AsyncActorRolloutRefWorker, CriticWorker

            # 根据模式选择同步或异步的Actor Worker
            actor_rollout_cls = AsyncActorRolloutRefWorker if config.actor_rollout_ref.rollout.mode == "async" else ActorRolloutRefWorker
            ray_worker_group_cls = RayWorkerGroup

        elif config.actor_rollout_ref.actor.strategy == "megatron":
            # Megatron策略：另一种分布式训练框架
            assert config.actor_rollout_ref.actor.strategy == config.critic.strategy
            from verl.single_controller.ray.megatron import NVMegatronRayWorkerGroup
            from verl.workers.megatron_workers import ActorRolloutRefWorker, CriticWorker

            actor_rollout_cls = ActorRolloutRefWorker
            ray_worker_group_cls = NVMegatronRayWorkerGroup

        else:
            raise NotImplementedError

        # 导入资源池管理器
        from verl.trainer.ppo.ray_trainer import ResourcePoolManager, Role

        # 定义角色到Worker类的映射
        role_worker_mapping = {
            Role.ActorRollout: ray.remote(actor_rollout_cls),  # Actor Worker
            Role.Critic: ray.remote(CriticWorker),  # Critic Worker
        }

        # 定义全局资源池
        global_pool_id = "global_pool"
        
        # 资源池规格：每个节点分配多少个GPU
        resource_pool_spec = {
            global_pool_id: [config.trainer.n_gpus_per_node] * config.trainer.nnodes,
        }
        
        # 角色到资源池的映射
        mapping = {
            Role.ActorRollout: global_pool_id,  # Actor使用全局资源池
            Role.Critic: global_pool_id,  # Critic使用全局资源池
        }

        # 配置奖励模型Worker
        if config.reward_model.enable:
            if config.reward_model.strategy in ["fsdp", "fsdp2"]:
                from verl.workers.fsdp_workers import RewardModelWorker
            elif config.reward_model.strategy == "megatron":
                from verl.workers.megatron_workers import RewardModelWorker
            else:
                raise NotImplementedError
                
            # 添加奖励模型Worker
            role_worker_mapping[Role.RewardModel] = ray.remote(RewardModelWorker)
            mapping[Role.RewardModel] = global_pool_id

        # 如果需要参考策略（用于KL散度计算）
        if config.algorithm.use_kl_in_reward or config.actor_rollout_ref.actor.use_kl_loss:
            role_worker_mapping[Role.RefPolicy] = ray.remote(ActorRolloutRefWorker)
            mapping[Role.RefPolicy] = global_pool_id

        # 配置奖励管理器
        reward_manager_name = config.reward_model.get("reward_manager", "episode")
        if reward_manager_name == 'episode':
            from agent_system.reward_manager import EpisodeRewardManager
            reward_manager_cls = EpisodeRewardManager
        else:
            raise NotImplementedError

        # 创建训练和验证的奖励函数
        reward_fn = reward_manager_cls(
            tokenizer=tokenizer, 
            num_examine=0,  # 训练时不详细检查
            normalize_by_length=False
        )
        
        val_reward_fn = reward_manager_cls(
            tokenizer=tokenizer, 
            num_examine=1,  # 验证时详细检查
            normalize_by_length=False
        )

        # 创建资源池管理器
        resource_pool_manager = ResourcePoolManager(
            resource_pool_spec=resource_pool_spec, 
            mapping=mapping
        )

        # 检查配置一致性
        assert config.actor_rollout_ref.rollout.n == 1, "In verl, actor_rollout_ref.rollout.n>1 is for GRPO. In verl+env, we keep n=1, and achieve GRPO by env.rollout.n"

        # 创建轨迹收集器（用于多轮交互）
        from agent_system.multi_turn_rollout import TrajectoryCollector
        traj_collector = TrajectoryCollector(
            config=config, 
            tokenizer=tokenizer, 
            processor=processor
        )

        # 创建数据加载相关组件
        from verl.utils.dataset.rl_dataset import collate_fn

        # 创建训练和验证数据集
        train_dataset = create_rl_dataset(
            config.data.train_files, 
            config.data, 
            tokenizer, 
            processor
        )
        
        val_dataset = create_rl_dataset(
            config.data.val_files, 
            config.data, 
            tokenizer, 
            processor
        )
        
        # 创建数据采样器
        train_sampler = create_rl_sampler(config.data, train_dataset)
        
        # 创建PPO Trainer
        trainer = RayPPOTrainer(
            config=config,  # 训练配置
            tokenizer=tokenizer,  # 分词器
            processor=processor,  # 多模态处理器
            role_worker_mapping=role_worker_mapping,  # 角色到Worker的映射
            resource_pool_manager=resource_pool_manager,  # 资源管理器
            ray_worker_group_cls=ray_worker_group_cls,  # Worker组类
            reward_fn=reward_fn,  # 训练奖励函数
            val_reward_fn=val_reward_fn,  # 验证奖励函数
            train_dataset=train_dataset,  # 训练数据集
            val_dataset=val_dataset,  # 验证数据集
            collate_fn=collate_fn,  # 数据批处理函数
            train_sampler=train_sampler,  # 训练采样器
            device_name=config.trainer.device,  # 设备名称
            traj_collector=traj_collector,  # 轨迹收集器
            envs=envs,  # 训练环境
            val_envs=val_envs,  # 验证环境
        )
        
        # 初始化Worker
        trainer.init_workers()
        
        # 开始训练
        trainer.fit()


def create_rl_dataset(data_paths, data_config, tokenizer, processor):
    """创建RL训练数据集
    
    Args:
        data_paths: 数据文件路径列表
        data_config: 数据配置
        tokenizer: 分词器
        processor: 多模态处理器
        
    Returns:
        dataset: 创建的数据集
    """
    from torch.utils.data import Dataset
    from verl.utils.dataset.rl_dataset import RLHFDataset

    # 检查是否有自定义数据集类
    if "custom_cls" in data_config and data_config.custom_cls.get("path", None) is not None:
        from verl.utils.import_utils import load_extern_type
        
        # 动态加载自定义数据集类
        dataset_cls = load_extern_type(
            data_config.custom_cls.path, 
            data_config.custom_cls.name
        )
        
        # 检查类是否继承自Dataset
        if not issubclass(dataset_cls, Dataset):
            raise TypeError(f"The custom dataset class '{data_config.custom_cls.name}' from '{data_config.custom_cls.path}' must inherit from torch.utils.data.Dataset")
    else:
        # 使用默认数据集类
        dataset_cls = RLHFDataset
        
    print(f"Using dataset class: {dataset_cls.__name__}")

    # 创建数据集实例
    dataset = dataset_cls(
        data_files=data_paths,  # 数据文件路径
        tokenizer=tokenizer,  # 分词器
        processor=processor,  # 多模态处理器
        config=data_config,  # 配置
    )

    return dataset


def create_rl_sampler(data_config, dataset):
    """创建数据采样器
    
    Args:
        data_config: 数据配置
        dataset: 数据集
        
    Returns:
        sampler: 采样器
    """
    import torch
    from torch.utils.data import RandomSampler, SequentialSampler

    # 根据配置选择随机采样或顺序采样
    if data_config.shuffle:
        # 创建随机数生成器用于可复现的随机采样
        train_dataloader_generator = torch.Generator()
        train_dataloader_generator.manual_seed(data_config.get("seed", 1))
        sampler = RandomSampler(
            data_source=dataset, 
            generator=train_dataloader_generator
        )
    else:
        # 顺序采样
        sampler = SequentialSampler(data_source=dataset)

    return sampler


# 程序入口
if __name__ == "__main__":
    main()  # 启动主函数