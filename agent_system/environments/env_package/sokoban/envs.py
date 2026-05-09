# Copyright 2025 Nanyang Technological University (NTU), Singapore
# and the verl-agent (GiGPO) team.
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

import ray
import gym
from agent_system.environments.env_package.sokoban.sokoban import SokobanEnv
import numpy as np

#基于Ray框架实现的Sokoban（仓库番）游戏多进程环境

class SokobanWorker:
    """
    Ray远程Actor，替代传统的worker函数。
    每个Actor持有自己独立的SokobanEnv实例。
    
    作用：
    1. 将环境实例封装在Ray Actor中
    2. 提供远程调用接口
    3. 隔离不同环境的状态
    """
    
    def __init__(self, mode, env_kwargs):
        """在此worker中初始化Sokoban环境"""
        # 创建独立的环境实例
        # 每个Worker有自己的SokobanEnv，状态完全隔离
        self.env = SokobanEnv(mode, **env_kwargs)
    
    def step(self, action):
        """在环境中执行一步动作"""
        # 调用底层SokobanEnv的step方法
        # 返回: 观察、奖励、完成标志、额外信息
        obs, reward, done, info = self.env.step(action)
        return obs, reward, done, info
    
    def reset(self, seed_for_reset):
        """用给定的种子重置环境"""
        # 重置环境到初始状态
        # seed_for_reset: 随机种子，确保可重现性
        obs, info = self.env.reset(seed=seed_for_reset)
        return obs, info
    
    def render(self, mode_for_render):
        """渲染环境"""
        # 调用底层环境的render方法
        # mode_for_render: 渲染模式，如'rgb_array'、'human'等
        rendered = self.env.render(mode=mode_for_render)
        return rendered


class SokobanMultiProcessEnv(gym.Env):
    """
    基于Ray的Sokoban环境包装器。
    每个Ray Actor创建一个独立的SokobanEnv实例。
    主进程与Ray Actor通信以收集step/reset结果。
    
    继承自gym.Env，保持标准接口
    """

    def __init__(self,
                 seed=0,  # 随机种子
                 env_num=1,  # 不同环境的数量
                 group_n=1,  # 每组中相同环境的数量（用于GRPO和GiGPO）
                 mode='rgb_array',  # 环境模式
                 resources_per_worker={"num_cpus": 0.1},  # 每个worker的资源分配
                 is_train=True,  # 是否为训练模式
                 env_kwargs=None):  # SokobanEnv的初始化参数
        """
        参数说明:
        - env_num: 不同环境的数量
        - group_n: 每组中相同环境的数量（用于GRPO和GiGPO）
        - env_kwargs: 初始化SokobanEnv的参数字典
        - seed: 随机种子，用于可重现性
        """
        super().__init__()  # 调用父类gym.Env的初始化

        # 如果Ray尚未初始化，则初始化Ray
        if not ray.is_initialized():
            ray.init()  # 初始化Ray分布式计算框架

        # 保存配置参数
        self.is_train = is_train  # 训练/评估模式
        self.group_n = group_n  # 组内重复数
        self.env_num = env_num  # 不同环境数
        self.num_processes = env_num * group_n  # 总进程数
        self.mode = mode  # 环境模式
        np.random.seed(seed)  # 设置numpy随机种子

        # 如果没有提供env_kwargs，使用空字典
        if env_kwargs is None:
            env_kwargs = {}

        # 创建Ray远程Actor，替代传统多进程
        # ray.remote装饰器将类转换为Ray Actor
        # resources_per_worker指定每个Actor的资源需求
        env_worker = ray.remote(**resources_per_worker)(SokobanWorker)
        
        # 创建多个Worker实例
        self.workers = []  # 存储所有Worker的引用
        for i in range(self.num_processes):
            # 创建Worker实例
            # remote()是异步调用，但这里立即执行
            worker = env_worker.remote(self.mode, env_kwargs)
            self.workers.append(worker)

    def step(self, actions):
        """
        并行执行一步动作。
        
        参数:
        :param actions: list[int]，动作列表，长度必须等于self.num_processes
        
        返回:
        :return: obs_list, reward_list, done_list, info_list
                每个都是长度为self.num_processes的列表
        """
        # 验证动作数量匹配
        assert len(actions) == self.num_processes

        # 向所有Worker发送step命令
        futures = []  # 存储未来结果
        for worker, action in zip(self.workers, actions):
            # 远程调用worker的step方法
            # .remote()表示异步调用，返回Future对象
            future = worker.step.remote(action)
            futures.append(future)

        # 收集所有结果
        # ray.get()阻塞等待所有Future完成
        results = ray.get(futures)
        
        # 解析结果
        obs_list, reward_list, done_list, info_list = [], [], [], []
        for obs, reward, done, info in results:
            obs_list.append(obs)
            reward_list.append(reward)
            done_list.append(done)
            info_list.append(info)

        return obs_list, reward_list, done_list, info_list

    def reset(self):
        """
        并行重置所有环境。
        
        返回:
        :return: obs_list和info_list，每个环境的初始观察
        """
        # 为每个不同的环境生成随机种子
        if self.is_train:
            # 训练模式：使用较小的种子范围
            seeds = np.random.randint(0, 2**16 - 1, size=self.env_num)
        else:
            # 评估模式：使用较大的种子范围，避免与训练集重叠
            seeds = np.random.randint(2**16, 2**32 - 1, size=self.env_num)

        # 为组内重复的环境复制种子
        # 示例: seeds=[1,2,3], group_n=2 → [1,1,2,2,3,3]
        seeds = np.repeat(seeds, self.group_n)
        seeds = seeds.tolist()  # 转换为列表

        # 向所有Worker发送reset命令
        futures = []
        for i, worker in enumerate(self.workers):
            # 传递对应的种子
            future = worker.reset.remote(seeds[i])
            futures.append(future)

        # 收集结果
        results = ray.get(futures)
        obs_list = []
        info_list = []
        for obs, info in results:
            obs_list.append(obs)
            info_list.append(info)
        return obs_list, info_list

    def render(self, mode='rgb_array', env_idx=None):
        """
        请求Ray Actor环境进行渲染。
        
        参数:
        :param mode: 渲染模式
        :param env_idx: 指定环境索引，如果为None则渲染所有环境
        
        返回:
        :return: 单个渲染结果或渲染结果列表
        """
        if env_idx is not None:
            # 只渲染指定环境
            future = self.workers[env_idx].render.remote(mode)
            return ray.get(future)
        else:
            # 渲染所有环境
            futures = []
            for worker in self.workers:
                future = worker.render.remote(mode)
                futures.append(future)
            results = ray.get(futures)
            return results

    def close(self):
        """关闭所有Ray Actor"""
        # 终止所有Ray Actor
        for worker in self.workers:
            ray.kill(worker)

    def __del__(self):
        """析构函数，确保资源被正确释放"""
        self.close()


def build_sokoban_envs(
        seed=0,
        env_num=1,
        group_n=1,
        mode='rgb_array',
        resources_per_worker={"num_cpus": 0.1},
        is_train=True,
        env_kwargs=None):
    """
    工厂函数，创建Sokoban多进程环境。
    
    作用:
    1. 统一的环境创建接口
    2. 便于配置管理
    3. 遵循gym环境创建模式
    """
    return SokobanMultiProcessEnv(
        seed, 
        env_num, 
        group_n, 
        mode, 
        resources_per_worker, 
        is_train, 
        env_kwargs=env_kwargs
    )