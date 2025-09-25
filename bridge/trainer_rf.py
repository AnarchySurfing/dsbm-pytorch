import os, sys, warnings, time
import re
from collections import OrderedDict
from functools import partial

import hydra
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import glob

from .data import DBDSB_CacheLoader
from .sde import *
from .runners import *
from .runners.config_getters import get_model, get_optimizer, get_plotter, get_logger
from .runners.ema import EMAHelper
from .trainer_dbdsb import IPF_DBDSB

# from torchdyn.core import NeuralODE
from torchdiffeq import odeint


# IPF_RF类，继承自IPF_DBDSB，用于实现Rectified Flow (RF)算法
class IPF_RF(IPF_DBDSB):
    # 初始化函数
    def __init__(self, init_ds, final_ds, mean_final, var_final, args, accelerator=None, final_cond_model=None,
                 valid_ds=None, test_ds=None, output_dir='.'):
        # 调用父类的初始化方法
        super().__init__(init_ds, final_ds, mean_final, var_final, args, accelerator=accelerator, final_cond_model=final_cond_model,
                         valid_ds=valid_ds, test_ds=test_ds, output_dir=output_dir)
        # 初始化Langevin采样器，这里使用DBDSB_VE
        self.langevin = DBDSB_VE(0., self.num_steps, self.timesteps, self.shape_x, self.shape_y, first_coupling="ind", ot_sampler=self.args.ot_sampler)

    # 构建检查点相关设置
    def build_checkpoints(self):
        self.first_pass = True  # Load and use checkpointed networks during first pass
        self.ckpt_dir = os.path.join(self.output_dir, 'checkpoints')
        self.ckpt_prefixes = ["net_b", "sample_net_b", "optimizer_b"]
        self.cache_dir=os.path.join(self.output_dir, 'cache')
        # 如果是主进程，则创建检查点和缓存目录
        if self.accelerator.is_main_process:
            os.makedirs(self.ckpt_dir, exist_ok=True)
            os.makedirs(self.cache_dir, exist_ok=True)

        # 如果指定了从检查点运行
        if self.args.get('checkpoint_run', False):
            self.resume, self.checkpoint_it, self.checkpoint_pass, self.step = \
                True, self.args.checkpoint_it, self.args.checkpoint_pass, self.args.checkpoint_iter
            print(f"Resuming training at iter {self.checkpoint_it} {self.checkpoint_pass} step {self.step}")

            # 获取检查点文件的绝对路径
            self.checkpoint_b = hydra.utils.to_absolute_path(self.args.checkpoint_b)
            self.sample_checkpoint_b = hydra.utils.to_absolute_path(self.args.sample_checkpoint_b)
            self.optimizer_checkpoint_b = hydra.utils.to_absolute_path(self.args.optimizer_checkpoint_b)
            
        else:
            # 自动查找最新的检查点
            self.ckpt_dir_load = os.path.abspath(self.ckpt_dir)
            ckpt_dir_load_list = os.path.normpath(self.ckpt_dir_load).split(os.sep)
            if 'test' in ckpt_dir_load_list:
                self.ckpt_dir_load = os.path.join(*ckpt_dir_load_list[:ckpt_dir_load_list.index('test')], "checkpoints/")
            # 查找最后一个检查点
            self.resume, self.checkpoint_it, self.checkpoint_pass, self.step, ckpt_b_suffix = self.find_last_ckpt()

            if self.resume:
                # 如果不自动开始下一次迭代，并且当前是第一步，则回退到上一次迭代的最后一步
                if not self.args.autostart_next_it and self.step == 1 and not (self.checkpoint_it == 1 and self.checkpoint_pass == 'b'): 
                    self.checkpoint_pass, self.checkpoint_it = self.compute_prev_it(self.checkpoint_pass, self.checkpoint_it)
                    self.step = self.compute_max_iter(self.checkpoint_pass, self.checkpoint_it) + 1

                print(f"Resuming training at iter {self.checkpoint_it} {self.checkpoint_pass} step {self.step}")
                # 构建检查点文件路径
                self.checkpoint_b, self.sample_checkpoint_b, self.optimizer_checkpoint_b = [os.path.join(self.ckpt_dir_load, f"{ckpt_prefix}_{ckpt_b_suffix}.ckpt") for ckpt_prefix in self.ckpt_prefixes[:3]]

    # 构建模型
    def build_models(self, forward_or_backward=None):
        # running network
        # 获取后向网络模型
        net_b = get_model(self.args)

        # 如果是第一次传递并且需要从检查点恢复
        if self.first_pass and self.resume:
            if self.resume:
                try:
                    # 加载模型状态字典
                    net_b.load_state_dict(torch.load(self.checkpoint_b))
                except:
                    # 处理带有 "module." 前缀的状态字典（通常来自DataParallel或DistributedDataParallel）
                    state_dict = torch.load(self.checkpoint_b)
                    new_state_dict = OrderedDict()
                    for k, v in state_dict.items():
                        name = k.replace("module.", "")  # remove "module."
                        new_state_dict[name] = v
                    net_b.load_state_dict(new_state_dict)

        # 如果没有指定前向或后向，则准备后向网络并存储
        if forward_or_backward is None:
            net_b = self.accelerator.prepare(net_b)
            self.net = torch.nn.ModuleDict({'b': net_b})
        # 如果指定为后向，则准备后向网络并更新
        if forward_or_backward == 'b':
            net_b = self.accelerator.prepare(net_b)
            self.net.update({'b': net_b})

    # 构建EMA（Exponential Moving Average）模型
    def build_ema(self):
        # 如果启用EMA
        if self.args.ema:
            self.ema_helpers = {}

            # 如果是第一次传递并且需要从检查点恢复
            if self.first_pass and self.resume:
                # sample network
                # 获取用于采样的后向网络模型
                sample_net_b = get_model(self.args)

                if self.resume:
                    # 加载采样网络的状态字典
                    sample_net_b.load_state_dict(
                        torch.load(self.sample_checkpoint_b))
                    sample_net_b = sample_net_b.to(self.device)
                    # 更新EMA助手
                    self.update_ema('b')
                    # 注册采样网络到EMA助手
                    self.ema_helpers['b'].register(sample_net_b)

    # 训练函数
    def train(self):
        # 从检查点迭代次数开始，循环到总IPF迭代次数
        for n in range(self.checkpoint_it, self.n_ipf + 1):
            self.accelerator.print('RF iteration: ' + str(n) + '/' + str(self.n_ipf))
            # BACKWARD OPTIMISATION
            # 执行后向IPF迭代
            self.ipf_iter('b', n)

    # 构建优化器
    def build_optimizers(self, forward_or_backward=None):
        # 获取后向网络的优化器
        optimizer_b = get_optimizer(self.net['b'], self.args)

        # 如果是第一次传递并且需要从检查点恢复
        if self.first_pass and self.resume:
            if self.resume:
                # 加载优化器状态字典
                optimizer_b.load_state_dict(torch.load(self.optimizer_checkpoint_b))

        # 如果没有指定前向或后向，则创建优化器字典
        if forward_or_backward is None:
            self.optimizer = {'b': optimizer_b}
        # 如果指定为后向，则更新优化器字典
        if forward_or_backward == 'b':
            self.optimizer.update({'b': optimizer_b})

    # 查找最后一个检查点
    def find_last_ckpt(self):
        existing_ckpts_dict = {}
        # 遍历所有检查点前缀，查找已存在的检查点文件
        for ckpt_prefix in self.ckpt_prefixes:
            existing_ckpts = sorted(glob.glob(os.path.join(self.ckpt_dir_load, f"{ckpt_prefix}_**.ckpt")))
            existing_ckpts_dict[ckpt_prefix] = set([os.path.basename(existing_ckpt)[len(ckpt_prefix)+1:-5] for existing_ckpt in existing_ckpts])
        
        # 找到所有类型检查点都存在的公共检查点（net_b, sample_net_b, optimizer_b）
        existing_ckpts_b = sorted(list(existing_ckpts_dict["net_b"].intersection(existing_ckpts_dict["sample_net_b"], existing_ckpts_dict["optimizer_b"])), reverse=True)

        # 如果没有找到公共检查点，则从头开始
        if len(existing_ckpts_b) == 0:
            return False, 1, 'b', 1, None
        
        # 定义一个辅助函数，用于返回有效的检查点组合信息
        def return_valid_ckpt_combi(b_i, b_n):
            # Return is_valid, checkpoint_it, checkpoint_pass, checkpoint_step
            # 如果在b pass期间，但迭代未完成
            if (b_n == 1 and b_i != self.first_num_iter) or (b_n > 1 and b_i != self.num_iter):  # during b pass
                return True, b_n, 'b', b_i + 1
            else:  
                # 否则，进入下一次迭代的第一步
                return True, b_n + 1, 'b', 1

        # 遍历找到的后向检查点
        for existing_ckpt_b in existing_ckpts_b:
            # 解析检查点文件名中的迭代次数和步骤
            ckpt_b_n, ckpt_b_i = existing_ckpt_b.split("_")
            ckpt_b_n, ckpt_b_i = int(ckpt_b_n), int(ckpt_b_i)
            
            # 获取有效的检查点信息
            is_valid, checkpoint_it, checkpoint_pass, checkpoint_step = return_valid_ckpt_combi(ckpt_b_i, ckpt_b_n)
            if is_valid:
                break

        # 如果没有找到有效的检查点
        if not is_valid:
            return False, 1, 'b', 1, None
        else:
            # 返回恢复训练所需的信息
            return True, checkpoint_it, checkpoint_pass, checkpoint_step, existing_ckpt_b

    # 应用网络模型
    def apply_net(self, x, y, t, net, fb, return_scale=False):
        # 前向传播
        out = net.forward(x, y, t)

        if return_scale:
            return out, 1
        else:
            return out

    # 计算上一次迭代
    def compute_prev_it(self, forward_or_backward, n):
        assert forward_or_backward == 'b'
        prev_direction = 'b'
        prev_n = n - 1
        return prev_direction, prev_n

    # 计算下一次迭代
    def compute_next_it(self, forward_or_backward, n):
        assert forward_or_backward == 'b'
        next_direction = 'b'
        next_n = n+1
        return next_direction, next_n