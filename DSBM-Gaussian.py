import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from tqdm import tqdm
from functools import partial
import copy
import ot as pot

from typing import List, Optional, Tuple
import hydra
import pytorch_lightning as pl
from omegaconf import DictConfig

device = 'cpu'
dataset_size = 10000
test_dataset_size = 10000
lr = 1e-4
batch_size = 128


class MLP(nn.Module):
    # 简单多层感知机（MLP）实现
    # 说明：此类用于构建基础前馈神经网络，支持自定义每层宽度和激活函数。
    # 输入：input_dim - 输入特征维度；layer_widths - 各隐藏层与输出层的宽度列表；
    # activate_final - 是否对最后一层也应用激活；activation_fn - 激活函数。
    def __init__(self, input_dim, layer_widths=[100, 100, 2], activate_final=False, activation_fn=F.tanh):
        super(MLP, self).__init__()
        layers = []
        prev_width = input_dim
        for layer_width in layer_widths:
            layers.append(torch.nn.Linear(prev_width, layer_width))
            prev_width = layer_width
        self.input_dim = input_dim
        self.layer_widths = layer_widths
        self.layers = nn.ModuleList(layers)
        self.activate_final = activate_final
        self.activation_fn = activation_fn

    def forward(self, x):
        for i, layer in enumerate(self.layers[:-1]):
            x = self.activation_fn(layer(x))
        x = self.layers[-1](x)
        if self.activate_final:
            x = self.activation_fn(x)
        return x


class ScoreNetwork(nn.Module):
    # Score 网络封装
    # 说明：将时间 t 与输入 x 级联后传入 MLP，用于学习带时间依赖的评分函数/向量场。
    def __init__(self, input_dim, layer_widths=[100, 100, 2], activate_final=False, activation_fn=F.tanh):
        super().__init__()
        self.net = MLP(input_dim, layer_widths=layer_widths, activate_final=activate_final, activation_fn=activation_fn)

    def forward(self, x_input, t):
        inputs = torch.cat([x_input, t], dim=1)
        return self.net(inputs)


# Original DSB
class DSB(nn.Module):
    # Original DSB (Discrete Schrödinger Bridge) 类
    # 说明：实现了用于原始 DSB 方法的数据生成与基于学习的时间步进采样。
    # 主要方法：generate_new_dataset_and_train_tuple - 生成训练样本 (z_t, t, target)
    #           sample_sde - 基于学习到的向量场，用欧拉法采样得到粒子轨迹。
    def __init__(self, net_fwd=None, net_bwd=None, num_steps=20, sig=0):
        super().__init__()
        self.net_fwd = net_fwd  # 前向网络，用于从 t=0 到 t=1 的向量场预测
        self.net_bwd = net_bwd  # 反向网络，用于从 t=1 到 t=0 的向量场预测
        self.net_dict = {"f": self.net_fwd, "b": self.net_bwd}  # 网络字典，便于根据方向选择网络
        # self.optimizer_dict = {"f": torch.optim.Adam(self.net_fwd.parameters(), lr=lr), "b": torch.optim.Adam(self.net_bwd.parameters(), lr=lr)}
        self.N = num_steps  # 时间步数，用于离散化时间区间 [0,1]
        self.sig = sig  # 噪声强度，用于添加随机噪声

    @torch.no_grad()
    def generate_new_dataset_and_train_tuple(self, x_pairs=None, fb='', first_it=False):
        # 生成新的数据集和训练元组，用于监督学习
        # 参数：
        #   x_pairs: 输入的配对样本，形状为 [batch_size, 2, dim]
        #   fb: 方向，'f' 表示前向 (forward)，'b' 表示反向 (backward)
        #   first_it: 是否为第一次迭代，如果是，则使用简单的噪声扰动
        # 返回：
        #   z_t: 中间状态 z_t
        #   tlist: 对应的时间 t
        #   target: 目标向量，用于训练网络
        assert fb in ['f', 'b']  # 确保方向参数正确

        if fb == 'f':
            prev_fb = 'b'  # 前向时，使用反向网络作为上一方向
            zstart = x_pairs[:, 1]  # 前向从目标分布开始
        else:
            prev_fb = 'f'  # 反向时，使用前向网络作为上一方向
            zstart = x_pairs[:, 0]  # 反向从源分布开始

        N = self.N  # 时间步数
        dt = 1. / N  # 时间步长
        traj = []  # 存储轨迹的列表 to store the trajectory
        signal = []  # 存储信号（目标向量）的列表
        tlist = []  # 存储时间点的列表
        z = zstart.detach().clone()  # 初始化 z 为起始点
        batchsize = zstart.shape[0]  # 批次大小
        dim = zstart.shape[1]  # 数据维度

        ts = np.arange(N) / N  # 时间点数组，从 0 到 1
        tl = np.arange(1, N + 1) / N  # 对应的下一个时间点
        if prev_fb == 'b':
            ts = 1 - ts  # 如果上一方向是反向，则反转时间
            tl = 1 - tl

        if first_it:  # 第一次迭代，使用简单的噪声扰动
            assert prev_fb == 'f'  # 确保上一方向是前向
            for i in range(N):
                t = torch.ones((batchsize, 1), device=device) * ts[i]  # 当前时间 t
                dz = self.sig * torch.randn_like(z) * np.sqrt(dt)  # 噪声增量
                z = z + dz  # 更新 z
                tlist.append(torch.ones((batchsize, 1), device=device) * tl[i])  # 添加时间点
                traj.append(z.detach().clone())  # 添加到轨迹
                signal.append(-dz)  # 信号为负噪声
        else:  # 非第一次迭代，使用网络预测
            for i in range(N):
                t = torch.ones((batchsize, 1), device=device) * ts[i]  # 当前时间 t
                pred = self.net_dict[prev_fb](z, t)  # 使用上一方向的网络预测向量
                z = z.detach().clone() + pred  # 更新 z
                dz = self.sig * torch.randn_like(z) * np.sqrt(dt)  # 噪声增量
                z = z + dz  # 添加噪声
                tlist.append(torch.ones((batchsize, 1), device=device) * tl[i])  # 添加时间点
                traj.append(z.detach().clone())  # 添加到轨迹
                signal.append(- self.net_dict[prev_fb](z, t) - dz)  # 信号为负预测和负噪声

        z_t = torch.stack(traj)  # 将轨迹堆叠成张量
        tlist = torch.stack(tlist)  # 将时间点堆叠
        target = torch.stack(signal)  # 将信号堆叠

        # 随机选择一个时间步，用于训练
        randint = torch.randint(N, (1, batchsize, 1), device=device)
        tlist = torch.gather(tlist, 0, randint).squeeze(0)  # 选择时间点
        z_t = torch.gather(z_t, 0, randint.expand(1, batchsize, dim)).squeeze(0)  # 选择 z_t
        target = torch.gather(target, 0, randint.expand(1, batchsize, dim)).squeeze(0)  # 选择目标
        return z_t, tlist, target

    @torch.no_grad()
    def sample_sde(self, zstart=None, fb='', first_it=False, N=None):
        # 使用学习到的向量场进行 SDE 采样，生成粒子轨迹
        # 参数：
        #   zstart: 起始点
        #   fb: 方向，'f' 或 'b'
        #   first_it: 是否为第一次迭代
        #   N: 时间步数，如果为 None 则使用 self.N
        # 返回：
        #   traj: 轨迹列表，包含 N+1 个点
        assert fb in ['f', 'b']  # 确保方向正确

        ### NOTE: Use Euler method to sample from the learned flow
        N = self.N if N is None else N  # 使用默认或指定时间步数
        dt = 1. / N  # 时间步长
        traj = []  # 存储轨迹
        z = zstart.detach().clone()  # 初始化 z
        batchsize = z.shape[0]  # 批次大小

        traj.append(z.detach().clone())  # 添加起始点
        ts = np.arange(N) / N  # 时间点
        if fb == 'b':
            ts = 1 - ts  # 反向时反转时间
        for i in range(N):
            t = torch.ones((batchsize, 1), device=device) * ts[i]  # 当前时间
            pred = self.net_dict[fb](z, t)  # 使用当前方向网络预测
            z = z.detach().clone() + pred  # 更新 z
            z = z + self.sig * torch.randn_like(z) * np.sqrt(dt)  # 添加噪声
            traj.append(z.detach().clone())  # 添加到轨迹
        return traj


def train_dsb_ipf(dsb_ipf, x_pairs, batch_size, inner_iters, fb='', first_it=False, **kwargs):
    # 训练循环辅助函数（用于原始 DSB 的内循环）
    # 说明：为指定方向（f/b）生成训练数据并执行多步优化，返回训练后的模型与损失曲线。
    # 断言方向参数必须是 'f' 或 'b'
    assert fb in ['f', 'b']
    # 设置模型的方向属性
    dsb_ipf.fb = fb
    optimizer = torch.optim.Adam(dsb_ipf.net_dict[fb].parameters(), lr=lr)
    # optimizer = dsb_ipf.optimizer_dict[fb]
    # 初始化损失曲线列表，用于记录训练过程中的损失
    loss_curve = []

    # 生成训练数据元组 (z_ts, ts, targets)，用于监督学习
    z_ts, ts, targets = dsb_ipf.generate_new_dataset_and_train_tuple(x_pairs=x_pairs, fb=fb, first_it=first_it)
    # 创建数据加载器，用于批量加载训练数据
    dl = iter(DataLoader(TensorDataset(z_ts, ts, targets), batch_size=batch_size, shuffle=True, pin_memory=False,
                         drop_last=True))

    # 开始内层训练循环，迭代 inner_iters 次
    for i in tqdm(range(inner_iters)):
        # 尝试从数据加载器中获取下一个批次的数据
        try:
            z_t, t, target = next(dl)
        # 如果数据加载器耗尽，则重新生成训练数据并创建新的数据加载器
        except StopIteration:
            z_ts, ts, targets = dsb_ipf.generate_new_dataset_and_train_tuple(x_pairs=x_pairs, fb=fb, first_it=first_it)
            dl = iter(
                DataLoader(TensorDataset(z_ts, ts, targets), batch_size=batch_size, shuffle=True, pin_memory=False,
                           drop_last=True))
            z_t, t, target = next(dl)

        optimizer.zero_grad()
        pred = dsb_ipf.net_dict[fb](z_t, t)
        loss = (target - pred).view(pred.shape[0], -1).abs().pow(2).sum(dim=1)
        # 对损失取平均值
        loss = loss.mean()
        loss.backward()

        # 检查损失是否为 NaN，如果是则抛出错误并停止训练
        if torch.isnan(loss).any():
            raise ValueError("Loss is nan")
            break

        optimizer.step()
        # 将对数损失值添加到损失曲线列表中，用于存储损失曲线
        loss_curve.append(np.log(loss.item()))  ## to store the loss curve

    # 返回训练后的模型和损失曲线
    return dsb_ipf, loss_curve


# DSBM
class DSBM(nn.Module):
    # DSBM（Discrete Schrödinger Bridge Model）类
    # 说明：实现 DSBM 特有的数据构造方法与基于步长为 1/N 的欧拉随机微分方程采样。
    # 主要方法：get_train_tuple - 给定配对样本生成用于监督学习的 (z_t, t, target)
    #           generate_new_dataset - 根据上一个模型或初始耦合生成新的配对样本
    #           sample_sde - 基于学习到的速度场对粒子进行前向或反向采样。
    def __init__(self, net_fwd=None, net_bwd=None, num_steps=1000, sig=0, eps=1e-3, first_coupling="ref"):
        super().__init__()
        self.net_fwd = net_fwd
        self.net_bwd = net_bwd
        self.net_dict = {"f": self.net_fwd, "b": self.net_bwd}
        # self.optimizer_dict = {"f": torch.optim.Adam(self.net_fwd.parameters(), lr=lr), "b": torch.optim.Adam(self.net_bwd.parameters(), lr=lr)}
        self.N = num_steps
        self.sig = sig
        self.eps = eps
        self.first_coupling = first_coupling

    @torch.no_grad()
    def get_train_tuple(self, x_pairs=None, fb='', **kwargs):
        # 生成用于监督学习的训练元组 (z_t, t, target)
        # 参数：
        #   x_pairs: 输入的配对样本，形状为 [batch_size, 2, dim]
        #   fb: 方向，'f' 表示前向 (forward)，'b' 表示反向 (backward)
        # 返回：
        #   z_t: 中间状态 z_t
        #   t: 时间 t
        #   target: 目标向量，用于训练网络
        z0, z1 = x_pairs[:, 0], x_pairs[:, 1]
        t = torch.rand((z1.shape[0], 1), device=device) * (1 - 2 * self.eps) + self.eps
        z_t = t * z1 + (1. - t) * z0
        z = torch.randn_like(z_t)
        z_t = z_t + self.sig * torch.sqrt(t * (1. - t)) * z
        if fb == 'f':
            # z1 - z_t / (1-t)
            target = z1 - z0
            target = target - self.sig * torch.sqrt(t / (1. - t)) * z
        else:
            # z0 - z_t / t
            target = - (z1 - z0)
            target = target - self.sig * torch.sqrt((1. - t) / t) * z
        return z_t, t, target

    @torch.no_grad()
    def generate_new_dataset(self, x_pairs, prev_model=None, fb='', first_it=False):
        # generate_new_dataset 支持首轮不同耦合策略（ref, ind），以及基于前一个迭代模型生成下一个配对。
        assert fb in ['f', 'b']

        if prev_model is None:
            assert first_it
            assert fb == 'b'
            zstart = x_pairs[:, 0]
            if self.first_coupling == "ref":
                # First coupling is x_0, x_0 perturbed
                zend = zstart + torch.randn_like(zstart) * self.sig
            elif self.first_coupling == "ind":
                zend = x_pairs[:, 1].clone()
                zend = zend[torch.randperm(len(zend))]
            else:
                raise NotImplementedError
            z0, z1 = zstart, zend
        else:
            assert not first_it
            if prev_model.fb == 'f':
                zstart = x_pairs[:, 0]
            else:
                zstart = x_pairs[:, 1]
            zend = prev_model.sample_sde(zstart=zstart, fb=prev_model.fb)[-1]
            if prev_model.fb == 'f':
                z0, z1 = zstart, zend
            else:
                z0, z1 = zend, zstart
        return z0, z1

    @torch.no_grad()
    def sample_sde(self, zstart=None, N=None, fb='', first_it=False):
        # sample_sde 返回粒子的轨迹列表（长度 N+1）。
        assert fb in ['f', 'b']
        ### NOTE: Use Euler method to sample from the learned flow
        if N is None:
            N = self.N
        dt = 1. / N
        traj = []  # to store the trajectory
        z = zstart.detach().clone()
        batchsize = z.shape[0]

        traj.append(z.detach().clone())
        ts = np.arange(N) / N
        if fb == 'b':
            ts = 1 - ts
        for i in range(N):
            t = torch.ones((batchsize, 1), device=device) * ts[i]
            pred = self.net_dict[fb](z, t)
            z = z.detach().clone() + pred * dt
            z = z + self.sig * torch.randn_like(z) * np.sqrt(dt)
            traj.append(z.detach().clone())

        return traj


def train_dsbm(dsbm_ipf, x_pairs, batch_size, inner_iters, prev_model=None, fb='', first_it=False):
    # DSBM 的训练函数封装
    # 说明：在每个 inner_iters 内循环中从 generate_new_dataset 采样配对，构建训练样本并优化当前方向的网络。
    assert fb in ['f', 'b']
    dsbm_ipf.fb = fb
    optimizer = torch.optim.Adam(dsbm_ipf.net_dict[fb].parameters(), lr=lr)
    # optimizer = dsbm_ipf.optimizer_dict[fb]
    loss_curve = []

    dl = iter(DataLoader(
        TensorDataset(*dsbm_ipf.generate_new_dataset(x_pairs, prev_model=prev_model, fb=fb, first_it=first_it)),
        batch_size=batch_size, shuffle=True, pin_memory=False, drop_last=True))

    for i in tqdm(range(inner_iters)):
        try:
            z0, z1 = next(dl)
        except StopIteration:
            dl = iter(DataLoader(
                TensorDataset(*dsbm_ipf.generate_new_dataset(x_pairs, prev_model=prev_model, fb=fb, first_it=first_it)),
                batch_size=batch_size, shuffle=True, pin_memory=False, drop_last=True))
            z0, z1 = next(dl)

        z_pairs = torch.stack([z0, z1], dim=1)
        z_t, t, target = dsbm_ipf.get_train_tuple(z_pairs, fb=fb, first_it=first_it)
        optimizer.zero_grad()
        pred = dsbm_ipf.net_dict[fb](z_t, t)
        loss = (target - pred).view(pred.shape[0], -1).abs().pow(2).sum(dim=1)
        loss = loss.mean()
        loss.backward()

        if torch.isnan(loss).any():
            raise ValueError("Loss is nan")
            break

        optimizer.step()
        loss_curve.append(np.log(loss.item()))  ## to store the loss curve

    return dsbm_ipf, loss_curve


# SB-CFM
class SBCFM(nn.Module):
    # SB-CFM（Schrödinger Bridge with Conditional Flow Matching）类
    # 说明：使用 OT 采样器产生耦合（z0, z1），再构造带噪声的中间状态供网络学习。
    # 主要方法：get_train_tuple - 返回 (z_t, t, target)
    def __init__(self, net=None, num_steps=1000, sig=0, eps=1e-3):
        super().__init__()
        self.net = net
        self.optimizer = torch.optim.Adam(self.net.parameters(),
                                          lr=lr)  # torch.optim.AdamW(self.net.parameters(), lr=lr, weight_decay=weight_decay)
        self.N = num_steps
        self.sig = sig
        self.eps = eps
        from bridge.sde.optimal_transport import OTPlanSampler
        self.ot_sampler = OTPlanSampler(method="sinkhorn", reg=2 * sig ** 2)

    @torch.no_grad()
    def get_train_tuple(self, x_pairs=None, **kwargs):
        x0, x1 = x_pairs[:, 0], x_pairs[:, 1]
        z0, z1 = self.ot_sampler.sample_plan(x0, x1)

        t = torch.rand((z1.shape[0], 1), device=device) * (1 - 2 * self.eps) + self.eps
        z_t = t * z1 + (1. - t) * z0
        z = torch.randn_like(z_t)
        z_t = z_t + self.sig * torch.sqrt(t * (1. - t)) * z
        target = z1 - z0
        target = target - self.sig * (torch.sqrt(t) / torch.sqrt(1. - t) - 0.5 / torch.sqrt(t * (1. - t))) * z
        return z_t, t, target

    @torch.no_grad()
    def generate_new_dataset(self, x_pairs, **kwargs):
        # 说明：generate_new_dataset 返回初始配对 (x0, permuted x1)，用于形成独立耦合样本
        return x_pairs[:, 0], x_pairs[torch.randperm(len(x_pairs)), 1]

    @torch.no_grad()
    def sample_ode(self, zstart=None, N=None, fb='', first_it=False):
        # 说明：sample_ode 使用确定性 ODE（无扩散项）沿学习到的速度场积分，返回轨迹。
        assert fb in ['f', 'b']
        ### NOTE: Use Euler method to sample from the learned flow
        if N is None:
            N = self.N
        dt = 1. / N
        traj = []  # to store the trajectory
        z = zstart.detach().clone()
        batchsize = z.shape[0]

        traj.append(z.detach().clone())
        ts = np.arange(N) / N
        if fb == 'b':
            ts = 1 - ts
        sign = 1 if fb == 'f' else -1
        for i in range(N):
            t = torch.ones((batchsize, 1), device=device) * ts[i]
            pred = sign * self.net(z, t)
            z = z.detach().clone() + pred * dt
            traj.append(z.detach().clone())

        return traj


# Rectified Flow
class RectifiedFlow(nn.Module):
    # Rectified Flow（整流流）类
    # 说明：实现了 Rectified Flow 的训练样本构造与基于 learned ODE 的采样方法（sample_ode）。
    def __init__(self, net=None, num_steps=1000, sig=0, eps=0):
        super().__init__()
        self.net = net
        self.optimizer = torch.optim.Adam(self.net.parameters(),
                                          lr=lr)  # torch.optim.AdamW(self.net.parameters(), lr=lr, weight_decay=weight_decay)
        self.N = num_steps
        self.sig = sig
        self.eps = eps

    @torch.no_grad()
    def get_train_tuple(self, x_pairs=None, fb='', first_it=False):
        z0, z1 = x_pairs[:, 0], x_pairs[:, 1]

        t = torch.rand((z1.shape[0], 1), device=device) * (1 - 2 * self.eps) + self.eps
        z_t = t * z1 + (1. - t) * z0
        target = z1 - z0
        return z_t, t, target

    @torch.no_grad()
    def generate_new_dataset(self, x_pairs, prev_model=None, fb='', first_it=False):
        # 说明：在 generate_new_dataset 中，若提供 prev_model，则使用其 sample_ode 生成新的配对；
        # sample_ode 返回沿 learned vector field 的确定性轨迹。
        if prev_model is None:
            assert first_it
            z0, z1 = x_pairs[:, 0], x_pairs[torch.randperm(len(x_pairs)), 1]
        else:
            assert not first_it
            if prev_model.fb == 'f':
                zstart = x_pairs[:, 0]
            else:
                zstart = x_pairs[:, 1]
            zend = prev_model.sample_ode(zstart=zstart, fb=prev_model.fb)[-1]
            if prev_model.fb == 'f':
                z0, z1 = zstart, zend
            else:
                z0, z1 = zend, zstart
        return z0, z1

    @torch.no_grad()
    def sample_ode(self, zstart=None, N=None, fb='', first_it=False):
        assert fb in ['f', 'b']
        ### NOTE: Use Euler method to sample from the learned flow
        if N is None:
            N = self.N
        dt = 1. / N
        traj = []  # to store the trajectory
        z = zstart.detach().clone()
        batchsize = z.shape[0]

        traj.append(z.detach().clone())
        ts = np.arange(N) / N
        if fb == 'b':
            ts = 1 - ts
        sign = 1 if fb == 'f' else -1
        for i in range(N):
            t = torch.ones((batchsize, 1), device=device) * ts[i]
            pred = sign * self.net(z, t)
            z = z.detach().clone() + pred * dt
            traj.append(z.detach().clone())

        return traj


def train_flow_model(flow_model, x_pairs, batch_size, inner_iters, prev_model=None, fb='', first_it=False):
    # 通用 Flow 模型训练函数
    # 说明：接受 RectifiedFlow 或 SBCFM 等具有相似接口的模型并执行训练内循环。
    assert fb in ['f', 'b']
    flow_model.fb = fb
    optimizer = flow_model.optimizer
    loss_curve = []

    dl = iter(DataLoader(
        TensorDataset(*flow_model.generate_new_dataset(x_pairs, prev_model=prev_model, fb=fb, first_it=first_it)),
        batch_size=batch_size, shuffle=True, pin_memory=False, drop_last=True))

    for i in tqdm(range(inner_iters)):
        try:
            z0, z1 = next(dl)
        except StopIteration:
            dl = iter(DataLoader(TensorDataset(
                *flow_model.generate_new_dataset(x_pairs, prev_model=prev_model, fb=fb, first_it=first_it)),
                batch_size=batch_size, shuffle=True, pin_memory=False, drop_last=True))
            z0, z1 = next(dl)

        z_pairs = torch.stack([z0, z1], dim=1)
        z_t, t, target = flow_model.get_train_tuple(x_pairs=z_pairs, fb=fb, first_it=first_it)

        optimizer.zero_grad()
        pred = flow_model.net(z_t, t)
        loss = (target - pred).view(pred.shape[0], -1).abs().pow(2).sum(dim=1)
        loss = loss.mean()
        loss.backward()

        if torch.isnan(loss).any():
            raise ValueError("Loss is nan")
            break

        optimizer.step()
        loss_curve.append(np.log(loss.item()))  ## to store the loss curve

    return flow_model, loss_curve


@torch.no_grad()
def draw_plot(sample_fn, z0, z1, N=None):
    # 可视化函数
    # 说明：绘制起始分布、目标分布以及生成样本的散点图，便于直观检查传输效果。
    traj = sample_fn(N=N)

    plt.figure(figsize=(4, 4))
    plt.xlim(-5, 5)
    plt.ylim(-5, 5)

    plt.scatter(z0[:, 0].cpu().numpy(), z0[:, 1].cpu().numpy(), label=r'$\pi_0$', alpha=0.15)
    plt.scatter(z1[:, 0].cpu().numpy(), z1[:, 1].cpu().numpy(), label=r'$\pi_1$', alpha=0.15)
    plt.scatter(traj[-1][:, 0].cpu().numpy(), traj[-1][:, 1].cpu().numpy(), label='Generated', alpha=0.15)
    plt.legend()
    plt.title('Distribution')
    plt.tight_layout()

    # traj_particles = torch.stack(traj)
    # plt.figure(figsize=(4,4))
    # plt.xlim(-5,5)
    # plt.ylim(-5,5)
    # plt.axis('equal')
    # for i in range(30):
    #   plt.plot(traj_particles[:, i, 0].cpu(), traj_particles[:, i, 1].cpu())
    # plt.title('Transport Trajectory')
    # plt.tight_layout()


def train(cfg: DictConfig):
    # set seed for random number generators in pytorch, numpy and python.random
    if cfg.get("seed"):
        print(f"Seed: <{cfg.seed}>")
        pl.seed_everything(cfg.seed, workers=True)

    # 读取配置并构造高斯源/目标分布：N(-a, I) -> N(a, I)
    a = cfg.a
    dim = cfg.dim
    # 这段代码定义了一个初始概率分布模型，它是均值为 -a、标准差为 1 的多维正态分布（高斯分布），维度由 dim 指定。用于生成源数据或起始样本。
    initial_model = Normal(-a * torch.ones((dim,)), 1)
    target_model = Normal(a * torch.ones((dim,)), 1)

    # 采样训练/测试数据对 (x0, x1)，并缓存到磁盘 data.pt
    x0 = initial_model.sample([dataset_size])
    x1 = target_model.sample([dataset_size])
    # 将源分布样本 x0 和目标分布样本 x1 按第1维堆叠成一个张量 x_pairs，形状为 [dataset_size, 2, dim]
    x_pairs = torch.stack([x0, x1], dim=1).to(device)

    x0_test = initial_model.sample([test_dataset_size])
    x1_test = target_model.sample([test_dataset_size])
    x0_test = x0_test.to(device)
    x1_test = x1_test.to(device)

    torch.save({'x0': x0, 'x1': x1, 'x0_test': x0_test, 'x1_test': x1_test}, "data.pt")

    # 测试阶段的起点映射：正向(f)从 x0 出发，反向(b)从 x1 出发
    x_test_dict = {'f': x0_test, 'b': x1_test}

    # 构建时间条件网络（ScoreNetwork/MLP），根据 cfg.net_name 与激活函数选择规模与结构
    net_split = cfg.net_name.split("_")
    if net_split[0] == "mlp":
        if net_split[1] == "small":
            # 构造一个较小的网络，其隐藏层宽度为 [128, 128, dim]
            net_fn = partial(ScoreNetwork, input_dim=dim + 1, layer_widths=[128, 128, dim],
                             activation_fn=hydra.utils.get_class(
                                 cfg.activation_fn)())  # hydra.utils.get_method(cfg.activation_fn))  #
            # 通过partial，你可以固定一个函数的部分参数，生成一个新的函数，这个新函数只需要传入剩余的参数即可。
            # 固定 ScoreNetwork 类的三个参数：input_dim、layer_widths 和 activation_fn。
            # 生成一个新的可调用对象 net_fn，之后只需调用 net_fn() 即可创建一个配置好的 ScoreNetwork 实例。
            # 这样可以方便地在不同地方多次创建结构相同的网络，而无需重复传递相同的参数
        else:
            net_fn = partial(ScoreNetwork, input_dim=dim + 1, layer_widths=[256, 256, dim],
                             activation_fn=hydra.utils.get_class(
                                 cfg.activation_fn)())  # hydra.utils.get_method(cfg.activation_fn))  #
    else:
        raise NotImplementedError

    # 读取训练超参数（时间步数/噪声/内外层迭代次数）
    num_steps = cfg.num_steps
    sigma = cfg.sigma
    inner_iters = cfg.inner_iters
    outer_iters = cfg.outer_iters

    # 根据 model_name 实例化相应模型与训练函数，并打印可训练参数量
    if cfg.model_name == "dsb":
        model = DSB(net_fwd=net_fn().to(device),
                    net_bwd=net_fn().to(device),
                    num_steps=num_steps, sig=sigma)
        train_fn = train_dsb_ipf
        print(f"Number of parameters: <{sum(p.numel() for p in model.net_fwd.parameters() if p.requires_grad)}>")
    elif cfg.model_name == "dsbm":
        model = DSBM(net_fwd=net_fn().to(device),
                     net_bwd=net_fn().to(device),
                     num_steps=num_steps, sig=sigma, first_coupling=cfg.first_coupling)
        train_fn = train_dsbm
        print(f"Number of parameters: <{sum(p.numel() for p in model.net_fwd.parameters() if p.requires_grad)}>")
    elif cfg.model_name == "sbcfm":
        model = SBCFM(net=net_fn().to(device),
                      num_steps=num_steps, sig=sigma)
        train_fn = train_flow_model
        print(f"Number of parameters: <{sum(p.numel() for p in model.net.parameters() if p.requires_grad)}>")
    elif cfg.model_name == "rectifiedflow":
        model = RectifiedFlow(net=net_fn().to(device),
                              num_steps=num_steps, sig=None)
        train_fn = train_flow_model
        print(f"Number of parameters: <{sum(p.numel() for p in model.net.parameters() if p.requires_grad)}>")
    else:
        raise ValueError("Wrong model_name!")

    # Training loop
    # 外层 IPF 式迭代：按 fb_sequence 在前/后向间交替训练；
    # 每一轮使用上一轮快照(prev_model)生成新的配对或伪标签，再进行当前方向的内层训练
    # first_it = True
    model_list = []
    it = 1

    # assert outer_iters % len(cfg.fb_sequence) == 0
    while it <= outer_iters:
        for fb in cfg.fb_sequence:
            print(f"Iteration {it}/{outer_iters} {fb}")
            # 是否为首轮（首轮不使用 prev_model）
            first_it = (it == 1)
            if first_it:
                prev_model = None
            else:
                # 使用上一轮的模型（eval 模式）生成耦合/轨迹，不参与梯度
                prev_model = model_list[-1]["model"].eval()

            # 单方向训练 inner_iters 步；返回当前模型与损失曲线（对数形式）
            model, loss_curve = train_fn(model, x_pairs, batch_size, inner_iters, prev_model=prev_model, fb=fb,
                                         first_it=first_it)
            # 保存本轮模型快照（仅用于后续生成与评估）
            model_list.append({'fb': fb, 'model': copy.deepcopy(model).eval()})

            if hasattr(model, "sample_sde"):
                # 若模型提供 SDE 采样：绘制生成结果并做收敛评估（均值/方差/协方差）
                # 其中 optimal_result_dict 为解析最优指标，用于对比曲线
                draw_plot(partial(model.sample_sde, zstart=x_test_dict[fb], fb=fb, first_it=first_it),
                          z0=x_test_dict['f'], z1=x_test_dict['b'])
                plt.savefig(f"{it}-{fb}.png")
                plt.close()

                # Evaluation
                # 评估收敛（SDE）：记录每一外层迭代的统计量与理论最优值的对比
                optimal_result_dict = {'mean': -a, 'var': 1, 'cov': (np.sqrt(5) - 1) / 2}
                result_list = {k: [] for k in optimal_result_dict.keys()}
                for i in range(it):
                    traj = model_list[i]['model'].sample_sde(zstart=x1_test, fb='b')
                    result_list['mean'].append(traj[-1].mean(0).mean(0).item())
                    result_list['var'].append(traj[-1].var(0).mean(0).item())
                    result_list['cov'].append(
                        torch.cov(torch.cat([traj[0], traj[-1]], dim=1).T)[dim:, :dim].diag().mean(0).item())
                for i, k in enumerate(result_list.keys()):
                    plt.plot(result_list[k], label=f"{cfg.model_name}-{cfg.net_name}")
                    plt.plot(np.arange(outer_iters), optimal_result_dict[k] * np.ones(outer_iters), label="optimal",
                             linestyle="--")
                    plt.title(k.capitalize())
                    if i == 0:
                        plt.legend()
                    plt.savefig(f"convergence_{k}.png")
                    plt.close()

                # 对比较小步数（N=100）的采样收敛曲线
                result_list_100 = {k: [] for k in optimal_result_dict.keys()}
                for i in range(it):
                    traj_100 = model_list[i]['model'].sample_sde(zstart=x1_test, fb='b', N=100)
                    result_list_100['mean'].append(traj_100[-1].mean(0).mean(0).item())
                    result_list_100['var'].append(traj_100[-1].var(0).mean(0).item())
                    result_list_100['cov'].append(
                        torch.cov(torch.cat([traj_100[0], traj_100[-1]], dim=1).T)[dim:, :dim].diag().mean(0).item())

            if hasattr(model, "sample_ode"):
                # 若模型提供 ODE 采样：绘制生成结果并做收敛评估（均值/方差）
                draw_plot(partial(model.sample_ode, zstart=x_test_dict[fb], fb=fb, first_it=first_it),
                          z0=x_test_dict['f'], z1=x_test_dict['b'])
                plt.savefig(f"{it}-{fb}-ode.png")
                plt.close()

                # Evaluation
                # 评估收敛（ODE）：记录每一外层迭代的统计量与理论最优值的对比
                optimal_result_dict_ode = {'mean': -a, 'var': 1}
                result_list_ode = {k: [] for k in optimal_result_dict_ode.keys()}
                for i in range(it):
                    traj_ode = model_list[i]['model'].sample_ode(zstart=x1_test, fb='b')
                    result_list_ode['mean'].append(traj_ode[-1].mean(0).mean(0).item())
                    result_list_ode['var'].append(traj_ode[-1].var(0).mean(0).item())
                for i, k in enumerate(result_list_ode.keys()):
                    plt.plot(result_list_ode[k], label=f"{cfg.model_name}-{cfg.net_name}-ode")
                    plt.plot(np.arange(outer_iters), optimal_result_dict_ode[k] * np.ones(outer_iters), label="optimal",
                             linestyle="--")
                    plt.title(k.capitalize())
                    if i == 0:
                        plt.legend()
                    plt.savefig(f"convergence_{k}-ode.png")
                    plt.close()

                # 对比较小步数（N=100）的 ODE 采样收敛曲线
                result_list_ode_100 = {k: [] for k in optimal_result_dict_ode.keys()}
                for i in range(it):
                    traj_ode_100 = model_list[i]['model'].sample_ode(zstart=x1_test, fb='b', N=100)
                    result_list_ode_100['mean'].append(traj_ode_100[-1].mean(0).mean(0).item())
                    result_list_ode_100['var'].append(traj_ode_100[-1].var(0).mean(0).item())

            # 进入下一外层迭代；达到外层上限后跳出
            # first_it = False
            it += 1

            if it > outer_iters:
                break

    # 持久化：保存模型快照列表与评估统计/轨迹（CSV/PKL/NPY）
    torch.save([{'fb': m['fb'], 'model': m['model'].state_dict()} for m in model_list], "model_list.pt")

    if hasattr(model, "sample_sde"):
        # 保存 SDE 收敛曲线与粒子轨迹
        df_result = pd.DataFrame(result_list)
        df_result_100 = pd.DataFrame(result_list_100)
        df_result.to_csv('df_result.csv')
        df_result.to_pickle('df_result.pkl')
        df_result_100.to_csv('df_result_100.csv')
        df_result_100.to_pickle('df_result_100.pkl')

        # Trajectory
        np.save("traj.npy", torch.stack(traj, dim=1).detach().cpu().numpy())
        np.save("traj_100.npy", torch.stack(traj_100, dim=1).detach().cpu().numpy())

    if hasattr(model, "sample_ode"):
        # 保存 ODE 收敛曲线与粒子轨迹
        df_result_ode = pd.DataFrame(result_list_ode)
        df_result_ode_100 = pd.DataFrame(result_list_ode_100)
        df_result_ode.to_csv('df_result_ode.csv')
        df_result_ode.to_pickle('df_result_ode.pkl')
        df_result_ode_100.to_csv('df_result_ode_100.csv')
        df_result_ode_100.to_pickle('df_result_ode_100.pkl')

        # Trajectory
        np.save("traj_ode.npy", torch.stack(traj_ode, dim=1).detach().cpu().numpy())
        np.save("traj_ode_100.npy", torch.stack(traj_ode_100, dim=1).detach().cpu().numpy())

    return {}, {}


@hydra.main(config_path="conf", config_name="gaussian.yaml")
def main(cfg: DictConfig) -> Optional[float]:
    # train the model
    train(cfg)


if __name__ == "__main__":
    main()
