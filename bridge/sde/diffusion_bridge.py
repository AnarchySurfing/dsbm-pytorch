import torch
import torch.nn as nn
import numpy as np
from .optimal_transport import OTPlanSampler

# 基于方差爆炸（VE）的薛定谔桥扩散模型
class DBDSB_VE:
  def __init__(self, sig, num_steps, timesteps, shape_x, shape_y, first_coupling, mean_match=False, ot_sampler=None, eps=1e-4, **kwargs):
    self.device = timesteps.device

    self.sig = sig              # total sigma from time 0 and T=1
    self.num_steps = num_steps  # num diffusion steps
    self.timesteps = timesteps  # schedule of timesteps
    assert len(self.timesteps) == self.num_steps
    # 确保所有时间步长的总和为 T=1
    assert torch.allclose(self.timesteps.sum(), torch.tensor(self.T))  # sum of timesteps is T=1
    assert (self.timesteps > 0).all()
    # 方差步长时间表
    self.gammas = self.timesteps * self.sig**2  # schedule of variance steps
    
    self.d_x = shape_x  # dimension of object to diffuse
    self.d_y = shape_y  # dimension of conditioning

    self.first_coupling = first_coupling
    self.eps = eps

    self.ot_sampler = None
    if ot_sampler is not None:
      # 初始化最优传输采样器
      self.ot_sampler = OTPlanSampler(ot_sampler, reg=2*self.sig**2)
    self.mean_match = mean_match

  @property
  def T(self):
    # 总扩散时间
    return 1.

  @property
  def alpha(self):
    # alpha 参数，在 VE SDE 中为 0
    return 0.

  @torch.no_grad()
  def marginal_prob(self, x, t, fb):
    # 计算边缘概率 p(x_t|x_0)
    if fb == "f":
      # 前向过程的均值和标准差
      std = self.sig * torch.sqrt(t)
    else:
      # 反向过程的均值和标准差
      std = self.sig * torch.sqrt(self.T - t)
    mean = x
    return mean, std

  @torch.no_grad()
  def record_langevin_seq(self, net, samples_x, init_samples_y, fb, sample=False, num_steps=None, **kwargs):
    # 记录朗之万动力学序列
    if fb == 'b':
      # 反向过程，反转 gammas 和 timesteps
      gammas = torch.flip(self.gammas, (0,))
      timesteps = torch.flip(self.timesteps, (0,))
      # 时间从 T=1 开始
      t = torch.ones((samples_x.shape[0], 1), device=self.device)
      sign = -1.
    elif fb == 'f':
      # 前向过程
      gammas = self.gammas
      timesteps = self.timesteps
      # 时间从 t=0 开始
      t = torch.zeros((samples_x.shape[0], 1), device=self.device)
      sign = 1.

    x = samples_x
    N = x.shape[0]

    if num_steps is None:
      num_steps = self.num_steps
    else:
      # 如果指定了不同的步数，则重新计算时间步长和方差
      timesteps = np.interp(np.arange(1, num_steps+1)/num_steps, np.arange(self.num_steps+1)/self.num_steps, [0, *np.cumsum(timesteps.cpu())])
      timesteps = torch.from_numpy(np.diff(timesteps, prepend=[0])).to(self.device)
      gammas = timesteps * self.sig**2

    # 初始化用于存储轨迹的张量
    x_tot = torch.Tensor(N, num_steps, *self.d_x).to(x.device)
    y_tot = None
    steps_expanded = torch.Tensor(N, num_steps, 1).to(x.device)
    
    # 获取用于预测的漂移函数
    drift_fn = self.get_drift_fn_pred(fb)
    
    # 迭代执行扩散步骤
    for k in range(num_steps):
      gamma = gammas[k]
      timestep = timesteps[k]

      # 网络进行原始预测
      pred = net(x, init_samples_y, t)  # Raw prediction of the network

      if sample and (k==num_steps-1) and self.mean_match:
        # 如果是采样最后一步且使用均值匹配，直接将预测作为结果
        x = pred
      else:
        # 计算漂移项
        drift = drift_fn(t, x, pred)
        # 更新 x (漂移部分)
        x = x + drift * timestep
        if not (sample and (k==num_steps-1)):
          # 添加噪声 (扩散部分)
          x = x + torch.sqrt(gamma) * torch.randn_like(x)

      # 记录当前步的结果
      x_tot[:, k, :] = x
      # y_tot[:, k, :] = y
      steps_expanded[:, k, :] = t
      # 更新时间
      t = t + sign * timestep
    
    if fb == 'b':
      # 检查反向过程是否在 t=0 结束
      assert torch.allclose(t, torch.zeros(1, device=self.device), atol=1e-4, rtol=1e-4), f"{t} != 0"
    else:
      # 检查前向过程是否在 t=1 结束
      assert torch.allclose(t, torch.ones(1, device=self.device) * self.T, atol=1e-4, rtol=1e-4), f"{t} != 1"

    return x_tot, y_tot, None, steps_expanded

  @torch.no_grad()
  def generate_new_dataset(self, x0, y0, x1, sample_fn, sample_direction, sample=False, num_steps=None):
    # 通过模拟生成新的数据集
    if sample_direction == 'f':
      zstart = x0
    else:
      zstart = x1
    # 通过朗之万动力学从起点采样到终点
    zend = self.record_langevin_seq(sample_fn, zstart, y0, sample_direction, sample=sample, num_steps=num_steps)[0][:, -1]
    if sample_direction == 'f':
      z0, z1 = zstart, zend
    else:
      z0, z1 = zend, zstart
    return z0, y0, z1

  @torch.no_grad()
  def probability_flow_ode(self, net_f=None, net_b=None, y=None):
    # 定义概率流常微分方程 (ODE)
    get_drift_fn_net = self.get_drift_fn_net

    class ODEfunc(nn.Module):
      def __init__(self, net_f=None, net_b=None):
        super().__init__()
        self.net_f = net_f
        self.net_b = net_b
        self.nfe = 0
        if self.net_f is not None:
          self.drift_fn_f = get_drift_fn_net(self.net_f, 'f', y=y)
        self.drift_fn_b = get_drift_fn_net(self.net_b, 'b', y=y)

      def forward(self, t, x):
        self.nfe += 1
        t = torch.ones((x.shape[0], 1), device=x.device) * t.item()
        if self.net_f is None:
          # 如果没有前向网络，只使用反向漂移
          return - self.drift_fn_b(t, x)
        # 概率流 ODE 的漂移是前向和反向漂移的平均
        return (self.drift_fn_f(t, x) - self.drift_fn_b(t, x)) / 2

    return ODEfunc(net_f=net_f, net_b=net_b)

  @torch.no_grad()
  def get_train_tuple(self, x0, x1, fb='', first_it=False):
    # 获取训练样本元组 (z_t, t, target)
    if first_it and fb == 'b':
      z0 = x0
      if self.first_coupling == "ref":
        # First coupling is x_0, x_0 perturbed
        # 第一次迭代的反向过程，使用参考耦合，z1 是 z0 的扰动版本
        z1 = z0 + torch.randn_like(z0) * self.sig
      elif self.first_coupling == "ind":
        # 独立耦合，z1 直接使用 x1
        z1 = x1
      else:
        raise NotImplementedError
    elif first_it and fb == 'f':
      assert self.first_coupling == "ind"
      z0, z1 = x0, x1
    else:
      z0, z1 = x0, x1
    
    if self.ot_sampler is not None:
      # 如果使用最优传输采样器，对 z0 和 z1 进行重采样
      assert z0.shape == z1.shape
      original_shape = z0.shape
      z0, z1 = self.ot_sampler.sample_plan(z0.flatten(start_dim=1), z1.flatten(start_dim=1))
      z0, z1 = z0.view(original_shape), z1.view(original_shape)

    # 随机采样时间 t
    t = torch.rand(z1.shape[0], device=self.device) * (1-2*self.eps) + self.eps
    t = t[:, None, None, None]
    # 线性插值构造 z_t
    z_t = t * z1 + (1.-t) * z0
    z = torch.randn_like(z_t)
    # 添加噪声，构造薛定谔桥的样本
    z_t = z_t + self.sig * torch.sqrt(t*(1.-t)) * z
    if self.mean_match:
      # 如果是均值匹配，目标是端点
      if fb == 'f':
        target = z1
      else:
        target = z0
    else:
      # 否则，目标是漂移项
      if fb == 'f':
        # (z1 - z_t) / (1-t)
        # target = z1 - z0 
        # target = target - self.sig * torch.sqrt(t/(1.-t)) * z
        # target = self.A_f(t) * z_t + self.M_f(t) * z1
        drift_f = self.drift_f(t, z_t, z0, z1)
        target = drift_f + self.alpha * z_t
      else:
        # (z0 - z_t) / t
        # target = - (z1 - z0)
        # target = target - self.sig * torch.sqrt((1.-t)/t) * z
        drift_b = self.drift_b(t, z_t, z0, z1)
        target = drift_b - self.alpha * z_t
    return z_t, t, target

  # 前向漂移系数 A(t)
  def A_f(self, t):
    return -1./(self.T-t)

  # 前向漂移系数 M(t)
  def M_f(self, t):
    return 1./(self.T-t)

  # 反向漂移系数 A(t)
  def A_b(self, t):
    return -1./t

  # 反向漂移系数 M(t)
  def M_b(self, t):
    return 1./t

  # 前向漂移函数
  def drift_f(self, t, x, init, final):
    t = t.view(t.shape[0], 1, 1, 1)
    return self.A_f(t) * x + self.M_f(t) * final

  # 反向漂移函数
  def drift_b(self, t, x, init, final):
    t = t.view(t.shape[0], 1, 1, 1)
    return self.A_b(t) * x + self.M_b(t) * init

  def get_drift_fn_net(self, net, fb, y=None):
    # 从神经网络获取漂移函数
    drift_fn_pred = self.get_drift_fn_pred(fb)
    def drift_fn(t, x):
      pred = net(x, y, t)  # Raw prediction of the network
      return drift_fn_pred(t, x, pred)
    return drift_fn

  def get_drift_fn_pred(self, fb):
    # 从网络预测值计算漂移函数
    def drift_fn(t, x, pred):
      if self.mean_match:
        # 均值匹配模式
        if fb == 'f':
          drift = self.drift_f(t, x, None, pred)
        else:
          drift = self.drift_b(t, x, pred, None)
      else:
        # 分数匹配（score-matching-like）模式
        if fb == 'f':
          drift = pred - self.alpha * x
        else:
          drift = pred + self.alpha * x
      return drift
    return drift_fn


# 基于方差保持（VP）的薛定谔桥扩散模型
class DBDSB_VP(DBDSB_VE):
  def __init__(self, sig, num_steps, timesteps, shape_x, shape_y, first_coupling, mean_match=False, ot_sampler=None, eps=1e-4, **kwargs):
    assert ot_sampler is None
    super().__init__(sig, num_steps, timesteps, shape_x, shape_y, first_coupling, mean_match=mean_match, ot_sampler=ot_sampler, eps=eps, **kwargs)

  @property
  def alpha(self):
    # alpha 参数，在 VP SDE 中为 0.5
    return 0.5

  @torch.no_grad()
  def marginal_prob(self, x, t, fb):
    # 计算 VP SDE 的边缘概率
    if fb == "f":
      mean = torch.exp(-0.5 * t) * x
      std = self.sig * torch.sqrt(1 - torch.exp(-t))
    else:
      raise NotImplementedError
    return mean, std

  # VP SDE 的前向漂移系数 A(t)
  def A_f(self, t: float) -> float:
    return -self.alpha / torch.tanh(self.alpha * (self.T - t))

  # VP SDE 的前向漂移系数 M(t)
  def M_f(self, t: float) -> float:
    return self.alpha / torch.sinh(self.alpha * (self.T - t))

  # VP SDE 的反向漂移系数 A(t)
  def A_b(self, t: float) -> float:
    return -self.alpha / torch.tanh(self.alpha * t)

  # VP SDE 的反向漂移系数 M(t)
  def M_b(self, t: float) -> float:
    return self.alpha / torch.sinh(self.alpha * t)
