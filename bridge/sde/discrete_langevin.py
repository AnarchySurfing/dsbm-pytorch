import torch

# 计算高斯分布的梯度
def grad_gauss(x, m, var):
    # 计算输出值，即均值为m，方差为var的高斯分布在x处的得分函数（score function）
    xout = (m - x) / var
    return xout

# def ornstein_ulhenbeck(x, gradx, gamma):
#     xout = x + gamma * gradx + torch.sqrt(2 * gamma) * torch.randn(x.shape, device=x.device)
#     return xout

# 定义Langevin动力学类
class Langevin:

    def __init__(self, num_steps, shape_x, shape_y, gammas, time_sampler,
                 mean_final=torch.tensor([0., 0.]), var_final=torch.tensor([.5, .5]), 
                 mean_match=True, out_scale=1, var_final_gamma_scale=False):
        # 获取gammas所在的设备
        self.device = gammas.device

        # 是否进行均值匹配
        self.mean_match = mean_match
        # 将最终均值张量移动到指定设备
        self.mean_final = mean_final.to(self.device) if mean_final is not None else None
        # 将最终方差张量移动到指定设备
        self.var_final = var_final.to(self.device) if var_final is not None else None
        
        self.num_steps = num_steps # num diffusion steps
        self.d_x = shape_x # dimension of object to diffuse
        self.d_y = shape_y # dimension of conditioning
        self.gammas = gammas # schedule

        # 创建步数张量
        self.steps = torch.arange(self.num_steps).to(self.device)
        # 计算gammas的累积和作为时间
        self.time = torch.cumsum(self.gammas, 0).to(self.device)
        # self.time_sampler = time_sampler
        # 设置输出缩放因子
        self.out_scale = out_scale
        # 是否根据gamma缩放最终方差
        self.var_final_gamma_scale = var_final_gamma_scale
            

    # 记录初始的Langevin动力学过程（前向过程）
    def record_init_langevin(self, init_samples_x, init_samples_y, mean_final=None, var_final=None):
        if mean_final is None:
            mean_final = self.mean_final
        if var_final is None:
            var_final = self.var_final
        
        x = init_samples_x
        # y = init_samples_y
        # 获取样本数量
        N = x.shape[0]
        # 调整步数张量的形状以匹配批处理
        steps = self.steps.reshape((1,self.num_steps,1)).repeat((N,1,1))


        # 初始化用于存储轨迹的张量
        x_tot = torch.Tensor(N, self.num_steps, *self.d_x).to(x.device)
        # y_tot = torch.Tensor(N, self.num_steps, *self.d_y).to(x.device)
        y_tot = None
        # 初始化用于存储输出的张量
        out = torch.Tensor(N, self.num_steps, *self.d_x).to(x.device)
        num_iter = self.num_steps
        steps_expanded = steps
        
        # 迭代执行Langevin步骤
        for k in range(num_iter):
            gamma = self.gammas[k]

            # 根据配置计算方差与gamma的比率和缩放后的gamma
            if self.var_final_gamma_scale:
                var_gamma_ratio = 1 / gamma
                scaled_gamma = gamma * var_final
            else:
                var_gamma_ratio = var_final / gamma
                scaled_gamma = gamma
            
            # 计算高斯梯度
            gradx = grad_gauss(x, mean_final, var_gamma_ratio)
            # 更新前的一步
            t_old = x + gradx / 2
            # 生成标准正态分布的噪声
            z = torch.randn(x.shape, device=x.device)
            # 更新x，加入噪声
            x = t_old + torch.sqrt(scaled_gamma)*z
            # 再次计算高斯梯度
            gradx = grad_gauss(x, mean_final, var_gamma_ratio)
            # 更新后的一步
            t_new = x + gradx / 2
            # 记录当前步的x值
            x_tot[:, k, :] = x
            # y_tot[:, k, :] = y
            # 根据是否进行均值匹配计算输出
            if self.mean_match:
                out[:, k, :] = (t_old - t_new) #/ (2 * gamma)
            else:
                # 如果out_scale是字符串，则评估它
                out_scale = eval(self.out_scale).to(self.device) if isinstance(self.out_scale, str) else self.out_scale
                out[:, k, :] = (t_old - t_new) / out_scale
            
        return x_tot, y_tot, out, steps_expanded

    # 记录使用神经网络的Langevin序列（反向过程）
    def record_langevin_seq(self, net, samples_x, init_samples_y, fb, sample=False, var_final=None):
        if var_final is None:
            var_final = self.var_final
        # 根据fb参数决定gammas的顺序（'b'为反向，'f'为正向）
        if fb == 'b':
            gammas = torch.flip(self.gammas, (0,))
        elif fb == 'f':
            gammas = self.gammas

        x = samples_x
        # y = init_samples_y
        N = x.shape[0]
        steps = self.steps.reshape((1,self.num_steps,1)).repeat((N,1,1))

        
        # 初始化用于存储轨迹和输出的张量
        x_tot = torch.Tensor(N, self.num_steps, *self.d_x).to(x.device)
        # y_tot = torch.Tensor(N, self.num_steps, *self.d_y).to(x.device)
        y_tot = None
        out = torch.Tensor(N, self.num_steps, *self.d_x).to(x.device)
        steps_expanded = steps
        num_iter = self.num_steps
        
        # 如果进行均值匹配
        if self.mean_match:
            for k in range(num_iter):
                gamma = gammas[k]

                scaled_gamma = gamma
                # 如果需要，根据最终方差缩放gamma
                if self.var_final_gamma_scale:
                    scaled_gamma = scaled_gamma * var_final

                # 使用神经网络计算更新前的值
                t_old = net(x, None, steps[:, k, :])
                
                # 如果是采样且是最后一步，则不加噪声
                if sample & (k==num_iter-1):
                    x = t_old
                else:
                    # 生成标准正态分布的噪声
                    z = torch.randn(x.shape, device=x.device)
                    # 更新x，加入噪声
                    x = t_old + torch.sqrt(scaled_gamma) * z
                    
                # 使用神经网络计算更新后的值
                t_new = net(x, None, steps[:, k, :])
                # 记录当前步的x值
                x_tot[:, k, :] = x
                # y_tot[:, k, :] = y
                # 计算输出
                out[:, k, :] = (t_old - t_new) 
        else:
            # 如果不进行均值匹配
            for k in range(num_iter):
                gamma = gammas[k]

                scaled_gamma = gamma
                if self.var_final_gamma_scale:
                    scaled_gamma = scaled_gamma * var_final
                # 如果out_scale是字符串，则评估它
                out_scale = eval(self.out_scale).to(self.device) if isinstance(self.out_scale, str) else self.out_scale

                # 使用神经网络计算漂移项并更新
                t_old = x + out_scale * net(x, None, steps[:, k, :])
                
                if sample & (k==num_iter-1):
                    x = t_old
                else:
                    z = torch.randn(x.shape, device=x.device)
                    x = t_old + torch.sqrt(scaled_gamma) * z
                # 再次使用神经网络计算漂移项
                t_new = x + out_scale * net(x, None, steps[:, k, :])
                
                x_tot[:, k, :] = x
                # y_tot[:, k, :] = y
                # 计算并缩放输出
                out[:, k, :] = (t_old - t_new) / out_scale
            

        return x_tot, y_tot, out, steps_expanded
