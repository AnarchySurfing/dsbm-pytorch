import torch
import hydra
import os

from bridge.trainer_dsb import IPF_DSB
from bridge.runners.config_getters import get_datasets, get_valid_test_datasets
from accelerate import Accelerator


def run(args, output_dir):
    # 创建Accelerator实例，根据设备选择CPU或GPU，并启用批次分割
    accelerator = Accelerator(cpu=args.device == 'cpu', split_batches=True)
    # 打印当前工作目录
    accelerator.print('Directory: ' + os.getcwd())

    # 获取初始数据集、最终数据集、最终均值和方差
    init_ds, final_ds, mean_final, var_final = get_datasets(args)
    # 获取验证和测试数据集
    valid_ds, test_ds = get_valid_test_datasets(args)

    # 初始化最终条件模型为None
    final_cond_model = None
    # 创建IPF_DSB实例，传入各种参数，包括数据集、模型和加速器
    ipf = IPF_DSB(init_ds, final_ds, mean_final, var_final, args, accelerator=accelerator,
                  final_cond_model=final_cond_model, valid_ds=valid_ds, test_ds=test_ds,
                  output_dir=output_dir)
    # 打印加速器状态信息
    accelerator.print(accelerator.state)
    # 打印网络'b'的结构信息
    accelerator.print(ipf.net['b'])
    # 打印可训练参数的数量
    accelerator.print('Number of parameters:', sum(p.numel() for p in ipf.net['b'].parameters() if p.requires_grad))
    # 开始训练过程
    ipf.train()

