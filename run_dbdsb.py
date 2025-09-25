import torch
import hydra
import os

# 从 bridge.trainer_dbdsb 导入 IPF_DBDSB 类，负责训练逻辑
from bridge.trainer_dbdsb import IPF_DBDSB
from bridge.runners.config_getters import get_datasets, get_valid_test_datasets
from accelerate import Accelerator


def run(args, output_dir):
    # 根据 args.device 决定是否使用 CPU，并启用批次拆分以节省内存
    accelerator = Accelerator(cpu=args.device == 'cpu', split_batches=True)
    # 打印当前工作目录，便于调试和日志记录
    accelerator.print('Directory: ' + os.getcwd())

    # 获取初始数据集、目标数据集以及目标分布的均值和方差
    init_ds, final_ds, mean_final, var_final = get_datasets(args)
    # 获取验证集和测试集（如果在配置中指定）
    valid_ds, test_ds = get_valid_test_datasets(args)

    # 最终条件模型占位，若有预训练条件模型可在此替换
    final_cond_model = None
    # 初始化 IPF_DBDSB 实例，传入数据集、分布统计、参数和加速器
    ipf = IPF_DBDSB(init_ds, final_ds, mean_final, var_final, args, accelerator=accelerator,
                    # 传入最终条件模型以及验证/测试数据集
                    final_cond_model=final_cond_model, valid_ds=valid_ds, test_ds=test_ds,
                    output_dir=output_dir)
    # 打印加速器当前状态，确认设备与进程信息
    accelerator.print(accelerator.state)
    # 打印模型中名为 'b' 的子网络结构，便于查看架构
    accelerator.print(ipf.net['b'])
    # 打印可训练参数总数，帮助评估模型规模与资源需求
    accelerator.print('Number of parameters:', sum(p.numel() for p in ipf.net['b'].parameters() if p.requires_grad))
    ipf.train()
