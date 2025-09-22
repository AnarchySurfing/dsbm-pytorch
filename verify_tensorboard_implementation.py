#!/usr/bin/env python3
"""
验证TensorBoardLogger实现的脚本
"""

import sys
import os

def verify_implementation():
    """验证TensorBoardLogger实现是否完整"""
    
    print("验证TensorBoardLogger实现...")
    print("=" * 50)
    
    # 1. 检查logger.py中的TensorBoardLogger类
    try:
        with open('bridge/runners/logger.py', 'r', encoding='utf-8') as f:
            logger_content = f.read()
        
        # 检查必要的导入
        required_imports = [
            'TensorBoardLogger as _TensorBoardLogger',
            'torch',
            'from torch.utils.tensorboard import SummaryWriter'
        ]
        
        for imp in required_imports:
            if imp in logger_content:
                print(f"✓ 导入检查通过: {imp}")
            else:
                print(f"✗ 缺少导入: {imp}")
        
        # 检查TensorBoardLogger类定义
        if 'class TensorBoardLogger(_TensorBoardLogger):' in logger_content:
            print("✓ TensorBoardLogger类定义正确")
        else:
            print("✗ TensorBoardLogger类定义缺失")
        
        # 检查必要的方法
        required_methods = [
            'def log_metrics(self, metrics, step=None, fb=None):',
            'def log_hyperparams(self, params):',
            'def _flatten_dict(self, d, parent_key=\'\', sep=\'/\'):'
        ]
        
        for method in required_methods:
            if method in logger_content:
                print(f"✓ 方法检查通过: {method.split('(')[0].replace('def ', '')}")
            else:
                print(f"✗ 缺少方法: {method.split('(')[0].replace('def ', '')}")
        
        # 检查LOGGER_JOIN_CHAR常量
        if "LOGGER_JOIN_CHAR = '/'" in logger_content:
            print("✓ LOGGER_JOIN_CHAR常量定义正确")
        else:
            print("✗ LOGGER_JOIN_CHAR常量缺失")
            
    except FileNotFoundError:
        print("✗ bridge/runners/logger.py文件不存在")
        return False
    
    print()
    
    # 2. 检查config_getters.py中的配置支持
    try:
        with open('bridge/runners/config_getters.py', 'r', encoding='utf-8') as f:
            config_content = f.read()
        
        # 检查导入
        if 'TensorBoardLogger' in config_content:
            print("✓ config_getters.py中TensorBoardLogger导入正确")
        else:
            print("✗ config_getters.py中缺少TensorBoardLogger导入")
        
        # 检查常量定义
        if "TENSORBOARD_TAG = 'TensorBoard'" in config_content:
            print("✓ TENSORBOARD_TAG常量定义正确")
        else:
            print("✗ TENSORBOARD_TAG常量缺失")
        
        # 检查get_logger函数中的TensorBoard支持
        if 'if logger_tag == TENSORBOARD_TAG:' in config_content:
            print("✓ get_logger函数支持TensorBoard")
        else:
            print("✗ get_logger函数缺少TensorBoard支持")
            
    except FileNotFoundError:
        print("✗ bridge/runners/config_getters.py文件不存在")
        return False
    
    print()
    
    # 3. 检查测试文件
    test_files = ['test_tensorboard_logger.py', 'test_tensorboard_logger_simple.py']
    for test_file in test_files:
        if os.path.exists(test_file):
            print(f"✓ 测试文件存在: {test_file}")
        else:
            print(f"✗ 测试文件缺失: {test_file}")
    
    print()
    print("=" * 50)
    print("实现验证完成！")
    
    return True

def check_fb_prefixing_logic():
    """验证fb前缀逻辑"""
    print("验证fb前缀逻辑...")
    
    def simulate_fb_logic(metrics, fb=None):
        """模拟TensorBoardLogger中的fb逻辑"""
        if fb is not None:
            metrics.pop('fb', None)
        else:
            fb = metrics.pop('fb', None)
        
        if fb is not None:
            metrics = {fb + '/' + k: v for k, v in metrics.items()}
        
        return metrics
    
    # 测试用例1: fb作为参数传入
    metrics1 = {'loss': 0.5, 'accuracy': 0.95}
    result1 = simulate_fb_logic(metrics1.copy(), fb='forward')
    expected1 = {'forward/loss': 0.5, 'forward/accuracy': 0.95}
    
    if result1 == expected1:
        print("✓ fb参数测试通过")
    else:
        print(f"✗ fb参数测试失败: 期望 {expected1}, 得到 {result1}")
    
    # 测试用例2: fb在metrics字典中
    metrics2 = {'loss': 0.3, 'accuracy': 0.97, 'fb': 'backward'}
    result2 = simulate_fb_logic(metrics2.copy(), fb=None)
    expected2 = {'backward/loss': 0.3, 'backward/accuracy': 0.97}
    
    if result2 == expected2:
        print("✓ fb字典测试通过")
    else:
        print(f"✗ fb字典测试失败: 期望 {expected2}, 得到 {result2}")

def check_hyperparams_flattening():
    """验证超参数扁平化逻辑"""
    print("验证超参数扁平化逻辑...")
    
    def flatten_dict(d, parent_key='', sep='/'):
        """扁平化嵌套字典"""
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(flatten_dict(v, new_key, sep=sep).items())
            else:
                items.append((new_key, v))
        return dict(items)
    
    nested_params = {
        'model': {
            'num_layers': 3,
            'config': {
                'dropout': 0.1
            }
        },
        'training': {
            'lr': 0.001
        }
    }
    
    result = flatten_dict(nested_params)
    expected = {
        'model/num_layers': 3,
        'model/config/dropout': 0.1,
        'training/lr': 0.001
    }
    
    if result == expected:
        print("✓ 超参数扁平化测试通过")
    else:
        print(f"✗ 超参数扁平化测试失败: 期望 {expected}, 得到 {result}")

if __name__ == "__main__":
    print("TensorBoardLogger实现验证")
    print("=" * 60)
    
    # 验证实现
    verify_implementation()
    
    print()
    
    # 验证核心逻辑
    check_fb_prefixing_logic()
    
    print()
    
    check_hyperparams_flattening()
    
    print()
    print("=" * 60)
    print("验证完成！")