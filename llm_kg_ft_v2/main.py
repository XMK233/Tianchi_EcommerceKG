#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
主程序模块，整合数据处理、模型训练和预测功能
"""
import os
# 设置清华tuna源以加速依赖包下载
os.environ["PIP_INDEX_URL"] = "https://pypi.tuna.tsinghua.edu.cn/simple"

# 在导入其他模块之前设置多进程启动方法为spawn，解决CUDA在子进程中无法初始化的问题
import multiprocessing
multiprocessing.set_start_method('spawn', force=True)

import sys
import torch
import random
import numpy as np
from config import config
from data_processor import KGDataProcessor
from trainer import KGTrainer

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def set_seed(seed):
    """设置随机种子，保证实验可重复性"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def main():
    """主函数"""
    # 设置随机种子
    set_seed(config.seed)
    
    # 检查CUDA是否可用
    print(f"CUDA可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU设备: {torch.cuda.get_device_name(0)}")
        print(f"GPU显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f}GB")
    
    # 创建数据处理器实例
    data_processor = KGDataProcessor(config)
    
    # 创建训练器实例
    trainer = KGTrainer(config, data_processor)
    
    # 检查是否存在已训练的模型
    model_save_path = os.path.join(config.output_dir, "lora_model")
    if os.path.exists(model_save_path):
        print(f"检测到已存在的模型: {model_save_path}")
        choice = input("是否重新训练模型？(y/n): ").strip().lower()
        if choice != 'y':
            # 加载已训练的模型
            print("加载已训练的模型...")
            model, tokenizer = trainer.load_model()
        else:
            # 重新训练模型
            model, tokenizer = trainer.train()
    else:
        # 训练新模型
        model, tokenizer = trainer.train()
    
    # 准备验证数据并进行评估
    print("\n准备验证数据...")
    dev_data = data_processor.prepare_evaluation_data(config.dev_file)
    print("在开发集上进行评估...")
    dev_accuracy = trainer.evaluate(model, tokenizer, dev_data)
    
    # 准备测试数据并进行预测
    print("\n准备测试数据...")
    test_data = data_processor.prepare_evaluation_data(config.test_file)
    
    # 保存测试集预测结果
    output_file = os.path.join(config.output_dir, "test_predictions.tsv")
    print(f"在测试集上进行预测并保存结果至: {output_file}")
    test_results = trainer.predict(model, tokenizer, test_data, output_file)
    
    # 打印总体结果
    print("\n===== 实验总结 ====")
    print(f"开发集准确率: {dev_accuracy:.4f}")
    print(f"测试集预测结果已保存至: {output_file}")
    print(f"总共预测了 {len(test_results)} 个三元组")
    print("=================")

if __name__ == "__main__":
    main()