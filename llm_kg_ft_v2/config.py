#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
知识图谱补全模型配置文件
"""
import os
import torch

class Config:
    # 数据集路径配置（注意路径格式转换）
    train_file = "/mnt/d/forCoding_data/Tianchi_EcommerceKG/originalData/OpenBG500/OpenBG500_train.tsv"
    test_file = "/mnt/d/forCoding_data/Tianchi_EcommerceKG/originalData/OpenBG500/OpenBG500_test.tsv"
    dev_file = "/mnt/d/forCoding_data/Tianchi_EcommerceKG/originalData/OpenBG500/OpenBG500_dev.tsv"
    entity_text_file = "/mnt/d/forCoding_data/Tianchi_EcommerceKG/originalData/OpenBG500/OpenBG500_entity2text.tsv"
    relation_text_file = "/mnt/d/forCoding_data/Tianchi_EcommerceKG/originalData/OpenBG500/OpenBG500_relation2text.tsv"
    
    # 模型配置
    model_name = "Qwen/Qwen-1_8B-Chat"
    model_cache_dir = "/mnt/d/forCoding_data/Tianchi_EcommerceKG/trained_models"
    
    # 训练参数（考虑到4070TiS 16G显存限制）
    batch_size = 4
    num_epochs = 3
    learning_rate = 2e-5
    lora_r = 8
    lora_alpha = 16
    lora_dropout = 0.1
    gradient_accumulation_steps = 4
    warmup_ratio = 0.1
    
    # 推理参数
    max_new_tokens = 20
    temperature = 0.7
    
    # 数据处理参数
    num_negatives = 1  # 每个正样本对应的负样本数量
    max_seq_length = 512
    
    # 输出配置
    output_dir = "/mnt/d/forCoding_code/Tianchi_EcommerceKG/llm_kg_ft_v2/outputs"
    log_dir = "/mnt/d/forCoding_code/Tianchi_EcommerceKG/llm_kg_ft_v2/logs"
    
    # 加速配置
    use_deepspeed = True
    fp16 = True  # 使用混合精度训练以减少显存占用
    
    # 其他配置
    device = "cuda" if torch.cuda.is_available() else "cpu"
    seed = 42
    
    def __init__(self):
        # 配置Hugging Face国内镜像源，加速模型下载
        os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
        
        # 创建必要的目录
        os.makedirs(self.model_cache_dir, exist_ok=True)
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)

# 实例化配置对象，供其他模块使用
config = Config()