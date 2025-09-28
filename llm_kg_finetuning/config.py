# LLM知识图谱预测器配置文件

class Config:
    def __init__(self):
        # 数据路径配置
        self.data_paths = {
            # 训练、测试、开发集文件路径
            "train_file": "/mnt/d/forCoding_data/Tianchi_EcommerceKG/originalData/OpenBG500/OpenBG500_train.tsv",
            "test_file": "/mnt/d/forCoding_data/Tianchi_EcommerceKG/originalData/OpenBG500/OpenBG500_test.tsv",
            "dev_file": "/mnt/d/forCoding_data/Tianchi_EcommerceKG/originalData/OpenBG500/OpenBG500_dev.tsv",
            
            # 实体和关系的中文文本映射文件路径
            "entity_text_file": "/mnt/d/forCoding_data/Tianchi_EcommerceKG/originalData/OpenBG500/OpenBG500_entity2text.tsv",
            "relation_text_file": "/mnt/d/forCoding_data/Tianchi_EcommerceKG/originalData/OpenBG500/OpenBG500_relation2text.tsv",
            
            # 输出目录配置
            "output_dir": "/mnt/d/forCoding_data/Tianchi_EcommerceKG/processedData/results",
            "model_dir": "/mnt/d/forCoding_data/Tianchi_EcommerceKG/trained_models"
        }
        
        # 模型配置
        self.model_config = {
            # 预训练模型名称或路径 - 使用Qwen系列轻量级模型
            "model_name": "Qwen/Qwen-1_8B-Chat",  # Qwen系列1.8B参数模型，内存占用极低且无需授权访问
            
            # 量化设置
            "use_quantization": True,
            "quantization_bits": 4,  # 4位量化减少内存使用
            
            # LoRA参数高效微调设置
            "use_lora": True,
            "lora_rank": 4,  # 保持小秩以减少内存使用
            "lora_alpha": 8,  # 相应的缩放因子
            "lora_dropout": 0.1,  # LoRA层的dropout率
            "target_modules": ['c_attn', 'c_proj'],  # Qwen模型常用的注意力和投影模块
            
            # DeepSpeed训练加速设置
            "use_deepspeed": True,  # 启用DeepSpeed以提高训练速度和GPU使用率
            "gradient_checkpointing": True,  # 启用梯度检查点以减少内存使用
            
            # cuDNN优化配置
            "cudnn_optimization": {
                "enabled": True,  # 启用cuDNN优化
                "benchmark": True,  # 启用cuDNN自动调优
                "deterministic": False,  # 禁用确定性以提高性能
                "allow_tf32": True,  # 允许使用TF32格式加速计算
                "conv_workspace_limit_mb": 256  # 设置卷积工作空间限制(MB)
            }
        }
        
        # 训练配置 - 性能优化版本
        self.training_config = {
            "per_device_train_batch_size": 2,  # 增加批次大小以提高GPU使用率
            "per_device_eval_batch_size": 2,  # 评估时也使用更大批次
            "gradient_accumulation_steps": 2,  # 减少梯度累积步数以提高训练速度
            "learning_rate": 2e-5,  # 学习率保持不变
            "num_train_epochs": 2,  # 训练轮数保持不变
            "warmup_ratio": 0.1,  # 学习率预热比例保持不变
            "weight_decay": 0.01,  # 权重衰减保持不变
            "save_steps": 1000,  # 减少保存频率以提高训练速度
            "logging_steps": 100,  # 减少日志记录频率以提高训练速度
            "max_seq_length": 150,  # 保持序列长度不变
            "fp16": True,  # 使用混合精度训练以节省内存
            "gradient_checkpointing": True  # 启用梯度检查点以减少内存使用
        }
        
        # 数据处理配置
        self.data_processing_config = {
            "num_negative_samples": 1,  # 每个正样本对应的负样本数量
            "seed": 42  # 随机种子，确保结果可复现
        }
        
        # 预测配置
        self.prediction_config = {
            "top_k_candidates": 10,  # 预测时考虑的候选实体数量
            "batch_size": 16  # 预测时的批次大小
        }

# 创建全局配置对象
def get_config():
    return Config()

# 打印配置摘要（用于调试）
def print_config_summary(config):
    print("===== 配置摘要 =====")
    print(f"模型: {config.model_config['model_name']}")
    print(f"量化: {'启用' if config.model_config['use_quantization'] else '禁用'} ({config.model_config['quantization_bits']}位)")
    print(f"LoRA: {'启用' if config.model_config['use_lora'] else '禁用'} (rank={config.model_config['lora_rank']})")
    print(f"DeepSpee/mnt/d {'启用' if config.model_config['use_deepspeed'] else '禁用'}")
    print(f"训练批次: {config.training_config['per_device_train_batch_size']} (累积{config.training_config['gradient_accumulation_steps']}步)")
    print(f"学习率: {config.training_config['learning_rate']}")
    print(f"训练轮数: {config.training_config['num_train_epochs']}")
    print(f"负样本比例: {config.data_processing_config['num_negative_samples']}")
    print(f"输出目录: {config.data_paths['output_dir']}")
    print(f"模型保存目录: {config.data_paths['model_dir']}")
    print("===================")

# 验证配置有效性
def validate_config(config):
    """验证配置是否有效"""
    valid = True
    error_messages = []
    
    # 验证数据路径
    required_files = [
        config.data_paths["train_file"],
        config.data_paths["test_file"],
        config.data_paths["dev_file"],
        config.data_paths["entity_text_file"],
        config.data_paths["relation_text_file"]
    ]
    
    for file_path in required_files:
        if not os.path.exists(file_path):
            valid = False
            error_messages.append(f"数据文件不存在: {file_path}")
    
    # 验证模型配置
    if config.model_config["quantization_bits"] not in [4, 8]:
        valid = False
        error_messages.append("量化位数必须是4或8")
    
    if config.model_config["lora_rank"] <= 0:
        valid = False
        error_messages.append("LoRA秩必须大于0")
    
    if config.training_config["learning_rate"] <= 0:
        valid = False
        error_messages.append("学习率必须大于0")
    
    if config.training_config["num_train_epochs"] <= 0:
        valid = False
        error_messages.append("训练轮数必须大于0")
    
    return valid, error_messages

# 保存配置到文件
def save_config(config, file_path="./config.json"):
    """将配置保存到JSON文件"""
    config_dict = {
        "data_paths": config.data_paths,
        "model_config": config.model_config,
        "training_config": config.training_config,
        "data_processing_config": config.data_processing_config,
        "prediction_config": config.prediction_config
    }
    
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(config_dict, f, ensure_ascii=False, indent=2)
    
    print(f"配置已保存到: {file_path}")

# 从文件加载配置
def load_config(file_path="./config.json"):
    """从JSON文件加载配置"""
    if not os.path.exists(file_path):
        print(f"配置文件不存在: {file_path}")
        return None
    
    with open(file_path, 'r', encoding='utf-8') as f:
        config_dict = json.load(f)
    
    config = Config()
    config.data_paths = config_dict.get("data_paths", config.data_paths)
    config.model_config = config_dict.get("model_config", config.model_config)
    config.training_config = config_dict.get("training_config", config.training_config)
    config.data_processing_config = config_dict.get("data_processing_config", config.data_processing_config)
    config.prediction_config = config_dict.get("prediction_config", config.prediction_config)
    
    print(f"配置已从: {file_path} 加载")
    return config

# 需要导入的模块
import os
import json