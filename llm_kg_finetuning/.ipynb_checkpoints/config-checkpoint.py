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
            "entity_text_file": "D:\\forCoding_data\\Tianchi_EcommerceKG\\originalData\\OpenBG500\\OpenBG500_entity2text.tsv",
            "relation_text_file": "D:\\forCoding_data\\Tianchi_EcommerceKG\\originalData\\OpenBG500\\OpenBG500_relation2text.tsv",
            
            # 输出目录配置
            "output_dir": "./results",
            "model_dir": "./model"
        }
        
        # 模型配置
        self.model_config = {
            # 预训练模型名称或路径
            "model_name": "mistralai/Mistral-7B-v0.1",  # 可替换为其他中文模型如"baichuan-inc/Baichuan-7B"等
            
            # 量化设置
            "use_quantization": True,
            "quantization_bits": 4,  # 4位或8位量化
            
            # LoRA参数高效微调设置
            "use_lora": True,
            "lora_rank": 8,  # LoRA秩，影响模型质量和参数量
            "lora_alpha": 16,  # LoRA缩放因子
            "lora_dropout": 0.1,  # LoRA层的dropout率
            "target_modules": ["q_proj", "v_proj"],  # 应用LoRA的目标模块
            
            # DeepSpeed训练加速设置
            "use_deepspeed": False,  # 默认禁用：需要CUDA环境和手动安装deepspeed
            "gradient_checkpointing": True  # 梯度检查点，节省显存
        }
        
        # 训练配置
        self.training_config = {
            "per_device_train_batch_size": 4,  # 每个设备的训练批次大小
            "per_device_eval_batch_size": 4,  # 每个设备的评估批次大小
            "gradient_accumulation_steps": 8,  # 梯度累积步数
            "learning_rate": 2e-5,  # 学习率
            "num_train_epochs": 3,  # 训练轮数
            "warmup_ratio": 0.1,  # 学习率预热比例
            "weight_decay": 0.01,  # 权重衰减
            "save_steps": 500,  # 每多少步保存一次模型
            "logging_steps": 100,  # 每多少步记录一次日志
            "max_seq_length": 200,  # 最大序列长度
            "fp16": True  # 使用混合精度训练
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
    print(f"DeepSpeed: {'启用' if config.model_config['use_deepspeed'] else '禁用'}")
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