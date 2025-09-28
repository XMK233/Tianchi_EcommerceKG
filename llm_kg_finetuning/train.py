import os
import sys
import json
import argparse
import torch
import numpy as np
import random
import time
from tqdm import tqdm
from datasets import Dataset, load_dataset, concatenate_datasets
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    DataCollatorForLanguageModeling,
    TrainingArguments,
    Trainer,
    BitsAndBytesConfig,
    logging as transformers_logging
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
# import deepspeed
import pandas as pd

# 设置日志级别
transformers_logging.set_verbosity_info()

# 导入自定义模块
from config import get_config, print_config_summary, validate_config, save_config, load_config
from data_processor import (
    EntityRelationMapper,
    KnowledgeGraphDataLoader,
    PromptGenerator,
    NegativeSampler,
    create_data_statistics,
    filter_unknown_entities
)

def set_seed(seed=42, config=None):
    """设置随机种子以确保结果可复现"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        
        # 检查是否有cuDNN优化配置
        if config and 'cudnn_optimization' in config.model_config:
            cudnn_config = config.model_config['cudnn_optimization']
            if cudnn_config.get('enabled', True):
                # 根据配置设置cuDNN参数
                torch.backends.cudnn.deterministic = cudnn_config.get('deterministic', False)
                torch.backends.cudnn.benchmark = cudnn_config.get('benchmark', True)
                torch.backends.cudnn.enabled = True
                
                # 设置TF32支持
                if cudnn_config.get('allow_tf32', True):
                    torch.backends.cudnn.allow_tf32 = True
                    torch.backends.cuda.matmul.allow_tf32 = True
                    # 设置相关环境变量
                    import os
                    os.environ["NVIDIA_TF32_OVERRIDE"] = "0"
                    os.environ["TORCH_CUDNN_FIND_ALGORITHM"] = "1"
                    
                # 设置工作空间限制
                conv_workspace_limit = cudnn_config.get('conv_workspace_limit_mb', 256)
                import os
                os.environ["CUDNN_CONV_WORKSPACE_LIMIT"] = str(conv_workspace_limit)
        else:
            # 默认cuDNN优化设置
            torch.backends.cudnn.deterministic = False  # 禁用确定性以提高性能
            torch.backends.cudnn.benchmark = True  # 启用自动寻找最佳卷积算法
            torch.backends.cudnn.enabled = True  # 确保cuDNN启用
            # 设置cuDNN的卷积优化参数
            torch.backends.cudnn.allow_tf32 = True  # 允许使用TF32格式加速计算
            torch.backends.cuda.matmul.allow_tf32 = True

def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="LLM知识图谱预测器训练脚本")
    
    # 配置文件参数
    parser.add_argument("--config", type=str, default="./config.json", 
                        help="配置文件路径")
    parser.add_argument("--save-config", action="store_true", 
                        help="保存当前配置到文件")
    
    # 模式选择
    parser.add_argument("--mode", type=str, default="train", choices=["train", "eval", "predict"],
                        help="运行模式: train(训练), eval(评估), predict(预测)")
    
    # 数据路径覆盖参数
    parser.add_argument("--train-file", type=str, default=None, 
                        help="训练集文件路径")
    parser.add_argument("--test-file", type=str, default=None, 
                        help="测试集文件路径")
    parser.add_argument("--dev-file", type=str, default=None, 
                        help="开发集文件路径")
    parser.add_argument("--entity-text-file", type=str, default=None, 
                        help="实体文本映射文件路径")
    parser.add_argument("--relation-text-file", type=str, default=None, 
                        help="关系文本映射文件路径")
    parser.add_argument("--output-dir", type=str, default=None, 
                        help="输出目录")
    parser.add_argument("--model-dir", type=str, default=None, 
                        help="模型保存目录")
    
    # 模型参数覆盖
    parser.add_argument("--model-name", type=str, default=None, 
                        help="预训练模型名称或路径")
    parser.add_argument("--use-quantization", action="store_true", 
                        help="启用模型量化")
    parser.add_argument("--no-use-quantization", action="store_true", 
                        help="禁用模型量化")
    parser.add_argument("--quantization-bits", type=int, default=None, 
                        help="量化位数(4或8)")
    parser.add_argument("--use-lora", action="store_true", 
                        help="启用LoRA参数高效微调")
    parser.add_argument("--no-use-lora", action="store_true", 
                        help="禁用LoRA参数高效微调")
    parser.add_argument("--lora-rank", type=int, default=None, 
                        help="LoRA秩")
    
    # 训练参数覆盖
    parser.add_argument("--batch-size", type=int, default=None, 
                        help="训练批次大小")
    parser.add_argument("--learning-rate", type=float, default=None, 
                        help="学习率")
    parser.add_argument("--num-epochs", type=int, default=None, 
                        help="训练轮数")
    parser.add_argument("--gradient-accumulation-steps", type=int, default=None, 
                        help="梯度累积步数")
    
    # 预测参数
    parser.add_argument("--top-k", type=int, default=None, 
                        help="预测时考虑的候选实体数量")
    
    # 其他选项
    parser.add_argument("--debug", action="store_true", 
                        help="启用调试模式")
    parser.add_argument("--use-deepspeed", action="store_true", 
                        help="启用DeepSpeed加速")
    parser.add_argument("--no-use-deepspeed", action="store_true", 
                        help="禁用DeepSpeed加速")
    
    args = parser.parse_args()
    return args

def update_config_with_args(config, args):
    """使用命令行参数更新配置"""
    # 更新数据路径
    if args.train_file is not None:
        config.data_paths["train_file"] = args.train_file
    if args.test_file is not None:
        config.data_paths["test_file"] = args.test_file
    if args.dev_file is not None:
        config.data_paths["dev_file"] = args.dev_file
    if args.entity_text_file is not None:
        config.data_paths["entity_text_file"] = args.entity_text_file
    if args.relation_text_file is not None:
        config.data_paths["relation_text_file"] = args.relation_text_file
    if args.output_dir is not None:
        config.data_paths["output_dir"] = args.output_dir
    if args.model_dir is not None:
        config.data_paths["model_dir"] = args.model_dir
    
    # 更新模型配置
    if args.model_name is not None:
        config.model_config["model_name"] = args.model_name
    if args.use_quantization:
        config.model_config["use_quantization"] = True
    if args.no_use_quantization:
        config.model_config["use_quantization"] = False
    if args.quantization_bits is not None:
        config.model_config["quantization_bits"] = args.quantization_bits
    if args.use_lora:
        config.model_config["use_lora"] = True
    if args.no_use_lora:
        config.model_config["use_lora"] = False
    if args.lora_rank is not None:
        config.model_config["lora_rank"] = args.lora_rank
    if args.use_deepspeed:
        config.model_config["use_deepspeed"] = True
    if args.no_use_deepspeed:
        config.model_config["use_deepspeed"] = False
    
    # 更新训练配置
    if args.batch_size is not None:
        config.training_config["per_device_train_batch_size"] = args.batch_size
        config.training_config["per_device_eval_batch_size"] = args.batch_size
    if args.learning_rate is not None:
        config.training_config["learning_rate"] = args.learning_rate
    if args.num_epochs is not None:
        config.training_config["num_train_epochs"] = args.num_epochs
    if args.gradient_accumulation_steps is not None:
        config.training_config["gradient_accumulation_steps"] = args.gradient_accumulation_steps
    
    # 更新预测配置
    if args.top_k is not None:
        config.prediction_config["top_k_candidates"] = args.top_k
    
    return config

def load_model_and_tokenizer(config):
    """加载模型和分词器"""
    print(f"\n加载模型: {config.model_config['model_name']}...")
    
    # 设置量化配置
    quantization_config = None
    if config.model_config["use_quantization"]:
        if config.model_config["quantization_bits"] == 4:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
        elif config.model_config["quantization_bits"] == 8:
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_enable_fp32_cpu_offload=True  # 关键：启用FP32参数的CPU卸载
            )
        print(f"启用量化: {config.model_config['quantization_bits']}位，已启用CPU卸载")
    else:
        # 即使不使用量化，也可以使用这些优化
        print("未启用量化，但仍会应用内存优化")
    
    # 配置Hugging Face国内镜像源和下载优化
    import os
    # 设置Hugging Face镜像源 - 全面配置以避免直接连接huggingface.co
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"  # 主镜像源
    os.environ["HF_MIRROR"] = "https://hf-mirror.com"  # 备用镜像源设置
    # 设置特定的模型镜像路径
    os.environ["HUGGINGFACE_HUB_CACHE"] = "/mnt/d/forCoding_data/Tianchi_EcommerceKG/trained_models"
    
    # 启用并行下载和分片下载
    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"  # 启用hf_transfer加速
    os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"  # 禁用进度条减少开销
    
    # 添加额外的网络配置以避免连接原始huggingface.co
    import huggingface_hub
    # 设置主要镜像源
    huggingface_hub.constants.HF_ENDPOINT = "https://hf-mirror.com"
    # 设置备用镜像源
    os.environ["HF_HUB_ENDPOINT"] = "https://hf-mirror.com"
    # 增加网络重试次数
    os.environ["HF_HUB_RETRY_MAX"] = "5"
    os.environ["HF_HUB_RETRY_DELAY"] = "10"
    
    # 性能优化环境变量（非DeepSpeed版本）
    # CUDA内存分配优化
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,max_split_size_mb:128"  # 启用可扩展内存段并设置最大分割大小
    os.environ["PYTORCH_NO_CUDA_MEMORY_CACHING"] = "0"  # 启用CUDA内存缓存以提高效率
    
    # NCCL通信优化
    os.environ["NCCL_DEBUG"] = "WARN"  # 仅显示警告级别的NCCL日志
    os.environ["NCCL_LAUNCH_MODE"] = "PARALLEL"  # 并行启动NCCL操作
    os.environ["NCCL_NET"] = "Socket"  # 使用Socket作为网络后端
    os.environ["NCCL_BLOCKING_WAIT"] = "0"  # 非阻塞等待以提高并行性
    
    # cuDNN优化环境变量
    os.environ["CUDNN_BENCHMARK"] = "1"  # 启用cuDNN自动调优
    os.environ["CUDNN_CONV_WORKSPACE_LIMIT"] = "256"  # 设置卷积工作空间限制(MB)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"  # 启用cublas工作空间配置提高性能
    
    # NVIDIA GPU优化
    os.environ["NVIDIA_TF32_OVERRIDE"] = "0"  # 允许使用TF32格式加速计算
    os.environ["TORCH_CUDNN_FIND_ALGORITHM"] = "1"  # 启用算法搜索以找到最佳卷积实现
    
    # 加载分词器 - 使用本地已下载的模型
    local_model_path = "/mnt/d/forCoding_data/Tianchi_EcommerceKG/trained_models"
    print(f"使用本地模型路径: {local_model_path}")
    
    # 验证本地模型路径存在性
    import os
    if not os.path.exists(local_model_path):
        raise FileNotFoundError(f"本地模型路径不存在: {local_model_path}")
    
    # 检查路径中的关键文件
    config_json_path = os.path.join(local_model_path, "config.json")
    tokenizer_config_path = os.path.join(local_model_path, "tokenizer_config.json")
    
    print(f"检查关键文件:\n- config.json: {os.path.exists(config_json_path)}\n- tokenizer_config.json: {os.path.exists(tokenizer_config_path)}")
    
    # 如果配置文件不存在，可能是目录结构问题，尝试使用原始模型名称作为后备方案
    try:
        if not os.path.exists(config_json_path):
            print(f"警告: 未在{local_model_path}找到完整的配置文件，尝试使用原始模型名称进行加载...")
            # 首先尝试从本地缓存加载原始模型的分词器
            try:
                tokenizer = AutoTokenizer.from_pretrained(
                    "Qwen/Qwen-1_8B-Chat",  # 使用原始模型名称
                    use_fast=True,
                    padding_side="right",
                    trust_remote_code=True,
                    local_files_only=True,  # 先尝试本地文件
                    cache_dir="/mnt/d/forCoding_data/Tianchi_EcommerceKG/trained_models",
                    revision="main"
                )
                print("成功从本地缓存加载原始模型的分词器")
            except Exception as e:
                print(f"从本地缓存加载分词器失败，尝试从镜像源下载: {str(e)}")
                # 放宽限制，允许从网络下载
                tokenizer = AutoTokenizer.from_pretrained(
                    "Qwen/Qwen-1_8B-Chat",  # 使用原始模型名称
                    use_fast=True,
                    padding_side="right",
                    trust_remote_code=True,
                    local_files_only=False,  # 允许在本地文件缺失时从网络下载
                    cache_dir="/mnt/d/forCoding_data/Tianchi_EcommerceKG/trained_models",
                    revision="main"
                )
        else:
            # 使用本地路径加载
            print(f"尝试从本地路径加载分词器: {local_model_path}")
            tokenizer = AutoTokenizer.from_pretrained(
                local_model_path,
                use_fast=True,
                padding_side="right",
                trust_remote_code=True,
                local_files_only=False,  # 允许在本地文件缺失时从网络下载
                cache_dir="/mnt/d/forCoding_data/Tianchi_EcommerceKG/trained_models"
            )
    except Exception as e:
        print(f"分词器加载失败，将尝试最后的后备方案: {str(e)}")
        # 最后的后备方案：使用原始模型名称并允许网络访问
        tokenizer = AutoTokenizer.from_pretrained(
            "Qwen/Qwen-1_8B-Chat",
            use_fast=True,
            padding_side="right",
            trust_remote_code=True,
            local_files_only=False,
            cache_dir="/mnt/d/forCoding_data/Tianchi_EcommerceKG/trained_models",
            mirror="tuna"
        )
    
    # 设置padding token（针对Qwen模型的特殊处理）
    print(f"设置前 - pad_token: {tokenizer.pad_token}, eos_token: {tokenizer.eos_token}")
    
    # 对于Qwen模型，我们需要特别处理pad_token
    if tokenizer.pad_token is None:
        # 首先尝试使用eos_token
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
            print(f"已将pad_token设置为eos_token: {tokenizer.pad_token}")
        else:
            # 如果eos_token也不存在，使用Qwen模型常用的空格token
            if hasattr(tokenizer, 'vocab'):
                space_token_id = 220  # Qwen模型中常用的空格token_id
                if space_token_id in tokenizer.vocab.values():
                    tokenizer.pad_token = tokenizer.convert_ids_to_tokens(space_token_id)
                    print(f"已将空格token作为pad_token: '{tokenizer.pad_token}'")
                else:
                    # 回退方案：使用词汇表中的第一个token
                    first_token = next(iter(tokenizer.vocab.keys()))
                    tokenizer.pad_token = first_token
                    print(f"已将词汇表第一个token作为pad_token: '{first_token}'")
            else:
                print("警告：无法访问tokenizer.vocab，pad_token可能未正确设置")
    
    # 确保分词器配置正确
    tokenizer.padding_side = "right"  # 确保padding在右侧
    print(f"最终配置 - pad_token: '{tokenizer.pad_token}', padding_side: {tokenizer.padding_side}")
    if tokenizer.pad_token is not None:
        print(f"pad_token_id: {tokenizer.pad_token_id}")
    
    # 优化设备映射 - 更精细地控制内存使用
    device_map = "auto"
    if torch.cuda.is_available():
        # 当显存不足时，使用更激进的设备映射策略
        # 先让Hugging Face自动分配，如果仍有问题则会回退到手动映射
        print(f"使用自动设备映射 + CPU卸载策略")
    
    # 设置设备内存限制 - 更激进的内存限制以防止OOM
    max_memory = {
        0: "6GiB",  # 进一步降低GPU内存限制至6GB，强制更多层卸载到CPU
        "cpu": "32GiB"  # 限制CPU内存使用32GB
    }
    
    # 添加额外的性能优化参数 - 针对Qwen模型的特殊处理
    model_kwargs = {
            "quantization_config": quantization_config,
            "torch_dtype": torch.float16,
            "trust_remote_code": True,  # Qwen模型需要此参数为True
            "mirror": "tuna",
            "cache_dir": "/mnt/d/forCoding_data/Tianchi_EcommerceKG/trained_models",
            "low_cpu_mem_usage": True,
            "use_safetensors": True,
        "local_files_only": False,  # 允许在本地文件缺失时从网络下载
        "max_memory": max_memory,
        "attn_implementation": "sdpa"  # 使用SDPA注意力实现提高性能
        }
    
    # 尝试处理cpp_kernels.py缺失问题
    try:
        # 检查是否存在必要的文件结构
        import sys
        import transformers
        import importlib.util
        
        # 尝试手动添加模型目录到Python路径
        if os.path.exists(local_model_path):
            sys.path.append(local_model_path)
            print(f"已将模型路径添加到Python路径: {local_model_path}")
    except Exception as e:
        print(f"添加Python路径时出错: {str(e)}")
    
    # 当使用DeepSpeed时，让DeepSpeed控制设备映射
    if config.model_config["use_deepspeed"]:
        model_kwargs["device_map"] = None  # 让DeepSpeed管理设备映射
    else:
        model_kwargs["device_map"] = device_map
    
    # 加载模型 - 改进的加载逻辑与错误处理
    try:
        if not os.path.exists(config_json_path):
            print(f"警告: 未在{local_model_path}找到完整的模型配置文件，尝试使用原始模型名称进行加载...")
            # 首先尝试从本地缓存加载原始模型
            try:
                # 创建本地缓存加载专用参数
                model_kwargs_local = model_kwargs.copy()
                model_kwargs_local['local_files_only'] = True  # 强制使用本地文件
                
                model = AutoModelForCausalLM.from_pretrained(
                    "Qwen/Qwen-1_8B-Chat",  # 使用原始模型名称
                    **model_kwargs_local
                )
                print("成功从本地缓存加载原始模型")
            except Exception as e:
                print(f"从本地缓存加载失败，尝试从镜像源下载: {str(e)}")
                # 放宽限制，允许从网络下载
                model_kwargs_local = model_kwargs.copy()
                model_kwargs_local['local_files_only'] = False  # 允许从网络下载
                model = AutoModelForCausalLM.from_pretrained(
                    "Qwen/Qwen-1_8B-Chat",  # 使用原始模型名称
                    **model_kwargs_local
                )
        else:
            # 尝试使用本地路径加载
            print(f"尝试从本地路径加载模型: {local_model_path}")
            
            # 特殊处理Qwen模型的cpp_kernels.py缺失问题
            try:
                # 首先尝试正常加载
                model = AutoModelForCausalLM.from_pretrained(
                    local_model_path,
                    **model_kwargs
                )
            except Exception as e:
                if "cpp_kernels.py" in str(e):
                    print("检测到cpp_kernels.py缺失问题，尝试使用兼容模式加载...")
                    # 禁用一些可能依赖cpp_kernels的特性
                    compatible_kwargs = model_kwargs.copy()
                    compatible_kwargs["attn_implementation"] = "eager"  # 不使用可能依赖cpp_kernels的SDPA实现
                    compatible_kwargs["torch_dtype"] = torch.float32  # 回退到float32以增加兼容性
                    
                    model = AutoModelForCausalLM.from_pretrained(
                        local_model_path,
                        **compatible_kwargs
                    )
                    print("已使用兼容模式加载模型")
                else:
                    raise
    except Exception as e:
        print(f"模型加载失败，将尝试最后的后备方案: {str(e)}")
        # 最后的后备方案：使用原始模型名称并允许网络访问
        final_model_kwargs = model_kwargs.copy()
        final_model_kwargs['local_files_only'] = False
        final_model_kwargs['mirror'] = 'tuna'  # 强制使用国内镜像
        
        # 如果是cpp_kernels问题，使用兼容模式
        if "cpp_kernels.py" in str(e):
            final_model_kwargs["attn_implementation"] = "eager"
            final_model_kwargs["torch_dtype"] = torch.float32
            print("使用最后的后备方案(兼容模式)")
        model = AutoModelForCausalLM.from_pretrained(
            "Qwen/Qwen-1_8B-Chat",
            **final_model_kwargs
        )
    
    # 打印当前内存使用情况
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated(0) / 1024**3
        reserved = torch.cuda.memory_reserved(0) / 1024**3
        print(f"模型加载后 - GPU内存已分配: {allocated:.2f}GB, 已保留: {reserved:.2f}GB")
    
    # 准备模型进行量化训练
    if config.model_config["use_quantization"]:
        model = prepare_model_for_kbit_training(model)
    
    # 应用LoRA
    if config.model_config["use_lora"]:
        # 添加调试代码查看模型模块结构
        print("\n查看模型的模块结构...")
        print(f"模型类型: {type(model)}")
        
        # 打印模型的顶层模块
        print("\n模型顶层模块:")
        for name, module in model.named_children():
            print(f"- {name}: {type(module)}")
        
        # 尝试获取模型的每一层来查看注意力模块名称
        if hasattr(model, 'model') and hasattr(model.model, 'layers'):
            print("\n第一层注意力模块结构:")
            first_layer = model.model.layers[0]
            for name, module in first_layer.named_children():
                print(f"- {name}: {type(module)}")
                # 如果找到注意力相关模块，进一步查看其结构
                if 'attn' in name.lower():
                    print("  注意力子模块:")
                    for attn_name, attn_module in module.named_children():
                        print(f"  - {attn_name}: {type(attn_module)}")
        
        # 设置适合Qwen模型的LoRA配置
        lora_config = LoraConfig(
            r=config.model_config["lora_rank"],
            lora_alpha=config.model_config["lora_alpha"],
            target_modules=config.model_config["target_modules"],
            lora_dropout=config.model_config["lora_dropout"],
            bias="none",
            task_type="CAUSAL_LM"
        )
        model = get_peft_model(model, lora_config)
        print("\n可训练参数:")
        model.print_trainable_parameters()
    
    print("模型加载完成")
    return tokenizer, model

def prepare_training_data(config, data_loader, mapper):
    """准备训练数据"""
    print("\n准备训练数据...")
    
    # 收集所有实体
    all_entities = list(data_loader.collect_all_entities())
    
    # 创建Prompt生成器和负采样器
    prompt_generator = PromptGenerator()
    negative_sampler = NegativeSampler(all_entities)
    
    # 准备训练数据
    train_data = []
    for h, r, t in tqdm(data_loader.train_triples, desc="处理训练数据"):
        # 获取中文文本
        h_text = mapper.get_entity_text(h)
        r_text = mapper.get_relation_text(r)
        t_text = mapper.get_entity_text(t)
        
        # 生成正样本
        pos_prompt = prompt_generator.generate_positive_prompt(h_text, r_text, t_text)
        train_data.append({
            "text": pos_prompt,
            "label": 1
        })
        
        # 生成负样本
        num_neg_samples = config.data_processing_config["num_negative_samples"]
        neg_entities = negative_sampler.sample_negative(t, num_neg_samples)
        for neg_t in neg_entities:
            neg_t_text = mapper.get_entity_text(neg_t)
            neg_prompt = prompt_generator.generate_negative_prompt(h_text, r_text, neg_t_text)
            train_data.append({
                "text": neg_prompt,
                "label": 0
            })
    
    # 转换为Dataset对象
    train_dataset = Dataset.from_pandas(
        pd.DataFrame(train_data).sample(frac=1, random_state=config.data_processing_config["seed"])
    )
    
    # 准备开发集数据
    dev_data = []
    for h, r, t in tqdm(data_loader.dev_triples[:1000], desc="处理开发数据"):  # 限制开发集大小以提高效率
        h_text = mapper.get_entity_text(h)
        r_text = mapper.get_relation_text(r)
        t_text = mapper.get_entity_text(t)
        
        # 只生成正样本用于评估
        pos_prompt = prompt_generator.generate_positive_prompt(h_text, r_text, t_text)
        dev_data.append({
            "text": pos_prompt,
            "label": 1
        })
    
    dev_dataset = Dataset.from_pandas(pd.DataFrame(dev_data))
    
    print(f"训练数据大小: {len(train_dataset)}")
    print(f"开发数据大小: {len(dev_dataset)}")
    
    return train_dataset, dev_dataset, all_entities

def setup_training_args(config):
    """设置训练参数"""
    print("\n设置训练参数...")
    
    # 确保输出目录存在
    os.makedirs(config.data_paths["output_dir"], exist_ok=True)
    os.makedirs(config.data_paths["model_dir"], exist_ok=True)
    
    # 配置DeepSpeed参数 - 高性能优化版本
    deepspeed_config_path = None
    if config.model_config["use_deepspeed"]:
        deepspeed_config = {
            "fp16": {
                "enabled": config.training_config["fp16"],
                "loss_scale": 0,
                "loss_scale_window": 1000,
                "initial_scale_power": 16,
                "hysteresis": 2
            },
            "bf16": {
                "enabled": False  # 如果支持bf16，可以设置为True以获得更好的性能
            },
            "optimizer": {
                "type": "AdamW",
                "params": {
                    "lr": config.training_config["learning_rate"],
                    "weight_decay": config.training_config["weight_decay"],
                    "bias_correction": True,
                    "adam_w_mode": True,
                    "eps": 1e-8
                }
            },
            "scheduler": {
                "type": "WarmupLR",
                "params": {
                    "warmup_min_lr": 0,
                    "warmup_max_lr": config.training_config["learning_rate"],
                    "warmup_num_steps": config.training_config["warmup_ratio"]
                }
            },
            "zero_optimization": {
                "stage": 3,  # 使用Stage 3获得最佳内存效率
                "allgather_partitions": True,
                "allgather_bucket_size": 2e8,
                "overlap_comm": True,
                "reduce_scatter": True,
                "reduce_bucket_size": 2e8,
                "contiguous_gradients": True,
                "cpu_offload": False,  # 禁用CPU卸载以提高速度
                "pin_memory": True,     # 启用内存固定
                "offload_param": {
                    "device": "none"  # 不使用参数卸载
                },
                "offload_optimizer": {
                    "device": "none"  # 不使用优化器卸载
                },
                "stage3_gather_fp16_weights_on_model_save": True  # 模型保存时聚集fp16权重
            },
            # 添加cuDNN相关优化
            "training_optimizer": {
                "fp16": True,
                "bf16": False,
                "tf32": True  # 启用TF32格式以提高性能
            },
            "steps_per_print": 50,  # 更频繁地打印进度信息
            "wall_clock_breakdown": False,
            "gradient_accumulation_steps": config.training_config["gradient_accumulation_steps"],
            "gradient_clipping": 1.0,
            "train_batch_size": config.training_config["per_device_train_batch_size"] * config.training_config["gradient_accumulation_steps"],
            "train_micro_batch_size_per_gpu": config.training_config["per_device_train_batch_size"],
            "activation_checkpointing": {
                "enabled": config.model_config["gradient_checkpointing"],
                "include_modules": ["QWenBlock"]  # Qwen模型特定的模块名称
            },
            "aio": {
                "block_size": 1048576,  # 1MB
                "queue_depth": 4,
                "thread_count": 1,
                "single_submit": True,
                "overlap_events": True
            }
        }
        
        # 保存DeepSpeed配置
        deepspeed_config_path = os.path.join(config.data_paths["output_dir"], "deepspeed_config.json")
        with open(deepspeed_config_path, "w") as f:
            json.dump(deepspeed_config, f, indent=2)
        print(f"DeepSpeed配置已保存到: {deepspeed_config_path}")
    
    # 设置训练参数
    training_args_dict = {
        "output_dir": config.data_paths["output_dir"],
        "per_device_train_batch_size": config.training_config["per_device_train_batch_size"],
        "per_device_eval_batch_size": config.training_config["per_device_eval_batch_size"],
        "gradient_accumulation_steps": config.training_config["gradient_accumulation_steps"],
        "learning_rate": config.training_config["learning_rate"],
        "num_train_epochs": config.training_config["num_train_epochs"],
        "warmup_ratio": config.training_config["warmup_ratio"],
        "weight_decay": config.training_config["weight_decay"],
        "save_steps": config.training_config["save_steps"],
        "logging_steps": config.training_config["logging_steps"],
        "eval_strategy": "steps",
        "save_total_limit": 3,
        "load_best_model_at_end": True,
        "metric_for_best_model": "loss",
        "greater_is_better": False,
        "fp16": config.training_config["fp16"],
        "gradient_checkpointing": config.model_config["gradient_checkpointing"]
    }
    
    # 仅当启用DeepSpeed时才添加deepspeed参数
    if config.model_config["use_deepspeed"]:
        training_args_dict["deepspeed"] = deepspeed_config_path
    
    training_args = TrainingArguments(**training_args_dict)
    
    return training_args

def create_trainer(model, tokenizer, training_args, train_dataset, dev_dataset, config):
    """创建Trainer实例"""
    print("\n创建Trainer...")
    
    # 验证tokenizer的padding设置
    print(f"进入create_trainer时 - pad_token: {tokenizer.pad_token}, eos_token: {tokenizer.eos_token}")
    
    # 确保padding_side正确设置为右侧
    tokenizer.padding_side = "right"
    
    # 强制设置有效的pad_token
    if tokenizer.pad_token is None:
        print("警告：tokenizer.pad_token为None，正在强制设置pad_token...")
        # 首先尝试使用eos_token
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
            print(f"已将pad_token设置为eos_token: {tokenizer.pad_token}")
        else:
            # 如果eos_token也不存在，使用一个常用的token_id（如0）作为备用
            default_pad_token_id = 0
            try:
                # 尝试使用词汇表中的第一个token
                if hasattr(tokenizer, 'vocab') and len(tokenizer.vocab) > 0:
                    first_token = next(iter(tokenizer.vocab.keys()))
                    tokenizer.pad_token = first_token
                    print(f"已将词汇表第一个token作为pad_token: '{first_token}'")
                else:
                    # 直接设置一个默认的token
                    tokenizer.pad_token = "<|padding|>"
                    print(f"已设置默认pad_token: '{tokenizer.pad_token}'")
            except Exception as e:
                print(f"设置pad_token时出错: {e}，将使用临时解决方案")
    
    # 最后的确认检查
    print(f"create_trainer中最终配置 - pad_token: '{tokenizer.pad_token}', pad_token_id: {tokenizer.pad_token_id if hasattr(tokenizer, 'pad_token_id') else 'None'}, padding_side: {tokenizer.padding_side}")
    
    # 预处理函数 - 修改为不使用自动padding，而是在collator中处理
    def preprocess_function(examples):
        # 不在这里进行padding，只进行tokenization和truncation
        return tokenizer(
            examples["text"],
            max_length=config.training_config["max_seq_length"],
            truncation=True,
            padding=False  # 不在这里进行padding
        )
    
    # 创建自定义数据收集器来处理padding
    class CustomDataCollator:
        def __init__(self, tokenizer, max_length):
            self.tokenizer = tokenizer
            self.max_length = max_length
            
        def __call__(self, examples):
            # 获取所有input_ids
            input_ids = [example['input_ids'] for example in examples]
            padded_input_ids = []
            attention_masks = []
            labels = []
            
            # 手动进行padding，添加安全检查
            # 获取有效的pad_token_id，如果没有则使用0作为备用
            pad_token_id = getattr(self.tokenizer, 'pad_token_id', 0)
            if pad_token_id is None:
                pad_token_id = 0
                print("警告: tokenizer.pad_token_id为None，使用0作为备用")
            
            # 验证input_ids是否有效
            for i, ids in enumerate(input_ids):
                # 确保ids是列表类型而不是None
                if ids is None:
                    print(f"警告: 输入ID为None，索引: {i}")
                    input_ids[i] = []
                # 确保ids中的元素都是整数
                if ids and not all(isinstance(x, int) for x in ids):
                    print(f"警告: 发现非整数ID，索引: {i}")
                    # 尝试转换非整数为整数
                    input_ids[i] = [int(x) if x is not None else 0 for x in ids]
            
            # 进行padding处理
            for ids in input_ids:
                # 计算需要填充的长度
                padding_length = self.max_length - len(ids)
                if padding_length > 0:
                    # 进行padding，使用安全的pad_token_id
                    padded_ids = ids + [pad_token_id] * padding_length
                    # 创建attention mask，0表示padding部分
                    attention_mask = [1] * len(ids) + [0] * padding_length
                    # 为labels也添加padding，但通常padding部分不参与loss计算
                    label = ids + [-100] * padding_length  # -100是PyTorch中忽略loss的标记
                else:
                    padded_ids = ids[:self.max_length]
                    attention_mask = [1] * self.max_length
                    label = ids[:self.max_length]
                
                padded_input_ids.append(padded_ids)
                attention_masks.append(attention_mask)
                labels.append(label)
                
            # 创建批次
            batch = {
                'input_ids': torch.tensor(padded_input_ids),
                'attention_mask': torch.tensor(attention_masks),
                'labels': torch.tensor(labels)  # 使用处理后的labels
            }
            
            return batch
    
    # 使用自定义数据收集器
    data_collator = CustomDataCollator(tokenizer, config.training_config["max_seq_length"])
    
    # 处理数据集
    train_dataset = train_dataset.map(
        preprocess_function,
        batched=True,
        remove_columns=["text", "label"],
        num_proc=4 if not config.debug else 1
    )
    
    dev_dataset = dev_dataset.map(
        preprocess_function,
        batched=True,
        remove_columns=["text", "label"],
        num_proc=4 if not config.debug else 1
    )
    
    # 创建Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer
    )
    
    return trainer

def save_model(trainer, tokenizer, model, config):
    """保存训练好的模型"""
    print(f"\n保存模型到: {config.data_paths['model_dir']}...")
    
    # 如果使用了LoRA，只保存LoRA权重
    if config.model_config["use_lora"]:
        trainer.model.save_pretrained(config.data_paths["model_dir"])
        tokenizer.save_pretrained(config.data_paths["model_dir"])
    else:
        # 保存完整模型
        model.save_pretrained(config.data_paths["model_dir"])
        tokenizer.save_pretrained(config.data_paths["model_dir"])
    
    print("模型保存完成")

def predict_test_set(test_triples, mapper, tokenizer, model, all_entities, config):
    """在测试集上进行预测"""
    print("\n在测试集上进行预测...")
    
    predictions = []
    model.eval()
    device = next(model.parameters()).device
    prompt_generator = PromptGenerator()
    
    # 批次处理预测
    batch_size = config.prediction_config["batch_size"]
    top_k = config.prediction_config["top_k_candidates"]
    
    for i in tqdm(range(0, len(test_triples), batch_size), desc="处理测试批次"):
        batch_triples = test_triples[i:i+batch_size]
        batch_predictions = []
        
        for h, r, _ in batch_triples:
            # 获取中文文本
            h_text = mapper.get_entity_text(h)
            r_text = mapper.get_relation_text(r)
            
            # 生成预测prompt
            prompt = prompt_generator.generate_prediction_prompt(h_text, r_text)
            
            # 为每个测试样本生成候选实体
            # 注意：这里为了简化，我们直接从所有实体中随机选择top_k个作为候选
            # 实际应用中，应该使用更智能的方法（如知识图谱嵌入）来预筛选候选实体
            selected_entities = random.sample(all_entities, min(top_k, len(all_entities)))
            
            # 为每个候选实体生成完整的预测prompt
            candidate_prompts = []
            candidate_entity_ids = []
            
            for candidate_entity in selected_entities:
                candidate_text = mapper.get_entity_text(candidate_entity)
                full_prompt = prompt + candidate_text
                candidate_prompts.append(full_prompt)
                candidate_entity_ids.append(candidate_entity)
            
            # 批量处理候选实体
            inputs = tokenizer(
                candidate_prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=config.training_config["max_seq_length"]
            ).to(device)
            
            # 计算每个候选实体的概率
            with torch.no_grad():
                outputs = model(**inputs)
                # 取最后一个token的logits
                logits = outputs.logits[:, -1, :]
                # 计算概率
                probs = torch.softmax(logits, dim=-1)
                # 取结束token的概率作为候选实体的得分
                eos_token_id = tokenizer.eos_token_id
                eos_probs = probs[:, eos_token_id].cpu().numpy()
            
            # 选择得分最高的实体作为预测结果
            best_idx = np.argmax(eos_probs)
            best_entity = candidate_entity_ids[best_idx]
            
            batch_predictions.append((h, r, best_entity))
        
        # 过滤掉不在已知实体集合中的预测结果
        filtered_predictions = filter_unknown_entities(batch_predictions, set(all_entities))
        predictions.extend(filtered_predictions)
    
    print(f"预测完成，共处理 {len(predictions)} 个测试样本")
    return predictions

def save_predictions(predictions, output_dir):
    """保存预测结果"""
    output_file = os.path.join(output_dir, "test_predictions.tsv")
    with open(output_file, 'w', encoding='utf-8') as f:
        for h, r, t_pred in predictions:
            f.write(f"{h}\t{r}\t{t_pred}\n")
    
    print(f"预测结果已保存到: {output_file}")

def run_training(config):
    """执行训练流程"""
    print("\n===== 开始训练流程 ====")
    
    # 1. 加载数据
    print("\n1. 加载数据...")
    data_loader = KnowledgeGraphDataLoader(
        config.data_paths["train_file"],
        config.data_paths["test_file"],
        config.data_paths["dev_file"]
    )
    
    # 2. 加载实体和关系映射
    print("\n2. 加载实体和关系映射...")
    mapper = EntityRelationMapper(
        config.data_paths["entity_text_file"],
        config.data_paths["relation_text_file"]
    )
    
    # 3. 准备训练数据
    print("\n3. 准备训练数据...")
    train_dataset, dev_dataset, all_entities = prepare_training_data(config, data_loader, mapper)
    
    # 4. 加载模型和分词器
    print("\n4. 加载模型和分词器...")
    tokenizer, model = load_model_and_tokenizer(config)
    
    # 5. 设置训练参数
    print("\n5. 设置训练参数...")
    training_args = setup_training_args(config)
    
    # 6. 创建Trainer
    print("\n6. 创建Trainer...")
    trainer = create_trainer(model, tokenizer, training_args, train_dataset, dev_dataset, config)
    
    # 7. 开始训练
    print("\n7. 开始训练...")
    start_time = time.time()
    trainer.train()
    end_time = time.time()
    print(f"训练完成，耗时: {(end_time - start_time) / 3600:.2f} 小时")
    
    # 8. 保存模型
    print("\n8. 保存模型...")
    save_model(trainer, tokenizer, model, config)
    
    # 9. 在测试集上进行预测
    print("\n9. 在测试集上进行预测...")
    predictions = predict_test_set(data_loader.test_triples, mapper, tokenizer, model, all_entities, config)
    
    # 10. 保存预测结果
    print("\n10. 保存预测结果...")
    save_predictions(predictions, config.data_paths["output_dir"])
    
    print("\n===== 训练流程完成 ====")

def run_evaluation(config):
    """执行评估流程"""
    print("\n===== 开始评估流程 ====")
    
    # 加载数据
    data_loader = KnowledgeGraphDataLoader(
        config.data_paths["train_file"],
        config.data_paths["test_file"],
        config.data_paths["dev_file"]
    )
    
    # 加载实体和关系映射
    mapper = EntityRelationMapper(
        config.data_paths["entity_text_file"],
        config.data_paths["relation_text_file"]
    )
    
    # 收集所有实体
    all_entities = list(data_loader.collect_all_entities())
    
    # 配置Hugging Face国内镜像源和下载优化
    import os
    # 设置Hugging Face镜像源
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
    # 启用并行下载和分片下载
    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"  # 启用hf_transfer加速
    os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"  # 禁用进度条减少开销
    os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "300"  # 增加下载超时时间
    os.environ["HF_HUB_ETAG_TIMEOUT"] = "30"  # 增加ETAG超时时间
    # 内存优化配置
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,max_split_size_mb:128"
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    os.environ["PYTORCH_NO_CUDA_MEMORY_CACHING"] = "0"
    
    # 加载模型和分词器 - 使用本地已下载的模型
    print("\n加载模型...")
    local_model_path = "/mnt/d/forCoding_data/Tianchi_EcommerceKG/trained_models/models--Qwen--Qwen-1_8B-Chat"
    print(f"使用本地模型路径: {local_model_path}")
    
    # 验证本地模型路径存在性
    import os
    config_json_path = os.path.join(local_model_path, "config.json")
    tokenizer_config_path = os.path.join(local_model_path, "tokenizer_config.json")
    
    print(f"检查关键文件:\n- config.json: {os.path.exists(config_json_path)}\n- tokenizer_config.json: {os.path.exists(tokenizer_config_path)}")
    
    # 改进的分词器加载逻辑
    if not os.path.exists(config_json_path):
        print("警告: 未找到完整的配置文件，尝试使用原始模型名称进行加载...")
        tokenizer = AutoTokenizer.from_pretrained(
            "Qwen/Qwen-1_8B-Chat",  # 使用原始模型名称
            trust_remote_code=True,
            local_files_only=True,  # 强制使用本地文件
            cache_dir="/mnt/d/forCoding_data/Tianchi_EcommerceKG/trained_models",
            revision="main"
        )
        model = AutoModelForCausalLM.from_pretrained(
            "Qwen/Qwen-1_8B-Chat",  # 使用原始模型名称
            device_map="auto",
            torch_dtype=torch.float16,
            trust_remote_code=True,
            local_files_only=True,  # 强制使用本地文件
            cache_dir="/mnt/d/forCoding_data/Tianchi_EcommerceKG/trained_models"
        )
    else:
        # 使用本地路径加载
        tokenizer = AutoTokenizer.from_pretrained(
            local_model_path,
            trust_remote_code=True,
            local_files_only=True,  # 强制使用本地文件
            cache_dir="/mnt/d/forCoding_data/Tianchi_EcommerceKG/trained_models"
        )
        model = AutoModelForCausalLM.from_pretrained(
            local_model_path,
            device_map="auto",
            torch_dtype=torch.float16,
            trust_remote_code=True,
            local_files_only=True,  # 强制使用本地文件
            cache_dir="/mnt/d/forCoding_data/Tianchi_EcommerceKG/trained_models"
        )
    
    # 在开发集上进行评估
    print("\n在开发集上进行评估...")
    # 注意：这里简化了评估过程，实际应用中应该实现更详细的评估指标计算
    dev_triples_with_t = [(h, r, t) for h, r, t in data_loader.dev_triples if t is not None]
    predictions = predict_test_set(dev_triples_with_t, mapper, tokenizer, model, all_entities, config)
    
    # 计算简单的准确率（仅供参考）
    correct = 0
    for (h, r, t_pred), (_, _, t_true) in zip(predictions, dev_triples_with_t):
        if t_pred == t_true:
            correct += 1
    
    accuracy = correct / len(predictions) if predictions else 0
    print(f"评估准确率: {accuracy:.4f}")
    
    # 保存评估结果
    eval_result = {
        "accuracy": accuracy,
        "total_samples": len(predictions),
        "correct_predictions": correct
    }
    
    with open(os.path.join(config.data_paths["output_dir"], "evaluation_result.json"), 'w', encoding='utf-8') as f:
        json.dump(eval_result, f, ensure_ascii=False, indent=2)
    
    print("\n===== 评估流程完成 ====")

def run_prediction(config):
    """执行预测流程"""
    print("\n===== 开始预测流程 ====")
    
    # 加载数据
    data_loader = KnowledgeGraphDataLoader(
        config.data_paths["train_file"],
        config.data_paths["test_file"],
        config.data_paths["dev_file"]
    )
    
    # 加载实体和关系映射
    mapper = EntityRelationMapper(
        config.data_paths["entity_text_file"],
        config.data_paths["relation_text_file"]
    )
    
    # 收集所有实体
    all_entities = list(data_loader.collect_all_entities())
    
    # 配置Hugging Face国内镜像源和下载优化
    import os
    # 设置Hugging Face镜像源
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
    # 启用并行下载和分片下载
    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"  # 启用hf_transfer加速
    os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"  # 禁用进度条减少开销
    os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "300"  # 增加下载超时时间
    os.environ["HF_HUB_ETAG_TIMEOUT"] = "30"  # 增加ETAG超时时间
    # 内存优化配置
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,max_split_size_mb:128"
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    os.environ["PYTORCH_NO_CUDA_MEMORY_CACHING"] = "0"
    
    # 加载模型和分词器
    print("\n加载模型...")
    tokenizer = AutoTokenizer.from_pretrained(config.data_paths["model_dir"], trust_remote_code=True, mirror="tuna", cache_dir="/mnt/d/forCoding_data/Tianchi_EcommerceKG/trained_models")
    model = AutoModelForCausalLM.from_pretrained(
        config.data_paths["model_dir"],
        device_map="auto",
        torch_dtype=torch.float16,
        trust_remote_code=True,
        cache_dir="/mnt/d/forCoding_data/Tianchi_EcommerceKG/trained_models"
    )
    
    # 在测试集上进行预测
    print("\n在测试集上进行预测...")
    predictions = predict_test_set(data_loader.test_triples, mapper, tokenizer, model, all_entities, config)
    
    # 保存预测结果
    save_predictions(predictions, config.data_paths["output_dir"])
    
    print("\n===== 预测流程完成 ====")

def main():
    """主函数"""
    # 解析命令行参数
    args = parse_arguments()
    
    # 加载配置
    if os.path.exists(args.config):
        config = load_config(args.config)
    else:
        print(f"配置文件不存在: {args.config}，使用默认配置")
        config = get_config()
    
    # 使用命令行参数更新配置
    config = update_config_with_args(config, args)
    
    # 设置调试模式
    if args.debug:
        print("\n启用调试模式")
        config.debug = True
    else:
        config.debug = False
    
    # 打印配置摘要
    print_config_summary(config)
    
    # 验证配置
    valid, errors = validate_config(config)
    if not valid:
        print("\n配置验证失败:")
        for error in errors:
            print(f"- {error}")
        sys.exit(1)
    
    # 保存配置（如果需要）
    if args.save_config:
        save_config(config, args.config)
        print("\n配置已保存")
        return
    
    # 设置随机种子并应用cuDNN优化配置
    set_seed(config.data_processing_config["seed"], config)
    
    # 根据模式执行相应的流程
    if args.mode == "train":
        run_training(config)
    elif args.mode == "eval":
        run_evaluation(config)
    elif args.mode == "predict":
        run_prediction(config)
    else:
        print(f"不支持的模式: {args.mode}")
        sys.exit(1)

if __name__ == "__main__":
    main()

# 导入pandas用于数据处理
import pandas as pd