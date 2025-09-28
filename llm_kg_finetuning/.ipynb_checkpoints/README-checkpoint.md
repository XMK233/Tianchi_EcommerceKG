# LLM知识图谱预测器

基于大型语言模型(LLM)的知识图谱尾实体预测系统，通过自然语言处理和参数高效微调技术，利用实体和关系的中文文本含义来提升预测性能。

## 项目概述

本项目实现了一种基于大型语言模型的知识图谱预测方案，主要特点包括：

- 将实体和关系的代码映射为自然语言含义
- 使用LoRA技术进行参数高效微调
- 集成DeepSpeed和量化技术提升训练和推理速度
- 确保预测结果仅包含已知实体
- 提供完整的训练、评估和预测流程

## 项目结构

```
llm_kg_finetuning/
├── config.py                # 配置管理模块
├── data_processor.py        # 数据处理工具
├── llm_kg_predictor.py      # 主预测器实现
├── train.py                 # 训练和预测脚本
├── requirements.txt         # 项目依赖
└── README.md                # 项目说明文档
```

## 环境配置

### 安装依赖

首先安装项目所需的依赖包：

```bash
cd llm_kg_finetuning
pip install -r requirements.txt
```

### DeepSpeed安装说明

项目中DeepSpeed依赖已被注释掉，因为它需要CUDA环境才能编译。如果您的环境中已配置好CUDA，可以手动安装DeepSpeed以获得更好的训练性能：

```bash
# 确保已配置CUDA环境
pip install deepspeed>=0.9.0
```

#### 替代方案

如果您没有CUDA环境或无法安装DeepSpeed，系统会自动使用以下替代优化策略：

1. 梯度累积以模拟更大批次
2. 混合精度训练(fp16)
3. 梯度检查点节省内存
4. 可通过Accelerate库的环境变量配置分布式训练

## 数据准备

项目需要以下数据文件：

1. 知识图谱训练集、测试集和开发集（TSV格式）
2. 实体和关系的中文文本映射文件（TSV格式）

请确保在`config.py`中正确配置这些文件的路径。

## 配置说明

项目的主要配置集中在`config.py`文件中，包括：

### 数据路径配置

```python
# 数据路径配置
data_paths = {
    "train_file": "/path/to/train.tsv",
    "test_file": "/path/to/test.tsv",
    "dev_file": "/path/to/dev.tsv",
    "entity_text_file": "/path/to/entity2text.tsv",
    "relation_text_file": "/path/to/relation2text.tsv",
    "output_dir": "/mnt/d/forCoding_data/Tianchi_EcommerceKG/processedData/results",
    "model_dir": "/mnt/d/forCoding_data/Tianchi_EcommerceKG/trained_models"
}
```

### 模型配置

```python
# 模型配置
model_config = {
    "model_name": "THUDM/chatglm3-6b-base",  # 开源中文模型，无需授权访问
    "use_quantization": True,                   # 是否启用量化
    "quantization_bits": 4,                     # 量化位数（4或8）
    "use_lora": True,                           # 是否使用LoRA
    "lora_rank": 8,                             # LoRA秩
    "lora_alpha": 16,                           # LoRA缩放因子
    "lora_dropout": 0.1,                        # LoRA层dropout率
    "target_modules": ["q_proj", "v_proj"],   # LoRA应用的目标模块
    "use_deepspeed": False,                     # 是否使用DeepSpeed（默认禁用：需要CUDA环境）
    "gradient_checkpointing": True              # 是否使用梯度检查点
}
```

### 训练配置

```python
# 训练配置
training_config = {
    "per_device_train_batch_size": 4,    # 每个设备的训练批次大小
    "per_device_eval_batch_size": 4,     # 每个设备的评估批次大小
    "gradient_accumulation_steps": 8,    # 梯度累积步数
    "learning_rate": 2e-5,               # 学习率
    "num_train_epochs": 3,               # 训练轮数
    "warmup_ratio": 0.1,                 # 学习率预热比例
    "weight_decay": 0.01,                # 权重衰减
    "save_steps": 500,                   # 保存步数
    "logging_steps": 100,                # 日志记录步数
    "max_seq_length": 200,               # 最大序列长度
    "fp16": True                         # 是否使用混合精度训练
}
```

### 预测配置

```python
# 预测配置
prediction_config = {
    "top_k_candidates": 10,              # 预测时考虑的候选实体数量
    "batch_size": 16                     # 预测批次大小
}
```

## 使用方法

### 训练模型

使用`train.py`脚本进行模型训练：

```bash
python train.py --mode train
```

可以通过命令行参数覆盖配置文件中的设置：

```bash
python train.py --mode train --model-name baichuan-inc/Baichuan-7B --use-quantization --use-lora --batch-size 8 --learning-rate 1e-5
```

### 评估模型

在开发集上评估模型性能：

```bash
python train.py --mode eval
```

### 生成预测

在测试集上生成预测结果：

```bash
python train.py --mode predict
```

## 参数说明

`train.py`脚本支持以下命令行参数：

### 配置文件参数

- `--config`: 配置文件路径（默认：`./config.json`）
- `--save-config`: 保存当前配置到文件

### 模式选择

- `--mode`: 运行模式，可选值：`train`（训练）、`eval`（评估）、`predict`（预测）

### 数据路径参数

- `--train-file`: 训练集文件路径
- `--test-file`: 测试集文件路径
- `--dev-file`: 开发集文件路径
- `--entity-text-file`: 实体文本映射文件路径
- `--relation-text-file`: 关系文本映射文件路径
- `--output-dir`: 输出目录
- `--model-dir`: 模型保存目录

### 模型参数

- `--model-name`: 预训练模型名称或路径
- `--use-quantization`/`--no-use-quantization`: 启用/禁用模型量化
- `--quantization-bits`: 量化位数（4或8）
- `--use-lora`/`--no-use-lora`: 启用/禁用LoRA参数高效微调
- `--lora-rank`: LoRA秩

### 训练参数

- `--batch-size`: 训练批次大小
- `--learning-rate`: 学习率
- `--num-epochs`: 训练轮数
- `--gradient-accumulation-steps`: 梯度累积步数

### 预测参数

- `--top-k`: 预测时考虑的候选实体数量

### 其他选项

- `--debug`: 启用调试模式
- `--use-deepspeed`/`--no-use-deepspeed`: 启用/禁用DeepSpeed加速

## 项目组件说明

### 1. 数据处理模块 (`data_processor.py`)

- `EntityRelationMapper`: 管理实体和关系的中文文本映射
- `KnowledgeGraphDataLoader`: 加载和处理知识图谱数据
- `PromptGenerator`: 生成用于LLM微调的prompt
- `NegativeSampler`: 生成负样本用于训练
- `ChineseTextProcessor`: 处理中文文本

### 2. 配置管理模块 (`config.py`)

- `Config`类: 存储和管理项目配置
- 配置验证和保存功能

### 3. 主预测器实现 (`llm_kg_predictor.py`)

- 集成了数据处理、模型加载、训练和预测的完整流程
- 支持量化、LoRA和DeepSpeed优化

### 4. 训练和预测脚本 (`train.py`)

- 提供命令行接口
- 支持训练、评估和预测模式
- 整合所有功能模块

## 性能优化策略

本项目采用了多种性能优化策略：

1. **量化技术**: 使用4位或8位量化减少模型内存占用
2. **参数高效微调**: 使用LoRA技术，只训练少量参数
3. **DeepSpeed**: 利用DeepSpeed框架加速训练和减少内存使用
4. **梯度检查点**: 进一步减少训练时的内存消耗
5. **批量处理**: 批量处理预测请求，提高推理效率

## 预测结果

预测结果将保存在配置的输出目录中，默认路径为`./results/test_predictions.tsv`，格式为：

```
头实体ID\t关系ID\t预测的尾实体ID
```

## 注意事项和建议

1. **内存需求**: 即使使用量化和LoRA，仍然建议在具有足够GPU内存的环境中运行（至少10GB显存）
2. **模型选择**: 根据实际需求选择合适的预训练模型，中文任务可考虑使用Baichuan、GLM等中文优化模型
3. **参数调整**: 根据硬件环境和数据集大小调整批次大小、学习率等超参数
4. **预测效率**: 对于大规模知识图谱，可以进一步优化候选实体的生成策略，提高预测速度
5. **结果过滤**: 系统会自动过滤掉不在已知实体集合中的预测结果，确保预测的可靠性

## 扩展方向

1. 集成更多类型的预训练模型
2. 实现更复杂的候选实体生成策略
3. 增加更多评估指标
4. 支持实体链接和关系抽取等扩展任务
5. 开发Web界面方便使用

## 许可证

本项目采用MIT许可证。

## 联系方式

如有问题或建议，请联系项目维护者。