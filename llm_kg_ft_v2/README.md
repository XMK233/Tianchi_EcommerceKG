# LLM-KG-FT-V2: 知识图谱增强的大语言模型微调框架

本项目实现了一个基于知识图谱(Knowledge Graph)的大语言模型(LLM)微调框架，旨在通过知识图谱中的三元组信息增强大语言模型的知识推理能力。

## 代码功能概述

本框架主要实现了以下功能：

1. 知识图谱数据处理与预处理
2. 负采样机制实现
3. 基于LoRA的参数高效微调
4. 自定义数据收集器确保训练稳定性
5. 支持DeepSpeed加速训练
6. 完整的评估和预测流程

## 项目结构

```
llm_kg_ft_v2/
├── config.py        # 项目配置参数
├── data_processor.py # 数据处理模块
├── trainer.py       # 模型训练和预测模块
├── main.py          # 主程序入口
└── README.md        # 项目说明文档
```

## 数据获取与预处理

### 数据集

本项目使用OpenBG500数据集进行训练和评估，数据集文件配置在`config.py`中：

```python
# config.py中的数据集配置
dataset_dir = os.path.join(project_dir, "data", "OpenBG500")
train_file = os.path.join(dataset_dir, "train.txt")
dev_file = os.path.join(dataset_dir, "valid.txt")
test_file = os.path.join(dataset_dir, "test.txt")
```

### 数据预处理流程

数据预处理主要在`KGDataProcessor`类中实现，包括以下步骤：

1. **实体和关系映射加载**：
   - 从数据集文件中加载实体ID到文本的映射
   - 从数据集文件中加载关系ID到文本的映射

2. **三元组数据加载**：
   - 读取训练、验证和测试数据中的三元组(head, relation, tail)

3. **负采样处理**：
   - 实现并行和串行两种负采样方法
   - 为每个正样本生成多个负样本

4. **Prompt构建**：
   - 根据三元组信息构建适合大语言模型输入的文本提示
   - 例如："已知头实体是[头实体文本]，关系是[关系文本]，那么尾实体是什么？"

5. **训练数据准备**：
   - 结合正样本和负样本生成最终的训练数据
   - 为每个样本添加标签以进行监督学习

## 数据采样方法

项目在数据采样方面实现了以下策略：

1. **训练集采样限制**：
   - 在`trainer.py`的`train`方法中限制采样10000行数据进行训练
   - 这是为了控制训练规模和时间

2. **动态进程配置**：
   - 根据运行环境自动调整数据处理的进程数
   - 在非CUDA环境下最多使用8个进程，以避免资源过度消耗

```python
# trainer.py中的动态进程配置
if not torch.cuda.is_available():
    num_proc = min(8, multiprocessing.cpu_count())
    logger.info(f"CUDA不可用，使用多进程加速: {num_proc}个进程")
```

## 预训练模型使用

本项目使用字节跳动的Qwen-1_8B-Chat模型作为基础模型：

```python
# config.py中的模型配置
model_name_or_path = "Qwen/Qwen-1_8B-Chat"
model_cache_dir = os.path.join(project_dir, "model_cache")
```

主要特点：
- 支持4bit量化以减少显存占用
- 实现了安全的tokenizer配置，解决了Qwen模型特有的pad_token_id问题

## 微调结构设计

本项目采用LoRA(Low-Rank Adaptation)技术进行参数高效微调，具体设计如下：

### LoRA配置

```python
# trainer.py中的LoRA配置
lora_config = LoraConfig(
    r=config.lora_rank,  # LoRA的秩，控制可训练参数的数量
    lora_alpha=config.lora_alpha,  # LoRA的缩放因子
    target_modules=["c_attn"],  # 针对Qwen模型的注意力模块
    lora_dropout=config.lora_dropout,
    bias="none",
    task_type="CAUSAL_LM"
)
```

### 训练优化器配置

项目中配置了AdamW优化器和学习率调度器，同时使用DeepSpeed进行训练加速：

```python
# config.py中的训练参数配置
learning_rate = 2e-5
weight_decay = 0.01
warmup_ratio = 0.1
batch_size = 4
num_epochs = 3
```

### 安全数据收集器

为解决训练过程中的张量长度不匹配问题，项目实现了自定义的`SafeDataCollator`类：

- 自动检测批次中最长的序列长度
- 对所有样本进行统一的padding处理
- 确保input_ids、attention_mask和labels张量长度一致
- 支持ignore_pad_token_for_loss参数，避免在计算损失时考虑padding值

## 训练和预测开展方式

### 训练流程

训练流程主要在`KGTrainer`类的`train`方法中实现：

1. **模型和分词器加载**：
   - 加载预训练的Qwen模型
   - 配置分词器，解决pad_token_id问题
   - 应用LoRA适配器

2. **数据集处理**：
   - 加载训练数据
   - 应用tokenize_function进行分词
   - 使用map操作进行并行处理

3. **训练参数配置**：
   - 设置TrainingArguments
   - 配置DeepSpeed参数
   - 应用混合精度训练(fp16)

4. **训练执行**：
   - 使用HuggingFace的Trainer API进行训练
   - 保存训练好的LoRA模型

### 评估流程

评估流程在`evaluate`方法中实现：

1. 使用beam search进行文本生成，提高生成质量
2. 计算模型在开发集上的准确率
3. 记录评估结果

### 预测流程

预测流程在`predict`方法中实现：

1. 对测试集中的每个三元组进行预测
2. 提取预测的尾实体文本
3. 将预测结果保存到输出文件

### 主程序流程

`main.py`整合了以上所有流程：

1. 设置随机种子，保证实验可重复性
2. 检查CUDA可用性
3. 创建数据处理器和训练器实例
4. 根据用户选择决定是否重新训练模型
5. 在开发集上评估模型
6. 在测试集上进行预测并保存结果

## 运行环境要求

- Python 3.8+
- PyTorch
- Transformers
- PEFT
- DeepSpeed
- CUDA环境(推荐，用于加速训练)

## 运行方式

直接运行主程序即可启动整个训练和评估流程：

```bash
python main.py
```

程序会自动：
1. 检查是否存在已训练的模型
2. 根据用户选择进行模型训练或加载已有模型
3. 在开发集上评估模型性能
4. 在测试集上进行预测并保存结果

## 输出结果

训练完成后，会生成以下输出：

1. 训练好的LoRA模型(保存在`output_dir/lora_model`目录下)
2. 测试集预测结果(保存在`output_dir/test_predictions.tsv`文件中)
3. 评估指标报告(包括开发集准确率)