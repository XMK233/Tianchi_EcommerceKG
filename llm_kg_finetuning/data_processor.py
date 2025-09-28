import os
import sys
import json
import pandas as pd
import numpy as np
import random
import jieba
from tqdm import tqdm
from typing import List, Tuple, Dict, Set, Optional

class EntityRelationMapper:
    """实体和关系的中文文本映射管理器"""
    def __init__(self, entity_text_file: str, relation_text_file: str):
        """初始化映射管理器"""
        self.entity_text_file = entity_text_file
        self.relation_text_file = relation_text_file
        self.entity_to_text: Dict[str, str] = {}
        self.text_to_entity: Dict[str, str] = {}
        self.relation_to_text: Dict[str, str] = {}
        self.text_to_relation: Dict[str, str] = {}
        
        # 加载映射
        self._load_mappings()
        
    def _load_mappings(self) -> None:
        """加载实体和关系的中文文本映射"""
        # 加载实体映射
        self.entity_to_text = self._load_text_mapping(self.entity_text_file)
        self.text_to_entity = {v: k for k, v in self.entity_to_text.items()}
        print(f"已加载 {len(self.entity_to_text)} 个实体文本映射")
        
        # 加载关系映射
        self.relation_to_text = self._load_text_mapping(self.relation_text_file)
        self.text_to_relation = {v: k for k, v in self.relation_to_text.items()}
        print(f"已加载 {len(self.relation_to_text)} 个关系文本映射")
    
    def _load_text_mapping(self, file_path: str) -> Dict[str, str]:
        """加载文本映射文件"""
        mapping = {}
        try:
            if not os.path.exists(file_path):
                print(f"警告: 文件不存在: {file_path}")
                return mapping
                
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    parts = line.strip().split('\t')
                    if len(parts) >= 2:
                        entity_id = parts[0]
                        entity_text = parts[1]
                        mapping[entity_id] = entity_text
        except Exception as e:
            print(f"加载文本映射文件出错: {e}")
        return mapping
    
    def get_entity_text(self, entity_id: str) -> str:
        """根据实体ID获取中文文本"""
        return self.entity_to_text.get(entity_id, entity_id)
    
    def get_relation_text(self, relation_id: str) -> str:
        """根据关系ID获取中文文本"""
        return self.relation_to_text.get(relation_id, relation_id)
    
    def get_entity_id(self, entity_text: str) -> Optional[str]:
        """根据中文文本获取实体ID"""
        return self.text_to_entity.get(entity_text)
    
    def get_relation_id(self, relation_text: str) -> Optional[str]:
        """根据中文文本获取关系ID"""
        return self.text_to_relation.get(relation_text)
    
    def save_mappings(self, output_dir: str) -> None:
        """保存映射到文件"""
        os.makedirs(output_dir, exist_ok=True)
        
        # 保存实体映射
        entity_output_file = os.path.join(output_dir, "entity_mappings.json")
        with open(entity_output_file, 'w', encoding='utf-8') as f:
            json.dump(self.entity_to_text, f, ensure_ascii=False, indent=2)
        
        # 保存关系映射
        relation_output_file = os.path.join(output_dir, "relation_mappings.json")
        with open(relation_output_file, 'w', encoding='utf-8') as f:
            json.dump(self.relation_to_text, f, ensure_ascii=False, indent=2)
        
        print(f"映射已保存到: {output_dir}")

class KnowledgeGraphDataLoader:
    """知识图谱数据加载器"""
    def __init__(self, train_file: str, test_file: str, dev_file: str):
        """初始化数据加载器"""
        self.train_file = train_file
        self.test_file = test_file
        self.dev_file = dev_file
        
        # 加载数据
        self.train_triples: List[Tuple[str, str, str]] = []
        self.test_triples: List[Tuple[str, str, Optional[str]]] = []
        self.dev_triples: List[Tuple[str, str, str]] = []
        
        self._load_data()
        
    def _load_data(self) -> None:
        """加载训练、测试和开发集数据"""
        # 加载训练集
        self.train_triples = self._load_triples(self.train_file, is_test=False)
        print(f"已加载训练集三元组: {len(self.train_triples)}")
        
        # 加载测试集
        self.test_triples = self._load_triples(self.test_file, is_test=True)
        print(f"已加载测试集三元组: {len(self.test_triples)}")
        
        # 加载开发集
        self.dev_triples = self._load_triples(self.dev_file, is_test=False)
        print(f"已加载开发集三元组: {len(self.dev_triples)}")
    
    def _load_triples(self, file_path: str, is_test: bool = False) -> List[Tuple[str, str, Optional[str]]]:
        """加载三元组数据"""
        triples = []
        try:
            if not os.path.exists(file_path):
                print(f"警告: 文件不存在: {file_path}")
                return triples
                
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    parts = line.strip().split('\t')
                    if is_test:
                        if len(parts) >= 2:
                            h, r = parts[0], parts[1]
                            triples.append((h, r, None))
                    else:
                        if len(parts) >= 3:
                            h, r, t = parts[0], parts[1], parts[2]
                            triples.append((h, r, t))
        except Exception as e:
            print(f"加载三元组数据出错: {e}")
        return triples
    
    def collect_all_entities(self) -> Set[str]:
        """收集所有实体"""
        entities = set()
        
        for h, r, t in self.train_triples:
            entities.add(h)
            if t is not None:
                entities.add(t)
        
        for h, r, t in self.dev_triples:
            entities.add(h)
            if t is not None:
                entities.add(t)
        
        for h, r, _ in self.test_triples:
            entities.add(h)
        
        print(f"总共收集到 {len(entities)} 个唯一实体")
        return entities
    
    def collect_all_relations(self) -> Set[str]:
        """收集所有关系"""
        relations = set()
        
        for h, r, _ in self.train_triples:
            relations.add(r)
        
        for h, r, _ in self.dev_triples:
            relations.add(r)
        
        for h, r, _ in self.test_triples:
            relations.add(r)
        
        print(f"总共收集到 {len(relations)} 个唯一关系")
        return relations
    
    def get_stats(self) -> Dict[str, int]:
        """获取数据集统计信息"""
        stats = {
            "train_triples": len(self.train_triples),
            "test_triples": len(self.test_triples),
            "dev_triples": len(self.dev_triples),
            "unique_entities": len(self.collect_all_entities()),
            "unique_relations": len(self.collect_all_relations())
        }
        return stats
    
    def save_stats(self, output_file: str) -> None:
        """保存数据集统计信息"""
        stats = self.get_stats()
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)
        print(f"数据集统计信息已保存到: {output_file}")

class PromptGenerator:
    """用于生成LLM微调的prompt"""
    def __init__(self):
        """初始化prompt生成器"""
        self.prompt_templates = {
            "positive": "input是`头实体是{h_text}，关系是{r_text}`，output是`正确的尾实体是{t_text}`",
            "negative": "input是`头实体是{h_text}，关系是{r_text}`，output是`错误的尾实体是{t_text}`",
            "prediction": "input是`头实体是{h_text}，关系是{r_text}`，output是`正确的尾实体是"
        }
    
    def generate_positive_prompt(self, h_text: str, r_text: str, t_text: str) -> str:
        """生成正样本prompt"""
        return self.prompt_templates["positive"].format(h_text=h_text, r_text=r_text, t_text=t_text)
    
    def generate_negative_prompt(self, h_text: str, r_text: str, t_text: str) -> str:
        """生成负样本prompt"""
        return self.prompt_templates["negative"].format(h_text=h_text, r_text=r_text, t_text=t_text)
    
    def generate_prediction_prompt(self, h_text: str, r_text: str) -> str:
        """生成预测用的prompt"""
        return self.prompt_templates["prediction"].format(h_text=h_text, r_text=r_text)
    
    def set_template(self, template_type: str, template: str) -> None:
        """设置自定义prompt模板"""
        if template_type in self.prompt_templates:
            self.prompt_templates[template_type] = template
            print(f"已更新 {template_type} 模板")
        else:
            print(f"无效的模板类型: {template_type}")

class NegativeSampler:
    """负采样器"""
    def __init__(self, all_entities: List[str]):
        """初始化负采样器"""
        self.all_entities = all_entities
    
    def sample_negative(self, positive_entity: str, num_samples: int = 1) -> List[str]:
        """采样不同于正样本的负实体"""
        negative_entities = []
        for _ in range(num_samples):
            # 随机选择一个不同于正样本的实体
            neg_entity = random.choice(self.all_entities)
            while neg_entity == positive_entity:
                neg_entity = random.choice(self.all_entities)
            negative_entities.append(neg_entity)
        return negative_entities
    
    def sample_batch_negatives(self, positive_entities: List[str], num_samples_per_positive: int = 1) -> List[List[str]]:
        """批量采样负样本"""
        batch_negatives = []
        for pos_entity in positive_entities:
            negatives = self.sample_negative(pos_entity, num_samples_per_positive)
            batch_negatives.append(negatives)
        return batch_negatives

class ChineseTextProcessor:
    """中文文本处理器"""
    def __init__(self):
        """初始化中文文本处理器"""
        # 加载停用词（可选）
        self.stopwords = set()
        
    def load_stopwords(self, stopwords_file: str) -> None:
        """加载停用词表"""
        try:
            if os.path.exists(stopwords_file):
                with open(stopwords_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        self.stopwords.add(line.strip())
                print(f"已加载 {len(self.stopwords)} 个停用词")
        except Exception as e:
            print(f"加载停用词表出错: {e}")
    
    def tokenize(self, text: str) -> List[str]:
        """分词处理"""
        if not text or not isinstance(text, str):
            return []
        
        # 使用jieba分词
        tokens = jieba.lcut(text)
        
        # 过滤停用词
        if self.stopwords:
            tokens = [token for token in tokens if token not in self.stopwords]
        
        return tokens
    
    def clean_text(self, text: str) -> str:
        """清理文本"""
        if not text or not isinstance(text, str):
            return ""
        
        # 去除多余空格
        text = ' '.join(text.split())
        
        # 去除特殊字符（保留中文、英文、数字）
        import re
        text = re.sub(r'[^一-龥a-zA-Z0-9]', ' ', text)
        
        # 再次去除多余空格
        text = ' '.join(text.split())
        
        return text
    
    def normalize(self, text: str) -> str:
        """文本标准化"""
        # 转小写
        text = text.lower()
        
        # 清理文本
        text = self.clean_text(text)
        
        return text

# 工具函数

def create_data_statistics(train_triples: List[Tuple[str, str, str]], 
                           dev_triples: List[Tuple[str, str, str]], 
                           test_triples: List[Tuple[str, str, Optional[str]]],
                           output_file: str) -> None:
    """创建数据集统计信息"""
    # 收集实体和关系
    all_entities = set()
    all_relations = set()
    
    for h, r, t in train_triples:
        all_entities.add(h)
        if t is not None:
            all_entities.add(t)
        all_relations.add(r)
    
    for h, r, t in dev_triples:
        all_entities.add(h)
        if t is not None:
            all_entities.add(t)
        all_relations.add(r)
    
    for h, r, _ in test_triples:
        all_entities.add(h)
        all_relations.add(r)
    
    # 创建统计信息
    stats = {
        "train_triples": len(train_triples),
        "dev_triples": len(dev_triples),
        "test_triples": len(test_triples),
        "total_triples": len(train_triples) + len(dev_triples) + len(test_triples),
        "unique_entities": len(all_entities),
        "unique_relations": len(all_relations),
        "data_distribution": {
            "train": len(train_triples) / (len(train_triples) + len(dev_triples) + len(test_triples)) * 100,
            "dev": len(dev_triples) / (len(train_triples) + len(dev_triples) + len(test_triples)) * 100,
            "test": len(test_triples) / (len(train_triples) + len(dev_triples) + len(test_triples)) * 100
        }
    }
    
    # 保存到文件
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    
    print(f"数据集统计信息已保存到: {output_file}")
    return stats

def filter_unknown_entities(predictions: List[Tuple[str, str, str]], 
                             all_entities: Set[str]) -> List[Tuple[str, str, str]]:
    """过滤掉不在已知实体集合中的预测结果"""
    filtered_predictions = []
    
    for h, r, t_pred in predictions:
        # 如果预测的实体在已知实体集合中，则保留
        if t_pred in all_entities:
            filtered_predictions.append((h, r, t_pred))
        else:
            # 否则，替换为默认值或其他处理方式
            # 这里简单替换为空字符串，实际应用中可以有更复杂的处理逻辑
            filtered_predictions.append((h, r, ""))
    
    # 统计被过滤的数量
    original_count = len(predictions)
    filtered_count = len(filtered_predictions)
    print(f"过滤掉 {original_count - filtered_count} 个不在已知实体集合中的预测结果")
    
    return filtered_predictions

def generate_prompt_examples(num_examples: int = 5):
    """生成示例prompt用于调试"""
    examples = []
    generator = PromptGenerator()
    
    # 生成一些示例
    sample_data = [
        ("e1", "r1", "e2", "手机", "品牌是", "苹果"),
        ("e3", "r2", "e4", "苹果", "属于", "科技公司"),
        ("e5", "r3", "e6", "中国", "首都是", "北京"),
        ("e7", "r4", "e8", "北京", "位于", "华北"),
        ("e9", "r5", "e10", "小明", "喜欢", "音乐")
    ]
    
    for i in range(min(num_examples, len(sample_data))):
        e_h, r, e_t, h_text, r_text, t_text = sample_data[i]
        
        # 生成正样本prompt
        pos_prompt = generator.generate_positive_prompt(h_text, r_text, t_text)
        examples.append({
            "type": "positive",
            "prompt": pos_prompt,
            "triple": (e_h, r, e_t),
            "text_triple": (h_text, r_text, t_text)
        })
        
        # 生成负样本prompt（简单替换尾实体）
        neg_t_text = "三星" if i % 2 == 0 else "上海"
        neg_prompt = generator.generate_negative_prompt(h_text, r_text, neg_t_text)
        examples.append({
            "type": "negative",
            "prompt": neg_prompt,
            "triple": (e_h, r, "e_invalid"),
            "text_triple": (h_text, r_text, neg_t_text)
        })
    
    return examples

def main():
    """主函数，用于测试数据处理工具"""
    print("=== 测试数据处理工具 ===")
    
    # 测试EntityRelationMapper
    print("\n测试EntityRelationMapper:")
    # 注意：这里使用示例路径，实际使用时需要替换为真实路径
    mapper = EntityRelationMapper(
        "/mnt/d/forCoding_data/Tianchi_EcommerceKG/originalData/OpenBG500/OpenBG500_entity2text.tsv",
        "/mnt/d/forCoding_data/Tianchi_EcommerceKG/originalData/OpenBG500/OpenBG500_relation2text.tsv"
    )
    
    # 测试PromptGenerator
    print("\n测试PromptGenerator:")
    generator = PromptGenerator()
    pos_prompt = generator.generate_positive_prompt("手机", "品牌是", "苹果")
    neg_prompt = generator.generate_negative_prompt("手机", "品牌是", "三星")
    pred_prompt = generator.generate_prediction_prompt("手机", "品牌是")
    print(f"正样本prompt: {pos_prompt}")
    print(f"负样本prompt: {neg_prompt}")
    print(f"预测prompt: {pred_prompt}")
    
    # 测试ChineseTextProcessor
    print("\n测试ChineseTextProcessor:")
    processor = ChineseTextProcessor()
    text = "这是一个测试文本，包含一些特殊字符！@#$%"
    cleaned_text = processor.clean_text(text)
    tokens = processor.tokenize(cleaned_text)
    print(f"原始文本: {text}")
    print(f"清理后: {cleaned_text}")
    print(f"分词结果: {tokens}")
    
    # 生成prompt示例
    print("\n生成prompt示例:")
    examples = generate_prompt_examples(3)
    for i, example in enumerate(examples):
        print(f"示例 {i+1} ({example['type']}): {example['prompt']}")

if __name__ == "__main__":
    main()