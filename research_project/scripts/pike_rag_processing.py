#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PIKE-RAG 数据处理脚本 - 文档分块和原子问题提取
"""

import os
import sys
sys.path.append('.')

import json
import yaml
import shutil
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime
import importlib
import re

# 添加PIKE-RAG相关导入
from pikerag.llm_client.company_openai_client import CompanyOpenAIClient
from pikerag.prompts.tagging.atom_question_tagging import (
    atom_question_tagging_template_cn, 
    AtomQuestionParser
)
from pikerag.prompts import CommunicationProtocol

# 简单的文本分割器实现
class SimpleTextSplitter:
    def __init__(self, chunk_size=1000, chunk_overlap=200, separators=None):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators or ["\n\n", "\n", "。", "；", "，", " "]
    
    def split_text(self, text: str) -> List[str]:
        """分割文本"""
        chunks = []
        current_chunk = ""
        
        # 按分隔符分割
        parts = []
        for separator in self.separators:
            if separator in text:
                parts = text.split(separator)
                break
        
        if not parts:
            parts = [text]
        
        for part in parts:
            if len(current_chunk) + len(part) < self.chunk_size:
                current_chunk += part + (separator if separator in text else "")
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = part + (separator if separator in text else "")
        
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return [chunk for chunk in chunks if chunk.strip()]


class PikeRagProcessor:
    """PIKE-RAG 数据处理器"""
    
    def __init__(self, config_path: str):
        self.config_path = config_path
        self.config = self.load_config()
        self.setup_directories()
        self.text_splitter = self.setup_text_splitter()
        self.setup_llm_client()
    
    def load_config(self) -> dict:
        """加载配置文件"""
        with open(self.config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        # 创建日志目录
        experiment_name = config["experiment_name"]
        log_dir = os.path.join(config["log_root_dir"], experiment_name)
        config["log_dir"] = log_dir
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        
        # 复制配置文件到日志目录
        if os.path.exists(self.config_path):
            shutil.copy(self.config_path, log_dir)
        
        return config
    
    def setup_directories(self):
        """设置目录"""
        self.input_dir = Path(self.config["input_doc_setting"]["doc_dir"])
        self.output_dir = Path(self.config["output_doc_setting"]["doc_dir"])
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def setup_text_splitter(self):
        """设置文本分割器"""
        doc_config = self.config.get("document_processing", {})
        return SimpleTextSplitter(
            chunk_size=doc_config.get("chunk_size", 1000),
            chunk_overlap=doc_config.get("chunk_overlap", 200),
            separators=doc_config.get("separators", ["\n\n", "\n", "。", "；", "，", " "])
        )
    
    def setup_llm_client(self):
        """设置LLM客户端用于原子问题生成"""
        try:
            self.llm_client = CompanyOpenAIClient()
            
            # 设置原子问题生成协议
            self.atom_question_protocol = CommunicationProtocol(
                template=atom_question_tagging_template_cn,
                parser=AtomQuestionParser(),
            )
            
            print("CompanyOpenAI 客户端初始化成功")
            self.use_llm = True
        except Exception as e:
            print(f"无法初始化CompanyOpenAI客户端: {e}")
            print("将使用简化的关键词生成方法")
            self.use_llm = False
    
    def load_documents(self) -> List[Dict[str, Any]]:
        """加载文档"""
        documents = []
        
        # 加载documents.jsonl
        docs_file = self.input_dir / "documents.jsonl"
        if docs_file.exists():
            with open(docs_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        doc = json.loads(line)
                        documents.append(doc)
        
        print(f"加载了 {len(documents)} 个文档")
        return documents
    
    def split_documents(self, documents: List[Dict]) -> List[Dict]:
        """分割文档为块"""
        all_chunks = []
        
        for doc in documents:
            print(f"分割文档: {doc['title']}")
            
            # 使用文本分割器
            text_chunks = self.text_splitter.split_text(doc["content"])
            
            # 为每个块创建元数据
            for i, chunk_text in enumerate(text_chunks):
                chunk = {
                    "chunk_id": f"{doc['id']}_chunk_{i:03d}",
                    "source_doc_id": doc["id"],
                    "source_doc_title": doc["title"],
                    "source_doc_filename": doc["filename"],
                    "chunk_index": i,
                    "content": chunk_text,
                    "word_count": len(chunk_text),
                    "metadata": {
                        "document_type": doc["metadata"]["document_type"],
                        "language": doc["metadata"]["language"],
                        "chunk_size": len(chunk_text)
                    }
                }
                all_chunks.append(chunk)
        
        print(f"生成了 {len(all_chunks)} 个文档块")
        return all_chunks
    
    def generate_atom_questions_with_llm(self, chunk: Dict) -> List[str]:
        """使用LLM生成原子问题"""
        try:
            # 准备输入内容
            content = chunk["content"]
            title = chunk.get("source_doc_title", "")
            
            # 处理输入
            messages = self.atom_question_protocol.process_input(
                content=content,
                title=title
            )
            
            # 调用LLM
            response = self.llm_client.generate_content_with_messages(messages)
            
            # 解析输出
            questions = self.atom_question_protocol.parse_output(response)
            
            # 限制问题数量
            if len(questions) > 5:
                questions = questions[:5]
            
            # 确保至少有一个问题
            if not questions:
                questions = [f"关于{title}的相关规定是什么？"]
            
            return questions
            
        except Exception as e:
            print(f"LLM生成原子问题失败: {e}")
            # 回退到简化方法
            return self.generate_atom_questions_fallback(chunk)
    
    def generate_atom_questions_fallback(self, chunk: Dict) -> List[str]:
        """回退的原子问题生成方法（改进的关键词匹配）"""
        content = chunk["content"]
        title = chunk.get("source_doc_title", "")
        questions = []
        
        # 更智能的关键词匹配
        keyword_patterns = {
            "验收": [
                f"{title}中的验收标准是什么？",
                f"{title}的验收程序如何进行？",
                f"{title}规定的验收要求有哪些？"
            ],
            "标准": [
                f"{title}中提到了哪些技术标准？",
                f"{title}的执行标准是什么？",
                f"{title}涉及的标准规范有哪些？"
            ],
            "规范": [
                f"{title}的技术规范要求是什么？",
                f"{title}中的规范条款有哪些？"
            ],
            "施工": [
                f"{title}对施工有什么具体要求？",
                f"{title}的施工技术规定是什么？",
                f"{title}中施工应注意哪些事项？"
            ],
            "质量": [
                f"{title}的质量控制要求是什么？",
                f"{title}中质量标准如何规定？"
            ],
            "安全": [
                f"{title}的安全规定有哪些？",
                f"{title}中安全措施如何要求？"
            ],
            "材料": [
                f"{title}对材料有什么要求？",
                f"{title}中材料标准是什么？"
            ],
            "检测": [
                f"{title}的检测方法是什么？",
                f"{title}中检测要求如何规定？"
            ]
        }
        
        # 检查关键词并生成问题
        for keyword, question_templates in keyword_patterns.items():
            if keyword in content:
                # 随机选择一个问题模板
                import random
                questions.append(random.choice(question_templates))
        
        # 确保至少有一个问题
        if not questions:
            questions = [
                f"关于{title}的主要规定是什么？",
                f"{title}的技术要求有哪些？",
                f"{title}中的重点内容是什么？"
            ]
        
        return questions[:3]  # 最多返回3个问题
    
    def process_chunks_with_atoms(self, chunks: List[Dict]) -> List[Dict]:
        """为文档块生成原子问题"""
        for i, chunk in enumerate(chunks):
            if i % 50 == 0:  # 每50个块打印一次进度
                print(f"处理块 {i+1}/{len(chunks)}")
            
            # 生成原子问题
            if self.use_llm:
                atom_questions = self.generate_atom_questions_with_llm(chunk)
            else:
                atom_questions = self.generate_atom_questions_fallback(chunk)
            
            chunk["atom_questions"] = atom_questions
            
            # 为每个原子问题创建独立对象
            atoms = []
            for j, question in enumerate(atom_questions):
                atom = {
                    "atom_id": f"{chunk['chunk_id']}_atom_{j:02d}",
                    "source_chunk_id": chunk["chunk_id"],
                    "source_doc_id": chunk["source_doc_id"],
                    "question": question,
                    "context": chunk["content"]
                }
                atoms.append(atom)
            
            chunk["atoms"] = atoms
        
        return chunks
    
    def save_processed_data(self, chunks: List[Dict]):
        """保存处理后的数据"""
        
        # 保存分块数据（用于chunk store）
        chunks_file = self.output_dir / "chunks_with_atoms.jsonl"
        with open(chunks_file, 'w', encoding='utf-8') as f:
            for chunk in chunks:
                f.write(json.dumps(chunk, ensure_ascii=False) + '\n')
        
        # 保存原子问题数据（用于atom store）
        atoms_file = self.output_dir / "atoms.jsonl"
        with open(atoms_file, 'w', encoding='utf-8') as f:
            for chunk in chunks:
                for atom in chunk.get("atoms", []):
                    f.write(json.dumps(atom, ensure_ascii=False) + '\n')
        
        # 保存统计信息
        total_atoms = sum(len(chunk.get("atoms", [])) for chunk in chunks)
        stats = {
            "processing_time": datetime.now().isoformat(),
            "total_chunks": len(chunks),
            "total_atoms": total_atoms,
            "average_atoms_per_chunk": total_atoms / len(chunks) if chunks else 0,
            "chunk_size": self.config.get("document_processing", {}).get("chunk_size", 1000),
            "chunk_overlap": self.config.get("document_processing", {}).get("chunk_overlap", 200)
        }
        
        stats_file = self.output_dir / "chunking_stats.json"
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)
        
        print(f"\n文档分块和原子问题生成完成!")
        print(f"文档块数量: {len(chunks)}")
        print(f"原子问题数量: {total_atoms}")
        print(f"平均每块原子问题数: {total_atoms / len(chunks):.2f}")
        print(f"输出目录: {self.output_dir}")
    
    def run(self):
        """执行处理流程"""
        print("开始PIKE-RAG数据处理...")
        
        # 加载文档
        documents = self.load_documents()
        
        # 分割文档
        chunks = self.split_documents(documents)
        
        # 生成原子问题
        chunks_with_atoms = self.process_chunks_with_atoms(chunks)
        
        # 保存处理后的数据
        self.save_processed_data(chunks_with_atoms)


def main():
    """主函数"""
    config_path = "research_project/configs/chunking_config.yml"
    
    processor = PikeRagProcessor(config_path)
    processor.run()


if __name__ == "__main__":
    main() 