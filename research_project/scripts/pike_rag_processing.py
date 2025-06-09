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
        # 首先尝试使用CompanyOpenAI
        try:
            # 从配置文件获取LLM设置
            llm_config = self.config.get("llm_client", {})
            llm_args = llm_config.get("args", {})
            
            # 初始化LLM客户端，传递配置参数
            self.llm_client = CompanyOpenAIClient(
                logger=None,  # 暂时不使用logger
                llm_config=llm_config.get("llm_config", {}),
                **llm_args
            )
            
            print("✅ CompanyOpenAI 客户端初始化成功")
            self.use_llm = True
            return
        except Exception as e:
            print(f"⚠️  CompanyOpenAI 初始化失败: {e}")
        
        # 如果Company OpenAI失败，尝试使用标准OpenAI
        try:
            import openai
            from openai import OpenAI
            
            # 检查是否有有效的API key
            api_key = os.environ.get("OPENAI_API_KEY")
            if not api_key or api_key == "1@1":
                raise ValueError("无效的OPENAI_API_KEY")
            
            # 创建简单的OpenAI客户端包装器
            class SimpleOpenAIClient:
                def __init__(self):
                    self.client = OpenAI(api_key=api_key)
                
                def generate_content_with_messages(self, messages, **kwargs):
                    response = self.client.chat.completions.create(
                        model=kwargs.get("model", "gpt-4o-mini"),
                        messages=messages,
                        temperature=kwargs.get("temperature", 0),
                        max_tokens=kwargs.get("max_tokens", 1000)
                    )
                    return response.choices[0].message.content
            
            self.llm_client = SimpleOpenAIClient()
            print("✅ 标准OpenAI 客户端初始化成功")
            self.use_llm = True
            return
        except Exception as e:
            print(f"⚠️  标准OpenAI 初始化失败: {e}")
        
        # 如果所有方法都失败
        print("❌ 无法初始化任何LLM客户端")
        print("请设置有效的API key或检查网络连接")
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
        content = chunk["content"]
        title = chunk.get("source_doc_title", "")
        
        # 如果内容太短，跳过
        if len(content.strip()) < 50:
            return []
        
        try:
            # 构建更好的prompt
            messages = [
                {
                    "role": "system",
                    "content": "你是一个专业的建筑规范问答专家。请基于给定的文档片段，生成2-3个具体、实用的问题。"
                },
                {
                    "role": "user", 
                    "content": f"""请基于以下文档片段生成2-3个问题：

文档标题：{title}

文档内容：
{content}

要求：
1. 问题应该具体明确，能被该片段完整回答
2. 问题应该关注技术规范、标准要求等关键信息
3. 问题应该对实际工程有参考价值
4. 避免过于宽泛的问题
5. 使用中文

请直接返回问题列表，每行一个问题："""
                }
            ]
            
            # 调用LLM
            response = self.llm_client.generate_content_with_messages(
                messages, 
                **self.config.get("llm_client", {}).get("llm_config", {})
            )
            
            # 解析响应
            questions = []
            for line in response.split('\n'):
                line = line.strip()
                if line and '?' in line or '？' in line:
                    # 清理格式
                    line = line.lstrip('0123456789.- ').strip()
                    if len(line) > 10:  # 过滤过短的问题
                        questions.append(line)
            
            # 限制问题数量
            questions = questions[:3]

            print(f"✅ 为块 {chunk['chunk_id']} 生成了 {len(questions)} 个问题")
            return questions
            
        except Exception as e:
            print(f"❌ LLM生成原子问题失败: {e}")

    
    def process_chunks_with_atoms(self, chunks: List[Dict]) -> List[Dict]:
        """为文档块生成原子问题"""
        if not self.use_llm:
            raise RuntimeError("❌ LLM客户端未正确初始化，无法生成原子问题。请检查API配置。")
        
        print(f"🚀 开始为 {len(chunks)} 个文档块生成原子问题...")
        
        for i, chunk in enumerate(chunks):
            if i % 10 == 0:  # 每10个块打印一次进度
                print(f"📝 处理进度: {i+1}/{len(chunks)}")
            
            # 使用LLM生成原子问题
            atom_questions = self.generate_atom_questions_with_llm(chunk)
            
            # 如果没有生成任何问题，跳过此块
            if not atom_questions:
                print(f"⚠️  块 {chunk['chunk_id']} 内容太短，跳过")
                chunk["atom_questions"] = []
                chunk["atoms"] = []
                continue
            
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
        
        # 统计信息
        total_atoms = sum(len(chunk.get("atoms", [])) for chunk in chunks)
        print(f"✅ 原子问题生成完成！")
        print(f"   总文档块: {len(chunks)}")
        print(f"   总原子问题: {total_atoms}")
        print(f"   平均每块: {total_atoms / len(chunks):.1f} 个问题")
        
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