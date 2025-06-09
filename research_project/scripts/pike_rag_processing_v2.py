#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PIKE-RAG 数据处理脚本 - 复用PIKE-RAG内置能力的优化版本
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
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from langchain_core.documents import Document

# PIKE-RAG 内置组件导入
from pikerag.document_transformers import LLMPoweredTagger
from pikerag.llm_client.company_openai_client import CompanyOpenAIClient
from pikerag.prompts.tagging.atom_question_tagging import (
    atom_question_tagging_protocol_cn, 
    AtomQuestionParser
)
from pikerag.utils.logger import Logger


class SimpleLLMClientWrapper:
    """简单的LLM客户端包装器"""
    def __init__(self, api_key: str):
        import openai
        from openai import OpenAI
        self.client = OpenAI(api_key=api_key)
        self.monitor = None
    
    def set_monitor(self, monitor):
        self.monitor = monitor
    
    def generate_content_with_messages(self, messages, **kwargs):
        response = self.client.chat.completions.create(
            model=kwargs.get("model", "gpt-4o-mini"),
            messages=messages,
            temperature=kwargs.get("temperature", 0),
            max_tokens=kwargs.get("max_tokens", 1000)
        )
        return response.choices[0].message.content


class SimpleTextSplitter:
    """简单的文本分割器"""
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


class PikeRagProcessorV2:
    """PIKE-RAG 数据处理器 - 优化版本"""
    
    def __init__(self, config_path: str):
        self.config_path = config_path
        self.config = self.load_config()
        self.setup_directories()
        self.text_splitter = self.setup_text_splitter()
        self.setup_llm_client()
        self.setup_tagger()
        
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
        
        # 检查输入目录是否存在
        if not self.input_dir.exists():
            print(f"⚠️  输入目录不存在: {self.input_dir}")
            print("正在创建输入目录...")
            self.input_dir.mkdir(parents=True, exist_ok=True)
    
    def setup_text_splitter(self):
        """设置文本分割器"""
        doc_config = self.config.get("document_processing", {})
        return SimpleTextSplitter(
            chunk_size=doc_config.get("chunk_size", 1000),
            chunk_overlap=doc_config.get("chunk_overlap", 200),
            separators=doc_config.get("separators", ["\n\n", "\n", "。", "；", "，", " "])
        )
    
    def setup_llm_client(self):
        """设置LLM客户端"""
        # 首先尝试使用CompanyOpenAI
        try:
            llm_config = self.config.get("llm_client", {})
            llm_args = llm_config.get("args", {})
            
            self.llm_client = CompanyOpenAIClient(
                logger=None,
                llm_config=llm_config.get("llm_config", {}),
                **llm_args
            )
            print("✅ CompanyOpenAI 客户端初始化成功")
            self.use_llm = True
            return
        except Exception as e:
            print(f"⚠️  CompanyOpenAI 初始化失败: {e}")
        
        # 如果失败，尝试使用标准OpenAI
        try:
            api_key = os.environ.get("OPENAI_API_KEY")
            if not api_key or api_key == "1@1":
                raise ValueError("无效的OPENAI_API_KEY")
            
            self.llm_client = SimpleLLMClientWrapper(api_key)
            print("✅ 标准OpenAI 客户端初始化成功")
            self.use_llm = True
            return
        except Exception as e:
            print(f"⚠️  标准OpenAI 初始化失败: {e}")
            print("❌ 无法初始化任何LLM客户端")
            self.use_llm = False
    
    def setup_tagger(self):
        """设置PIKE-RAG内置的标记器"""
        if not self.use_llm:
            self.tagger = None
            return
        
        # 创建logger
        logger = Logger(
            name="tagger",
            dump_mode="w",
            dump_folder=self.config["log_dir"]
        )
        
        # 设置并行数量
        num_parallel = self.config.get("tagger", {}).get("num_parallel", 4)
        
        # 使用PIKE-RAG内置的LLMPoweredTagger
        self.tagger = LLMPoweredTagger(
            llm_client=self.llm_client,
            tagging_protocol=atom_question_tagging_protocol_cn,  # 使用中文协议
            num_parallel=num_parallel,
            tag_name="atom_questions",
            llm_config=self.config.get("llm_client", {}).get("llm_config", {}),
            logger=logger
        )
        
        print(f"✅ PIKE-RAG标记器初始化成功，并行数: {num_parallel}")
    
    def load_documents(self) -> List[Dict[str, Any]]:
        """加载文档"""
        documents = []
        
        # 加载documents.jsonl
        docs_file = self.input_dir / "documents.jsonl"
        print(f"🔍 查找文档文件: {docs_file}")
        
        if docs_file.exists():
            print(f"✅ 找到文档文件，开始加载...")
            with open(docs_file, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    if line.strip():
                        try:
                            doc = json.loads(line)
                            documents.append(doc)
                        except json.JSONDecodeError as e:
                            print(f"⚠️  第 {line_num} 行JSON解析错误: {e}")
        else:
            print(f"❌ 文档文件不存在: {docs_file}")
            print("请确保以下文件存在:")
            print(f"   - {docs_file}")
            print("\n💡 提示: 你可能需要先运行数据预处理脚本")
        
        print(f"📚 加载了 {len(documents)} 个文档")
        return documents
    
    def split_documents(self, documents: List[Dict]) -> List[Dict]:
        """分割文档为块"""
        if not documents:
            print("⚠️  没有文档需要分割")
            return []
            
        all_chunks = []
        
        for doc in documents:
            print(f"📄 分割文档: {doc['title']}")
            
            # 使用文本分割器
            text_chunks = self.text_splitter.split_text(doc["content"])
            
            # 为每个块创建元数据
            for i, chunk_text in enumerate(text_chunks):
                chunk = {
                    "chunk_id": f"{doc['id']}_chunk_{i:03d}",
                    "source_doc_id": doc["id"],
                    "source_doc_title": doc["title"],
                    "source_doc_filename": doc["filename"],
                    "title": doc["title"],  # 添加PIKE-RAG期望的title字段
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
        
        print(f"✂️  生成了 {len(all_chunks)} 个文档块")
        return all_chunks
    
    def process_chunks_with_atoms(self, chunks: List[Dict]) -> List[Dict]:
        """使用PIKE-RAG内置能力为文档块生成原子问题"""
        if not chunks:
            print("⚠️  没有文档块需要处理")
            return []
            
        if not self.use_llm or not self.tagger:
            raise RuntimeError("❌ LLM客户端或标记器未正确初始化")
        
        print(f"🚀 开始使用PIKE-RAG内置能力为 {len(chunks)} 个文档块生成原子问题...")
        
        # 转换为PIKE-RAG的Document格式
        documents = []
        for chunk in chunks:
            doc = Document(
                page_content=chunk["content"],
                metadata={
                    "chunk_id": chunk["chunk_id"],
                    "title": chunk.get("source_doc_title", ""),
                    "original_chunk": chunk  # 保存原始chunk数据
                }
            )
            documents.append(doc)
        
        # 使用PIKE-RAG内置的并行标记器
        print(f"⚡ 使用并行标记器处理 {len(documents)} 个文档块...")
        tagged_documents = self.tagger.transform_documents(documents)
        
        # 转换回我们的格式并流式写入
        print("💾 开始流式写入结果...")
        processed_chunks = []
        chunks_file = self.output_dir / "chunks_with_atoms.jsonl"
        atoms_file = self.output_dir / "atoms.jsonl"
        
        with open(chunks_file, 'w', encoding='utf-8') as chunks_writer, \
             open(atoms_file, 'w', encoding='utf-8') as atoms_writer:
            
            for tagged_doc in tagged_documents:
                # 获取原始chunk
                original_chunk = tagged_doc.metadata["original_chunk"]
                
                # 获取生成的原子问题
                atom_questions = tagged_doc.metadata.get("atom_questions", [])
                
                # 更新chunk数据
                original_chunk["atom_questions"] = atom_questions
                
                # 创建atom对象
                atoms = []
                for j, question in enumerate(atom_questions):
                    atom = {
                        "atom_id": f"{original_chunk['chunk_id']}_atom_{j:02d}",
                        "source_chunk_id": original_chunk["chunk_id"],
                        "source_doc_id": original_chunk["source_doc_id"],
                        "question": question,
                        "context": original_chunk["content"]
                    }
                    atoms.append(atom)
                    # 流式写入atom
                    atoms_writer.write(json.dumps(atom, ensure_ascii=False) + '\n')
                
                original_chunk["atoms"] = atoms
                processed_chunks.append(original_chunk)
                
                # 流式写入chunk
                chunks_writer.write(json.dumps(original_chunk, ensure_ascii=False) + '\n')
        
        # 统计信息
        total_atoms = sum(len(chunk.get("atoms", [])) for chunk in processed_chunks)
        print(f"✅ 原子问题生成完成！")
        print(f"   📊 总文档块: {len(processed_chunks)}")
        print(f"   🔬 总原子问题: {total_atoms}")
        
        # 修复除零错误
        if processed_chunks:
            avg_atoms = total_atoms / len(processed_chunks)
            print(f"   📈 平均每块: {avg_atoms:.1f} 个问题")
        else:
            print(f"   📈 平均每块: 0.0 个问题")
        
        return processed_chunks
    
    def save_statistics(self, chunks: List[Dict]):
        """保存统计信息"""
        total_atoms = sum(len(chunk.get("atoms", [])) for chunk in chunks)
        avg_atoms = total_atoms / len(chunks) if chunks else 0
        
        stats = {
            "processing_time": datetime.now().isoformat(),
            "total_chunks": len(chunks),
            "total_atoms": total_atoms,
            "average_atoms_per_chunk": avg_atoms,
            "chunk_size": self.config.get("document_processing", {}).get("chunk_size", 1000),
            "chunk_overlap": self.config.get("document_processing", {}).get("chunk_overlap", 200),
            "num_parallel": self.config.get("tagger", {}).get("num_parallel", 4),
            "llm_model": self.config.get("llm_client", {}).get("llm_config", {}).get("model", "未知")
        }
        
        stats_file = self.output_dir / "processing_stats.json"
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)
        
        print(f"\n📈 处理统计信息已保存到: {stats_file}")
    
    def run(self):
        """执行处理流程"""
        print("🎯 开始PIKE-RAG数据处理（优化版）...")
        
        # 加载文档
        documents = self.load_documents()
        
        if not documents:
            print("❌ 没有加载到任何文档，处理终止")
            print("\n💡 请检查:")
            print("   1. 输入目录是否正确")
            print("   2. documents.jsonl 文件是否存在")
            print("   3. 文件格式是否正确")
            return
        
        # 分割文档
        chunks = self.split_documents(documents)
        
        if not chunks:
            print("❌ 没有生成任何文档块，处理终止")
            return
        
        # 使用PIKE-RAG内置能力生成原子问题（并行 + 流式写入）
        chunks_with_atoms = self.process_chunks_with_atoms(chunks)
        
        # 保存统计信息
        self.save_statistics(chunks_with_atoms)
        
        print(f"\n🎉 PIKE-RAG数据处理完成！")
        print(f"📂 输出目录: {self.output_dir}")
        print(f"📄 文件:")
        print(f"   - chunks_with_atoms.jsonl: 包含原子问题的文档块")
        print(f"   - atoms.jsonl: 原子问题数据")
        print(f"   - processing_stats.json: 处理统计信息")


def main():
    """主函数"""
    config_path = "research_project/configs/chunking_config.yml"
    
    processor = PikeRagProcessorV2(config_path)
    processor.run()


if __name__ == "__main__":
    main() 