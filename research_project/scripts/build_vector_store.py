#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
向量存储构建脚本 - 将分块数据和原子问题数据构建为向量数据库
"""

import os
import sys
sys.path.append('.')

import json
import importlib
from pathlib import Path
from typing import List, Dict, Any
import warnings

# 忽略警告
warnings.filterwarnings("ignore")

from pikerag.utils.config_loader import load_dot_env
import chromadb
from chromadb.config import Settings


class VectorStoreBuilder:
    """向量存储构建器"""
    
    def __init__(self, data_dir: str, vector_store_dir: str, collection_name: str):
        self.data_dir = Path(data_dir)
        self.vector_store_dir = Path(vector_store_dir)
        self.collection_name = collection_name
        self.vector_store_dir.mkdir(parents=True, exist_ok=True)
        
        # 初始化embedding客户端
        self.setup_embedding_client()
        
        # 初始化Chroma客户端
        self.setup_chroma_client()
    
    def setup_embedding_client(self):
        """设置Embedding客户端"""
        try:
            from pikerag.llm_client.company_openai_client import CompanyOpenAIEmbedding
            
            # 配置embedding客户端
            client_config = {
                "timeout": 30.0, # 30秒
            }
            
            self.embedding_client = CompanyOpenAIEmbedding(
                client_config=client_config,
                model="text-embedding-ada-002"
            )
            print("使用 CompanyOpenAI Embedding 客户端")
        except Exception as e:
            print(f"无法加载CompanyOpenAI客户端: {e}")
            # 使用默认的sentence-transformers
            try:
                from sentence_transformers import SentenceTransformer
                self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
                self.embedding_client = None
                print("使用本地 sentence-transformers 模型")
            except ImportError:
                raise ImportError("需要安装 sentence-transformers: pip install sentence-transformers")
    
    def setup_chroma_client(self):
        """设置Chroma客户端"""
        self.client = chromadb.PersistentClient(
            path=str(self.vector_store_dir),
            settings=Settings(anonymized_telemetry=False)
        )
    
    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """获取文本embeddings"""
        if self.embedding_client:
            # 使用CompanyOpenAI embedding
            try:
                print(f"使用CompanyOpenAI生成 {len(texts)} 个文本的embeddings...")
                
                # 使用正确的方法名
                if len(texts) == 1:
                    # 单个文本使用embed_query
                    embedding = self.embedding_client.embed_query(texts[0])
                    return [embedding]
                else:
                    # 多个文本使用embed_documents，批量处理更高效
                    embeddings = self.embedding_client.embed_documents(texts, batch_call=True)
                    return embeddings
                    
            except Exception as e:
                print(f"CompanyOpenAI embedding失败: {e}")
                print("回退到本地sentence-transformers模型")
                
                # 回退到本地模型
                try:
                    from sentence_transformers import SentenceTransformer
                    if not hasattr(self, 'embedding_model') or self.embedding_model is None:
                        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
                    return self.embedding_model.encode(texts).tolist()
                except Exception as fallback_e:
                    print(f"本地模型也失败: {fallback_e}")
                    # 使用零向量作为最后fallback
                    return [[0.0] * 384] * len(texts)  # all-MiniLM-L6-v2的维度是384
        else:
            # 使用本地模型
            return self.embedding_model.encode(texts).tolist()
    
    def load_chunks_data(self) -> List[Dict]:
        """加载chunks数据"""
        chunks_file = self.data_dir / "chunks.jsonl"
        chunks = []
        
        with open(chunks_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    chunk = json.loads(line)
                    chunks.append(chunk)
        
        print(f"加载了 {len(chunks)} 个文档块")
        return chunks
    
    def load_atoms_data(self) -> List[Dict]:
        """加载atoms数据"""
        atoms_file = self.data_dir / "atoms.jsonl"
        atoms = []
        
        with open(atoms_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    atom = json.loads(line)
                    atoms.append(atom)
        
        print(f"加载了 {len(atoms)} 个原子问题")
        return atoms
    
    def build_chunk_collection(self, chunks: List[Dict]):
        """构建chunk collection"""
        collection_name = f"{self.collection_name}_chunks"
        
        # 删除已存在的collection
        try:
            self.client.delete_collection(collection_name)
        except:
            pass
        
        # 创建新的collection
        collection = self.client.create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        
        # 批量处理
        batch_size = 50
        for i in range(0, len(chunks), batch_size):
            batch_chunks = chunks[i:i+batch_size]
            
            # 准备数据
            ids = [chunk["chunk_id"] for chunk in batch_chunks]
            documents = [chunk["content"] for chunk in batch_chunks]
            metadatas = []
            
            for chunk in batch_chunks:
                metadata = {
                    "source_doc_id": chunk["source_doc_id"],
                    "source_doc_title": chunk["source_doc_title"],
                    "source_doc_filename": chunk["source_doc_filename"],
                    "chunk_index": chunk["chunk_index"],
                    "word_count": chunk["word_count"],
                    "document_type": chunk["metadata"]["document_type"],
                    "language": chunk["metadata"]["language"]
                }
                metadatas.append(metadata)
            
            # 获取embeddings
            print(f"处理chunk批次 {i//batch_size + 1}/{(len(chunks) + batch_size - 1)//batch_size}")
            embeddings = self.get_embeddings(documents)
            
            # 添加到collection
            collection.add(
                embeddings=embeddings,
                documents=documents,
                metadatas=metadatas,
                ids=ids
            )
        
        print(f"Chunk collection '{collection_name}' 构建完成，包含 {len(chunks)} 个文档块")
    
    def build_atom_collection(self, atoms: List[Dict]):
        """构建atom collection"""
        collection_name = f"{self.collection_name}_atoms"
        
        # 删除已存在的collection
        try:
            self.client.delete_collection(collection_name)
        except:
            pass
        
        # 创建新的collection
        collection = self.client.create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        
        # 批量处理
        batch_size = 50
        for i in range(0, len(atoms), batch_size):
            batch_atoms = atoms[i:i+batch_size]
            
            # 准备数据
            ids = [atom["atom_id"] for atom in batch_atoms]
            documents = [atom["question"] for atom in batch_atoms]  # 对问题进行embedding
            metadatas = []
            
            for atom in batch_atoms:
                metadata = {
                    "source_chunk_id": atom["source_chunk_id"],
                    "source_doc_id": atom["source_doc_id"],
                    "question": atom["question"],
                    "context": atom["context"][:500]  # 限制context长度
                }
                metadatas.append(metadata)
            
            # 获取embeddings
            print(f"处理atom批次 {i//batch_size + 1}/{(len(atoms) + batch_size - 1)//batch_size}")
            embeddings = self.get_embeddings(documents)
            
            # 添加到collection
            collection.add(
                embeddings=embeddings,
                documents=documents,
                metadatas=metadatas,
                ids=ids
            )
        
        print(f"Atom collection '{collection_name}' 构建完成，包含 {len(atoms)} 个原子问题")
    
    def run(self):
        """执行向量存储构建"""
        print("开始构建向量存储...")
        
        # 加载数据
        chunks = self.load_chunks_data()
        atoms = self.load_atoms_data()
        
        # 构建chunk collection
        self.build_chunk_collection(chunks)
        
        # 构建atom collection
        self.build_atom_collection(atoms)
        
        print(f"\n向量存储构建完成!")
        print(f"存储位置: {self.vector_store_dir}")
        print(f"Collection名称: {self.collection_name}_chunks, {self.collection_name}_atoms")


def main():
    """主函数"""
    data_dir = "research_project/chunked_data"
    vector_store_dir = "research_project/vector_stores"
    collection_name = "building_regulations"
    
    builder = VectorStoreBuilder(data_dir, vector_store_dir, collection_name)
    builder.run()


if __name__ == "__main__":
    main() 