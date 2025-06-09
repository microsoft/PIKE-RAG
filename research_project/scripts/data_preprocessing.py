#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据预处理脚本 - 将原始文档和问答对转换为PIKE-RAG标准格式
"""

import os
import json
import pandas as pd
import re
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime


class DataPreprocessor:
    """数据预处理器"""
    
    def __init__(self, markdown_dir: str, excel_file: str, output_dir: str):
        self.markdown_dir = Path(markdown_dir)
        self.excel_file = Path(excel_file)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def clean_markdown_content(self, content: str) -> str:
        """清理markdown内容"""
        # 移除图片链接
        content = re.sub(r'!\[.*?\]\(.*?\)', '', content)
        # 移除多余的空行
        content = re.sub(r'\n\s*\n\s*\n', '\n\n', content)
        # 移除开头和结尾的空白
        content = content.strip()
        return content
    
    def extract_document_metadata(self, content: str, filename: str) -> Dict[str, Any]:
        """提取文档元数据"""
        lines = content.split('\n')
        title = ""
        
        # 尝试提取标题
        for line in lines[:10]:  # 只检查前10行
            line = line.strip()
            if line and not line.startswith('#'):
                title = line
                break
        
        if not title:
            title = filename.replace('.md', '')
        
        return {
            "title": title,
            "filename": filename,
            "document_type": "建筑工程规范",
            "language": "zh-CN"
        }
    
    def process_markdown_documents(self) -> List[Dict[str, Any]]:
        """处理markdown文档"""
        documents = []
        
        for md_file in self.markdown_dir.glob("*.md"):
            print(f"处理文档: {md_file.name}")
            
            with open(md_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # 清理内容
            cleaned_content = self.clean_markdown_content(content)
            
            # 提取元数据
            metadata = self.extract_document_metadata(cleaned_content, md_file.name)
            
            # 创建文档对象
            doc = {
                "id": f"doc_{len(documents):03d}",
                "title": metadata["title"],
                "filename": metadata["filename"],
                "content": cleaned_content,
                "metadata": metadata,
                "word_count": len(cleaned_content)
            }
            
            documents.append(doc)
        
        return documents
    
    def process_qa_examples(self) -> List[Dict[str, Any]]:
        """处理问答样例"""
        print(f"处理问答样例: {self.excel_file.name}")
        
        # 读取Excel文件
        df = pd.read_excel(self.excel_file)
        
        qa_pairs = []
        
        for idx, row in df.iterrows():
            # 提取问题和答案
            question = str(row.get('测试问题', '')).strip()
            answer_content = str(row.get('回答中应包含的关键内容', '')).strip()
            related_docs = str(row.get('指向相关文档和相关章节内容', '')).strip()
            question_type = str(row.get('类型', '')).strip()
            
            if question and answer_content and question != 'nan' and answer_content != 'nan':
                qa_pair = {
                    "id": f"qa_{idx:03d}",
                    "question": question,
                    "answer": answer_content,
                    "answer_labels": [answer_content],
                    "related_documents": related_docs,
                    "question_type": question_type,
                    "metadata": {
                        "source": "qa_examples.xlsx",
                        "row_index": idx,
                        "language": "zh-CN"
                    }
                }
                qa_pairs.append(qa_pair)
        
        return qa_pairs
    
    def save_processed_data(self, documents: List[Dict], qa_pairs: List[Dict]):
        """保存处理后的数据"""
        
        # 保存文档数据
        docs_file = self.output_dir / "documents.jsonl"
        with open(docs_file, 'w', encoding='utf-8') as f:
            for doc in documents:
                f.write(json.dumps(doc, ensure_ascii=False) + '\n')
        
        # 保存问答对数据
        qa_file = self.output_dir / "qa_pairs.jsonl"
        with open(qa_file, 'w', encoding='utf-8') as f:
            for qa in qa_pairs:
                f.write(json.dumps(qa, ensure_ascii=False) + '\n')
        
        # 保存统计信息
        stats = {
            "processing_time": datetime.now().isoformat(),
            "total_documents": len(documents),
            "total_qa_pairs": len(qa_pairs),
            "total_words": sum(doc["word_count"] for doc in documents),
            "document_files": [doc["filename"] for doc in documents]
        }
        
        stats_file = self.output_dir / "processing_stats.json"
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)
        
        print(f"\n数据预处理完成!")
        print(f"文档数量: {len(documents)}")
        print(f"问答对数量: {len(qa_pairs)}")
        print(f"总字数: {sum(doc['word_count'] for doc in documents):,}")
        print(f"输出目录: {self.output_dir}")
    
    def run(self):
        """执行数据预处理"""
        print("开始数据预处理...")
        
        # 处理markdown文档
        documents = self.process_markdown_documents()
        
        # 处理问答样例
        qa_pairs = self.process_qa_examples()
        
        # 保存处理后的数据
        self.save_processed_data(documents, qa_pairs)


def main():
    """主函数"""
    # 配置路径
    markdown_dir = "research_project/data_process/markdown"
    excel_file = "research_project/data_process/qa_examples.xlsx"
    output_dir = "research_project/processed_data"
    
    # 创建预处理器并运行
    preprocessor = DataPreprocessor(markdown_dir, excel_file, output_dir)
    preprocessor.run()


if __name__ == "__main__":
    main() 