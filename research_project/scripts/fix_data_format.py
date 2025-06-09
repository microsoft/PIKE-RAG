#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
数据格式修正脚本

修正PIKE-RAG数据格式，使其符合系统期望的格式
"""

import json
import os
from pathlib import Path

def fix_chunk_format():
    """修正chunk数据格式"""
    input_file = "research_project/chunked_data/chunks_with_atoms.jsonl"
    output_file = "research_project/chunked_data/chunks.jsonl"
    
    print("🔧 修正chunk数据格式...")
    
    with open(input_file, 'r', encoding='utf-8') as f_in, \
         open(output_file, 'w', encoding='utf-8') as f_out:
        
        for line_num, line in enumerate(f_in, 1):
            try:
                data = json.loads(line.strip())
                
                # 修正字段名：source_doc_title -> title
                if 'source_doc_title' in data:
                    data['title'] = data['source_doc_title']
                    # 保留原字段以兼容
                
                # 确保必要字段存在
                if 'title' not in data:
                    data['title'] = data.get('source_doc_filename', f"Document {data.get('source_doc_id', 'unknown')}")
                
                # 写入修正后的数据
                f_out.write(json.dumps(data, ensure_ascii=False) + '\n')
                
                if line_num % 100 == 0:
                    print(f"  处理了 {line_num} 个chunk...")
                    
            except Exception as e:
                print(f"❌ 处理第 {line_num} 行时出错: {e}")
                continue
    
    print(f"✅ Chunk数据格式修正完成: {output_file}")

def fix_atom_format():
    """修正atom数据格式（如果需要）"""
    input_file = "research_project/chunked_data/atoms.jsonl"
    
    print("🔧 检查atom数据格式...")
    
    # 检查前几行数据格式
    with open(input_file, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= 3:  # 只检查前3行
                break
            try:
                data = json.loads(line.strip())
                required_fields = ['atom_id', 'question', 'context']
                missing_fields = [field for field in required_fields if field not in data]
                if missing_fields:
                    print(f"⚠️ Atom数据缺少字段: {missing_fields}")
                else:
                    print(f"✅ Atom数据格式正确")
                    break
            except Exception as e:
                print(f"❌ Atom数据格式错误: {e}")
                break

def main():
    """主函数"""
    print("🚀 开始修正PIKE-RAG数据格式...")
    
    # 检查输入文件是否存在
    chunks_file = "research_project/chunked_data/chunks_with_atoms.jsonl"
    atoms_file = "research_project/chunked_data/atoms.jsonl"
    
    if not os.path.exists(chunks_file):
        print(f"❌ 找不到文件: {chunks_file}")
        return
    
    if not os.path.exists(atoms_file):
        print(f"❌ 找不到文件: {atoms_file}")
        return
    
    # 修正数据格式
    fix_chunk_format()
    fix_atom_format()
    
    print("✅ 数据格式修正完成!")
    print("\n📁 修正后的文件:")
    print("  - research_project/chunked_data/chunks.jsonl")
    print("  - research_project/chunked_data/atoms.jsonl")

if __name__ == "__main__":
    main() 