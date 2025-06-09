#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PIKE-RAG 建筑工程规范问答系统 - 完整流水线执行脚本
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path


def run_command(command: str, description: str) -> bool:
    """运行命令并显示结果"""
    print(f"\n{'='*60}")
    print(f"🚀 {description}")
    print(f"{'='*60}")
    print(f"执行命令: {command}")
    print("-" * 60)
    
    try:
        result = subprocess.run(command, shell=True, check=True)
        print(f"✅ {description} 完成")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} 失败: {e}")
        return False


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="PIKE-RAG 建筑工程规范问答系统流水线")
    parser.add_argument(
        "--step", 
        type=str, 
        choices=["all", "preprocess", "chunk", "vector", "qa"],
        default="all",
        help="执行步骤: all(全部), preprocess(数据预处理), chunk(文档分块), vector(向量存储), qa(问答系统)"
    )
    parser.add_argument(
        "--skip-vector", 
        action="store_true",
        help="跳过向量存储构建（如果已经构建过）"
    )
    
    args = parser.parse_args()
    
    print("🏗️  PIKE-RAG 建筑工程规范智能问答系统")
    print("   数据处理和问答流水线")
    print("=" * 80)
    
    success = True
    
    # 步骤1: 数据预处理
    if args.step in ["all", "preprocess"]:
        success &= run_command(
            "python research_project/scripts/data_preprocessing.py",
            "步骤1: 数据预处理 - 转换原始文档和问答对"
        )
    
    # 步骤2: 文档分块和原子问题提取
    if args.step in ["all", "chunk"] and success:
        success &= run_command(
            "python research_project/scripts/pike_rag_processing.py",
            "步骤2: 文档分块和原子问题提取"
        )
    
    # 步骤3: 构建向量存储
    if args.step in ["all", "vector"] and success and not args.skip_vector:
        success &= run_command(
            "python research_project/scripts/build_vector_store.py",
            "步骤3: 构建向量存储"
        )
    
    # 步骤4: 启动问答系统
    if args.step in ["all", "qa"] and success:
        print(f"\n{'='*60}")
        print("🎉 流水线执行完成!")
        print("💡 现在可以启动交互式问答系统")
        print(f"{'='*60}")
        
        # 检查必要文件是否存在
        required_files = [
            "research_project/processed_data/documents.jsonl",
            "research_project/processed_data/qa_pairs.jsonl",
            "research_project/chunked_data/chunks_with_atoms.jsonl",
            "research_project/chunked_data/atoms.jsonl",
            "research_project/vector_stores",
            "research_project/configs/qa_config.yml"
        ]
        
        missing_files = []
        for file_path in required_files:
            if not os.path.exists(file_path):
                missing_files.append(file_path)
        
        if missing_files:
            print("❌ 缺少必要文件:")
            for file_path in missing_files:
                print(f"  - {file_path}")
            print("\n请先运行完整的流水线: python research_project/run_pipeline.py --step all")
        else:
            print("✅ 所有必要文件已准备就绪")
            print("\n启动交互式问答系统:")
            print("python research_project/scripts/interactive_qa.py")
            
            # 询问是否立即启动
            try:
                user_input = input("\n是否现在启动问答系统? [y/N]: ").strip().lower()
                if user_input in ['y', 'yes']:
                    os.system("python research_project/scripts/interactive_qa.py")
            except KeyboardInterrupt:
                print("\n用户取消启动")
    
    if not success:
        print(f"\n❌ 流水线执行失败，请检查错误信息")
        sys.exit(1)
    
    print(f"\n🎉 所有步骤执行完成!")


if __name__ == "__main__":
    main() 