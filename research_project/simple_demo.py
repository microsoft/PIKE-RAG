#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
简化演示脚本 - 快速测试 PIKE-RAG 建筑工程规范问答功能
不需要完整的向量存储，使用简单的文本匹配进行演示
"""

import os
import sys
import json
import re
from pathlib import Path
from typing import List, Dict, Any


class SimpleQADemo:
    """简化问答演示系统"""
    
    def __init__(self):
        self.documents = []
        self.qa_pairs = []
        self.load_data()
    
    def load_data(self):
        """加载数据"""
        # 加载处理后的文档
        docs_file = Path("research_project/processed_data/documents.jsonl")
        if docs_file.exists():
            with open(docs_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        doc = json.loads(line)
                        self.documents.append(doc)
        
        # 加载问答对
        qa_file = Path("research_project/processed_data/qa_pairs.jsonl")
        if qa_file.exists():
            with open(qa_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        qa = json.loads(line)
                        self.qa_pairs.append(qa)
        
        print(f"📚 加载了 {len(self.documents)} 个文档")
        print(f"💬 加载了 {len(self.qa_pairs)} 个问答对")
    
    def simple_search(self, query: str, top_k: int = 3) -> List[Dict]:
        """简单的关键词搜索"""
        results = []
        
        # 提取查询关键词
        keywords = re.findall(r'[\u4e00-\u9fff]+|[a-zA-Z]+', query)
        
        for doc in self.documents:
            score = 0
            content = doc["content"]
            title = doc["title"]
            
            # 计算匹配分数
            for keyword in keywords:
                if keyword in content:
                    score += content.count(keyword)
                if keyword in title:
                    score += 5  # 标题匹配权重更高
            
            if score > 0:
                results.append({
                    "document": doc,
                    "score": score,
                    "matched_content": self.extract_relevant_content(content, keywords)
                })
        
        # 按分数排序
        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:top_k]
    
    def extract_relevant_content(self, content: str, keywords: List[str], context_length: int = 200) -> str:
        """提取相关内容片段"""
        for keyword in keywords:
            if keyword in content:
                start = content.find(keyword)
                start = max(0, start - context_length // 2)
                end = min(len(content), start + context_length)
                return content[start:end] + "..."
        return content[:context_length] + "..."
    
    def generate_simple_answer(self, query: str, search_results: List[Dict]) -> str:
        """生成简单回答"""
        if not search_results:
            return "抱歉，我在建筑工程规范文档中没有找到相关信息。"
        
        # 构建回答
        answer_parts = []
        answer_parts.append(f"根据建筑工程规范，关于您的问题「{query}」，我找到以下相关信息：\n")
        
        for i, result in enumerate(search_results, 1):
            doc = result["document"]
            content = result["matched_content"]
            answer_parts.append(f"{i}. 来源：{doc['title']}")
            answer_parts.append(f"   内容：{content}\n")
        
        return "\n".join(answer_parts)
    
    def ask_question(self, query: str) -> Dict:
        """处理问题"""
        try:
            # 搜索相关文档
            search_results = self.simple_search(query)
            
            # 生成回答
            answer = self.generate_simple_answer(query, search_results)
            
            return {
                "success": True,
                "query": query,
                "answer": answer,
                "sources": len(search_results),
                "search_results": search_results
            }
        
        except Exception as e:
            return {
                "success": False,
                "query": query,
                "error": str(e),
                "answer": f"处理问题时出现错误：{e}"
            }
    
    def run_demo_questions(self):
        """运行演示问题"""
        demo_questions = [
            "建筑电气工程施工质量验收规范中对接地系统有什么要求？",
            "地下防水工程的验收标准是什么？",
            "建筑基桩检测的主要方法有哪些？"
        ]
        
        print("\n" + "="*80)
        print("🏗️  建筑工程规范智能问答系统 - 快速演示")
        print("="*80)
        
        for i, question in enumerate(demo_questions, 1):
            print(f"\n📝 演示问题 {i}: {question}")
            print("-" * 60)
            
            result = self.ask_question(question)
            
            if result["success"]:
                print(f"💡 回答：\n{result['answer']}")
                print(f"📊 找到 {result['sources']} 个相关文档片段")
            else:
                print(f"❌ 错误：{result['error']}")
            
            print("-" * 60)
    
    def run_validation(self):
        """运行验证测试"""
        print("\n" + "="*80)
        print("🔬 验证测试 - 对比标准答案")
        print("="*80)
        
        if not self.qa_pairs:
            print("❌ 没有可用的验证问答对")
            return
        
        test_count = min(3, len(self.qa_pairs))
        correct_count = 0
        
        for i, qa in enumerate(self.qa_pairs[:test_count], 1):
            question = qa["question"]
            expected_answer = qa["answer"]
            
            print(f"\n📝 测试问题 {i}: {question}")
            print("-" * 60)
            
            result = self.ask_question(question)
            
            if result["success"]:
                system_answer = result["answer"]
                print(f"💬 系统回答：\n{system_answer}")
                print(f"\n✅ 标准答案：\n{expected_answer}")
                
                # 简单的关键词匹配检查
                keywords = expected_answer.split()[:10]  # 取前10个词
                matches = sum(1 for keyword in keywords if keyword in system_answer)
                
                if matches >= len(keywords) * 0.3:  # 30%的关键词匹配
                    correct_count += 1
                    print("\n✅ 评估：答案相关")
                else:
                    print("\n❌ 评估：答案不够相关")
            else:
                print(f"❌ 系统错误：{result['error']}")
            
            print("-" * 60)
        
        accuracy = (correct_count / test_count) * 100 if test_count > 0 else 0
        print(f"\n📊 验证结果：{correct_count}/{test_count} 正确 (准确率: {accuracy:.1f}%)")
    
    def interactive_mode(self):
        """交互模式"""
        print("\n" + "="*80)
        print("💬 交互模式 - 输入问题开始对话")
        print("💡 输入 'quit' 退出，'demo' 运行演示，'validate' 运行验证")
        print("="*80)
        
        while True:
            try:
                user_input = input("\n❓ 请输入您的问题：").strip()
                
                if not user_input:
                    continue
                
                if user_input.lower() in ['quit', 'exit', 'q']:
                    print("👋 谢谢使用！再见！")
                    break
                elif user_input.lower() == 'demo':
                    self.run_demo_questions()
                    continue
                elif user_input.lower() == 'validate':
                    self.run_validation()
                    continue
                
                print(f"\n🤔 正在查找相关信息...")
                result = self.ask_question(user_input)
                
                if result["success"]:
                    print(f"\n💡 回答：\n{result['answer']}")
                else:
                    print(f"\n❌ 错误：{result['answer']}")
                
            except KeyboardInterrupt:
                print("\n\n👋 收到中断信号，正在退出...")
                break
            except Exception as e:
                print(f"\n❌ 系统错误：{e}")


def main():
    """主函数"""
    # 检查数据文件是否存在
    required_files = [
        "research_project/processed_data/documents.jsonl",
        "research_project/processed_data/qa_pairs.jsonl"
    ]
    
    missing_files = [f for f in required_files if not os.path.exists(f)]
    
    if missing_files:
        print("❌ 缺少必要的数据文件，请先运行数据预处理：")
        print("python research_project/scripts/data_preprocessing.py")
        return
    
    # 创建并运行演示系统
    demo = SimpleQADemo()
    
    print("🚀 选择运行模式：")
    print("1. 演示问题 (demo)")
    print("2. 验证测试 (validate)")
    print("3. 交互模式 (interactive)")
    
    try:
        choice = input("\n请选择模式 [1/2/3]: ").strip()
        
        if choice == '1' or choice.lower() == 'demo':
            demo.run_demo_questions()
        elif choice == '2' or choice.lower() == 'validate':
            demo.run_validation()
        elif choice == '3' or choice.lower() == 'interactive':
            demo.interactive_mode()
        else:
            print("自动进入交互模式...")
            demo.interactive_mode()
    
    except KeyboardInterrupt:
        print("\n👋 用户取消，退出程序")


if __name__ == "__main__":
    main() 