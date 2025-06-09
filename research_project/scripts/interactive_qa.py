#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
交互式问答脚本 - 支持实时查询和推理过程展示
"""

import os
import sys
sys.path.append('.')

import json
import yaml
import argparse
import importlib
from typing import Dict, List, Any
from pathlib import Path

from pikerag.utils.config_loader import load_dot_env
from pikerag.workflows.qa import QaWorkflow
from pikerag.workflows.common import GenerationQaData


class InteractiveQASystem:
    """交互式问答系统"""
    
    def __init__(self, config_path: str):
        self.config_path = config_path
        self.config = self.load_config()
        self.qa_pairs = self.load_qa_pairs()
        self.setup_workflow()
    
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
        
        return config
    
    def load_qa_pairs(self) -> List[Dict]:
        """加载问答对用于验证"""
        qa_pairs = []
        qa_file = Path("research_project/processed_data/qa_pairs.jsonl")
        
        if qa_file.exists():
            with open(qa_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        qa_pair = json.loads(line)
                        qa_pairs.append(qa_pair)
        
        return qa_pairs
    
    def setup_workflow(self):
        """设置问答工作流"""
        # 加载环境变量
        load_dot_env(env_path=self.config.get("dotenv_path", None))
        
        # 动态导入工作流类
        workflow_config = self.config["workflow"]
        workflow_module = importlib.import_module(workflow_config["module_path"])
        workflow_class = getattr(workflow_module, workflow_config["class_name"])
        
        # 确保是QaWorkflow的子类
        assert issubclass(workflow_class, QaWorkflow)
        
        # 创建工作流实例
        self.workflow = workflow_class(self.config)
    
    def print_header(self):
        """打印系统头部信息"""
        print("=" * 80)
        print("🏗️  建筑工程规范智能问答系统")
        print("   基于 PIKE-RAG 技术，支持复杂推理和多跳问答")
        print("=" * 80)
        print(f"📚 已加载 {len(self.qa_pairs)} 个验证问答对")
        print("💡 输入问题开始对话，输入 'quit' 退出，输入 'help' 查看帮助")
        print("-" * 80)
    
    def print_help(self):
        """打印帮助信息"""
        print("\n📖 使用说明:")
        print("  • 直接输入问题进行问答")
        print("  • 输入 'examples' 查看示例问题")
        print("  • 输入 'validate' 运行验证测试")
        print("  • 输入 'quit' 或 'exit' 退出系统")
        print("  • 输入 'help' 显示此帮助信息")
        print()
    
    def print_examples(self):
        """打印示例问题"""
        print("\n📝 示例问题:")
        if self.qa_pairs:
            for i, qa in enumerate(self.qa_pairs[:5], 1):  # 显示前5个
                print(f"  {i}. {qa['question']}")
        else:
            print("  暂无示例问题")
        print()
    
    def display_reasoning_process(self, result: Dict):
        """显示推理过程"""
        print("\n🧠 推理过程:")
        
        # 显示问题分解
        if "decomposition" in result:
            decomp = result["decomposition"]
            if "sub_questions" in decomp and decomp["sub_questions"]:
                print("  📋 问题分解:")
                for i, sub_q in enumerate(decomp["sub_questions"], 1):
                    print(f"    {i}. {sub_q}")
        
        # 显示检索信息
        if "retrieved_info" in result:
            retrieved = result["retrieved_info"]
            if retrieved:
                print(f"  🔍 检索到 {len(retrieved)} 条相关信息")
                for i, info in enumerate(retrieved[:3], 1):  # 显示前3条
                    source = info.get("source", "未知来源")
                    content = info.get("content", "")[:100] + "..." if len(info.get("content", "")) > 100 else info.get("content", "")
                    print(f"    {i}. 来源: {source}")
                    print(f"       内容: {content}")
        
        # 显示选择的信息
        if "selected_info" in result:
            selected = result["selected_info"]
            if selected:
                print(f"  ✅ 选择了 {len(selected)} 条最相关信息用于回答")
        
        print()
    
    def ask_question(self, question: str) -> Dict:
        """处理问题并返回结果"""
        try:
            # 构建测试数据格式 - 需要使用BaseQaData的子类
            test_data = GenerationQaData(
                id="interactive_question",
                question=question,
                answer_labels=[""],  # 交互模式下没有标准答案
                metadata={"source": "interactive"}
            )
            
            # 调用工作流处理 - 使用正确的answer方法
            result = self.workflow.answer(test_data, 0)
            
            return {
                "success": True,
                "result": result,
                "answer": result.get("answer", "抱歉，无法找到相关答案。"),
                "reasoning": result
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "answer": f"处理问题时出现错误: {e}"
            }
    
    def validate_with_qa_pairs(self):
        """使用预置问答对进行验证"""
        print("\n🔬 开始验证测试...")
        
        if not self.qa_pairs:
            print("❌ 没有可用的验证问答对")
            return
        
        correct_count = 0
        total_count = min(3, len(self.qa_pairs))  # 只测试前3个问题
        
        for i, qa_pair in enumerate(self.qa_pairs[:total_count], 1):
            question = qa_pair["question"]
            expected_answer = qa_pair["answer"]
            
            print(f"\n📝 测试问题 {i}/{total_count}:")
            print(f"Q: {question}")
            
            # 获取系统回答
            result = self.ask_question(question)
            
            if result["success"]:
                system_answer = result["answer"]
                print(f"💬 系统回答: {system_answer}")
                print(f"✅ 标准答案: {expected_answer}")
                
                # 简单的相似度检查（这里可以用更复杂的方法）
                if any(keyword in system_answer for keyword in expected_answer.split()[:5]):
                    correct_count += 1
                    print("✅ 答案相关性: 匹配")
                else:
                    print("❌ 答案相关性: 不匹配")
            else:
                print(f"❌ 系统错误: {result['error']}")
            
            print("-" * 60)
        
        accuracy = (correct_count / total_count) * 100 if total_count > 0 else 0
        print(f"\n📊 验证结果: {correct_count}/{total_count} 正确 (准确率: {accuracy:.1f}%)")
    
    def run(self):
        """运行交互式问答系统"""
        self.print_header()
        
        while True:
            try:
                # 获取用户输入
                user_input = input("\n💬 请输入您的问题: ").strip()
                
                if not user_input:
                    continue
                
                # 处理特殊命令
                if user_input.lower() in ['quit', 'exit', 'q']:
                    print("\n👋 谢谢使用！再见！")
                    break
                elif user_input.lower() in ['help', 'h']:
                    self.print_help()
                    continue
                elif user_input.lower() in ['examples', 'ex']:
                    self.print_examples()
                    continue
                elif user_input.lower() in ['validate', 'test']:
                    self.validate_with_qa_pairs()
                    continue
                
                # 处理问题
                print(f"\n🤔 正在思考问题: {user_input}")
                print("⏳ 请稍候...")
                
                result = self.ask_question(user_input)
                
                if result["success"]:
                    print(f"\n💡 回答: {result['answer']}")
                    
                    # 显示推理过程
                    self.display_reasoning_process(result["reasoning"])
                else:
                    print(f"\n❌ 错误: {result['answer']}")
                
                print("-" * 80)
                
            except KeyboardInterrupt:
                print("\n\n👋 收到中断信号，正在退出...")
                break
            except Exception as e:
                print(f"\n❌ 系统错误: {e}")
                print("请重试或输入 'help' 查看帮助")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="建筑工程规范智能问答系统")
    parser.add_argument(
        "--config", 
        type=str, 
        default="research_project/configs/qa_config.yml",
        help="配置文件路径"
    )
    
    args = parser.parse_args()
    
    # 检查配置文件是否存在
    if not os.path.exists(args.config):
        print(f"❌ 错误: 配置文件 {args.config} 不存在")
        return
    
    # 创建并运行问答系统
    qa_system = InteractiveQASystem(args.config)
    qa_system.run()


if __name__ == "__main__":
    main() 