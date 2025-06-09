#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试交互式问答系统的核心功能
"""

import sys
sys.path.append('.')

from research_project.scripts.interactive_qa import InteractiveQASystem

def test_qa_system():
    """测试问答系统的基本功能"""
    print("🧪 开始测试交互式问答系统...")
    
    try:
        # 初始化系统
        qa_system = InteractiveQASystem("research_project/configs/qa_config.yml")
        print("✅ 系统初始化成功")
        
        # 测试简单问题
        test_question = "建筑高度有什么规定"
        print(f"\n🔍 测试问题: {test_question}")
        
        result = qa_system.ask_question(test_question)
        
        if result["success"]:
            print("✅ 问答成功!")
            print(f"💬 回答: {result['answer'][:200]}...")
            
            # 检查推理过程
            if "reasoning" in result:
                reasoning = result["reasoning"]
                print(f"🧠 推理过程包含 {len(reasoning)} 个键: {list(reasoning.keys())}")
        else:
            print(f"❌ 问答失败: {result['error']}")
            
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_qa_system() 