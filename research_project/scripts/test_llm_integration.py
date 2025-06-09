#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
LLM集成测试脚本

用于测试CompanyOpenAI客户端和原子问题生成是否正常工作
"""

import sys
import os
from pathlib import Path

# 添加项目根目录到Python路径
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from pikerag.llm_client.company_openai_client import CompanyOpenAIClient, CompanyOpenAIEmbedding
from pikerag.prompts.tagging.atom_question_tagging import (
    atom_question_tagging_template_cn, 
    AtomQuestionParser
)
from pikerag.prompts import CommunicationProtocol


def test_llm_client():
    """测试LLM客户端基本功能"""
    print("🧪 测试CompanyOpenAI客户端...")
    
    try:
        client = CompanyOpenAIClient()
        
        # 测试简单对话
        messages = [
            {"role": "user", "content": "你好，请简单介绍一下你自己。"}
        ]
        
        response = client.generate_content_with_messages(messages)
        print(f"✅ LLM响应: {response[:100]}...")
        return True
        
    except Exception as e:
        print(f"❌ LLM客户端测试失败: {e}")
        return False


def test_atom_question_generation():
    """测试原子问题生成"""
    print("\n🧪 测试原子问题生成...")
    
    try:
        # 初始化客户端和协议
        client = CompanyOpenAIClient()
        protocol = CommunicationProtocol(
            template=atom_question_tagging_template_cn,
            parser=AtomQuestionParser(),
        )
        
        # 测试文本
        test_content = """
        3.3.1 施工前应进行现场勘察，查明工程地质、水文地质条件，确定施工方案。
        3.3.2 施工中应建立质量管理体系，制定质量控制措施。
        3.3.3 施工结束后应按规定进行验收，验收合格后方可投入使用。
        """
        
        test_title = "建筑施工技术规范"
        
        # 处理输入
        messages = protocol.process_input(
            content=test_content,
            title=test_title
        )
        
        # 调用LLM
        response = client.generate_content_with_messages(messages)
        print(f"📝 LLM原始响应:\n{response}")
        
        # 解析输出
        questions = protocol.parse_output(response)
        
        print(f"✅ 生成的原子问题:")
        for i, question in enumerate(questions, 1):
            print(f"   {i}. {question}")
            
        return len(questions) > 0
        
    except Exception as e:
        print(f"❌ 原子问题生成测试失败: {e}")
        return False


def test_embedding_client():
    """测试Embedding客户端"""
    print("\n🧪 测试CompanyOpenAI Embedding...")
    
    try:
        # 配置embedding客户端
        client_config = {
            "timeout": 60.0,
        }
        
        embedding_client = CompanyOpenAIEmbedding(
            client_config=client_config,
            model="text-embedding-ada-002"
        )
        
        # 测试单个文本
        test_text = "这是一个测试文本，用于验证embedding功能是否正常。"
        
        print(f"📝 测试文本: {test_text}")
        
        embedding = embedding_client.embed_query(test_text)
        
        print(f"✅ Embedding维度: {len(embedding)}")
        print(f"✅ Embedding前5个值: {embedding[:5]}")
        
        # 测试多个文本
        test_texts = [
            "施工质量控制要求",
            "验收标准和程序",
            "安全措施规定"
        ]
        
        embeddings = embedding_client.embed_documents(test_texts, batch_call=True)
        print(f"✅ 批量embedding: {len(embeddings)} 个向量，每个维度: {len(embeddings[0]) if embeddings else 0}")
        
        return True
        
    except Exception as e:
        print(f"❌ Embedding客户端测试失败: {e}")
        return False


def test_environment_variables():
    """测试环境变量配置"""
    print("🧪 检查环境变量配置...")
    
    api_key = os.environ.get("COMPANY_OPENAI_API_KEY")
    base_url = os.environ.get("COMPANY_OPENAI_BASE_URL")
    
    if api_key:
        print(f"✅ API Key: {api_key[:10]}...")
    else:
        print("❌ 缺少 COMPANY_OPENAI_API_KEY 环境变量")
        return False
    
    if base_url:
        print(f"✅ Base URL: {base_url}")
    else:
        print("📝 Base URL: 使用默认值")
    
    return True


def main():
    """主测试函数"""
    print("🚀 开始LLM集成测试...\n")
    
    tests = [
        ("环境变量检查", test_environment_variables),
        ("LLM客户端基本功能", test_llm_client),
        ("原子问题生成", test_atom_question_generation),
        ("Embedding客户端", test_embedding_client),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n{'='*50}")
        print(f"测试: {test_name}")
        print(f"{'='*50}")
        
        try:
            result = test_func()
            results.append((test_name, result))
        except KeyboardInterrupt:
            print("\n⚠️ 用户中断测试")
            break
        except Exception as e:
            print(f"❌ 测试异常: {e}")
            results.append((test_name, False))
    
    # 汇总结果
    print(f"\n{'='*50}")
    print("测试结果汇总")
    print(f"{'='*50}")
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✅ 通过" if result else "❌ 失败"
        print(f"{test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\n总计: {passed}/{total} 个测试通过")
    
    if passed == total:
        print("🎉 所有测试通过！LLM集成正常工作。")
        return 0
    else:
        print("⚠️ 部分测试失败，请检查配置和网络连接。")
        return 1


if __name__ == "__main__":
    exit(main()) 