#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Embedding调试脚本

用于诊断和解决embedding请求卡住的问题。
"""

import os
import sys
import time
import signal
import threading
from pathlib import Path

# 添加项目根目录到Python路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from pikerag.utils.logger import Logger
from pikerag.llm_client.company_openai_client import CompanyOpenAIEmbedding


class EmbeddingDiagnostic:
    """Embedding诊断工具"""
    
    def __init__(self, timeout=60):
        self.timeout = timeout
        self.logger = Logger("EmbeddingDiagnostic")
        self.interrupted = False
        
        # 设置信号处理器
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _signal_handler(self, signum, frame):
        """信号处理器"""
        self.logger.warning(f"接收到信号 {signum}，正在中断...")
        self.interrupted = True
    
    def check_api_config(self):
        """检查API配置"""
        self.logger.info("🔧 检查API配置...")
        
        api_key = os.environ.get("COMPANY_OPENAI_API_KEY")
        base_url = os.environ.get("COMPANY_OPENAI_BASE_URL")
        
        if not api_key:
            self.logger.error("❌ 缺少API Key: COMPANY_OPENAI_API_KEY 环境变量未设置")
            return False
        else:
            self.logger.info(f"✅ API Key: {api_key[:10]}...")
        
        if base_url:
            self.logger.info(f"✅ Base URL: {base_url}")
        else:
            self.logger.info("📝 Base URL: 使用默认值")
        
        return True
    
    def test_simple_embedding(self):
        """测试简单的embedding请求"""
        self.logger.info("🧪 测试简单embedding请求...")
        
        try:
            # 创建embedding客户端
            client_config = {
                "timeout": self.timeout
            }
            
            embedding_client = CompanyOpenAIEmbedding(
                client_config=client_config,
                logger=self.logger,
                timeout=self.timeout
            )
            
            # 测试简单文本
            test_text = "Hello, this is a test."
            self.logger.info(f"📝 测试文本: {test_text}")
            
            start_time = time.time()
            
            # 使用线程来执行embedding，以便可以设置超时
            result = []
            exception = []
            
            def embedding_worker():
                try:
                    embedding = embedding_client.embed_query(test_text)
                    result.append(embedding)
                except Exception as e:
                    exception.append(e)
            
            # 启动embedding线程
            thread = threading.Thread(target=embedding_worker)
            thread.daemon = True
            thread.start()
            
            # 等待完成或超时
            thread.join(timeout=self.timeout)
            
            duration = time.time() - start_time
            
            if thread.is_alive():
                self.logger.error(f"❌ Embedding请求超时 ({duration:.1f}s > {self.timeout}s)")
                self.logger.error("💡 建议:")
                self.logger.error("   1. 检查网络连接")
                self.logger.error("   2. 检查API服务器状态") 
                self.logger.error("   3. 尝试增加timeout设置")
                self.logger.error("   4. 检查防火墙设置")
                return False
            
            if exception:
                self.logger.error(f"❌ Embedding请求异常: {exception[0]}")
                return False
            
            if result and result[0]:
                self.logger.info(f"✅ Embedding成功完成，耗时: {duration:.2f}秒")
                self.logger.info(f"📊 向量维度: {len(result[0])}")
                return True
            else:
                self.logger.error("❌ Embedding返回空结果")
                return False
                
        except Exception as e:
            self.logger.error(f"❌ 测试过程异常: {str(e)}")
            return False
    
    def test_network_connectivity(self):
        """测试网络连通性"""
        self.logger.info("🌐 测试网络连通性...")
        
        import socket
        
        base_url = os.environ.get("COMPANY_OPENAI_BASE_URL", "https://aigc.sankuai.com")
        
        # 解析主机名和端口
        try:
            if base_url.startswith("https://"):
                host = base_url[8:].split("/")[0]
                port = 443
            elif base_url.startswith("http://"):
                host = base_url[7:].split("/")[0]
                port = 80
            else:
                host = base_url.split("/")[0]
                port = 443
            
            # 测试TCP连接
            self.logger.info(f"🔗 测试连接到 {host}:{port}")
            
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(10)
            
            start_time = time.time()
            result = sock.connect_ex((host, port))
            duration = time.time() - start_time
            
            sock.close()
            
            if result == 0:
                self.logger.info(f"✅ 网络连接正常，耗时: {duration:.2f}秒")
                return True
            else:
                self.logger.error(f"❌ 网络连接失败，错误码: {result}")
                return False
                
        except Exception as e:
            self.logger.error(f"❌ 网络测试异常: {str(e)}")
            return False
    
    def run_full_diagnostic(self):
        """运行完整诊断"""
        self.logger.info("🚀 开始Embedding诊断...")
        self.logger.info("=" * 50)
        
        all_passed = True
        
        # 1. 检查API配置
        if not self.check_api_config():
            all_passed = False
        
        self.logger.info("-" * 30)
        
        # 2. 测试网络连通性
        if not self.test_network_connectivity():
            all_passed = False
        
        self.logger.info("-" * 30)
        
        # 3. 测试embedding请求
        if not self.interrupted and not self.test_simple_embedding():
            all_passed = False
        
        self.logger.info("=" * 50)
        
        if all_passed:
            self.logger.info("🎉 所有诊断测试通过！")
            self.logger.info("💡 如果您的程序仍然卡住，请考虑:")
            self.logger.info("   1. 增加并行处理的timeout配置")
            self.logger.info("   2. 检查是否有其他进程占用网络资源")
            self.logger.info("   3. 使用更小的批次大小")
        else:
            self.logger.error("❌ 诊断发现问题，请根据上述建议进行修复")
        
        return all_passed


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Embedding调试诊断工具")
    parser.add_argument("--timeout", type=int, default=60, help="超时时间（秒），默认60秒")
    args = parser.parse_args()
    
    diagnostic = EmbeddingDiagnostic(timeout=args.timeout)
    
    try:
        success = diagnostic.run_full_diagnostic()
        exit_code = 0 if success else 1
    except KeyboardInterrupt:
        print("\n🛑 诊断被用户中断")
        exit_code = 2
    except Exception as e:
        print(f"\n💥 诊断过程异常: {str(e)}")
        exit_code = 3
    
    sys.exit(exit_code)


if __name__ == "__main__":
    main() 