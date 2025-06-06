#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
实时RPM监控功能测试脚本

这个脚本用于测试和演示PIKE-RAG的实时RPM监控功能。
它模拟API调用并展示监控输出效果。
"""

import time
import random
from typing import Any
from dataclasses import dataclass

from pikerag.utils.monitor import create_monitor
from pikerag.utils.logger import Logger


@dataclass
class MockResponse:
    """模拟API响应对象"""
    usage: Any = None
    id: str = "mock_response"
    model: str = "gpt-4o-mini"
    created: int = 0


@dataclass
class MockUsage:
    """模拟Token使用情况"""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


def simulate_api_call(duration_range=(1.0, 3.0), token_range=(100, 500), success_rate=0.95):
    """模拟一个API调用
    
    Args:
        duration_range: 耗时范围（秒）
        token_range: Token数量范围
        success_rate: 成功率
    
    Returns:
        (duration, response, success, error_type)
    """
    # 模拟请求耗时
    duration = random.uniform(*duration_range)
    time.sleep(duration)
    
    # 模拟成功/失败
    success = random.random() < success_rate
    error_type = None if success else "RateLimitError"
    
    # 模拟响应
    response = None
    if success:
        total_tokens = random.randint(*token_range)
        prompt_tokens = int(total_tokens * 0.6)
        completion_tokens = total_tokens - prompt_tokens
        
        usage = MockUsage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens
        )
        
        response = MockResponse(usage=usage)
    
    return duration, response, success, error_type


def test_realtime_monitoring():
    """测试实时RPM监控功能"""
    print("🚀 开始测试实时RPM监控功能")
    print("=" * 60)
    
    # 创建日志器
    logger = Logger(name="TestMonitor", dump_mode="w", dump_folder="logs/test_monitoring")
    
    # 创建监控器（每30秒打印一次统计）
    monitor = create_monitor(
        logger=logger,
        window_minutes=1,
        print_interval=30  # 30秒间隔，便于测试观察
    )
    
    # 开始监控
    monitor.start_monitoring()
    
    print("📊 监控已启动，将模拟API调用...")
    print("💡 观察日志中的实时RPM统计输出")
    print("⏱️  测试将运行2分钟，每次调用间隔1-5秒")
    print("-" * 60)
    
    try:
        # 模拟运行2分钟
        start_time = time.time()
        request_count = 0
        
        while time.time() - start_time < 120:  # 运行2分钟
            # 模拟API调用
            duration, response, success, error_type = simulate_api_call(
                duration_range=(0.5, 2.0),  # 减少单次耗时以便更快看到效果
                token_range=(50, 300),
                success_rate=0.9
            )
            
            # 记录到监控器
            monitor.record_request(
                duration=duration,
                response=response,
                model="gpt-4o-mini",
                success=success,
                error_type=error_type
            )
            
            request_count += 1
            
            # 打印当前进度（每10个请求打印一次）
            if request_count % 10 == 0:
                current_rpm = monitor.get_current_rpm()
                print(f"📈 已发送 {request_count} 个请求，当前RPM: {current_rpm:.1f}")
            
            # 随机间隔1-5秒
            time.sleep(random.uniform(1, 5))
        
    except KeyboardInterrupt:
        print("\n⚡ 用户中断测试")
    
    finally:
        # 停止监控
        monitor.stop_monitoring()
        
        print("\n" + "=" * 60)
        print("🏁 测试完成")
        
        # 打印最终统计
        monitor.print_summary()
        
        # 保存详细报告
        monitor.save_detailed_report("logs/test_monitoring/test_realtime_monitoring_report.json")
        
        print("\n📋 详细报告已保存到: logs/test_monitoring/test_realtime_monitoring_report.json")
        print("📂 日志文件保存在: logs/test_monitoring/")


def test_different_scenarios():
    """测试不同场景的监控配置"""
    scenarios = [
        {
            "name": "高频监控（5秒间隔）",
            "print_interval": 5,
            "duration": 30,
            "call_interval": (0.5, 1.5)
        },
        {
            "name": "标准监控（30秒间隔）",
            "print_interval": 30,
            "duration": 60,
            "call_interval": (1.0, 3.0)
        },
        {
            "name": "禁用实时打印",
            "print_interval": 0,
            "duration": 20,
            "call_interval": (0.5, 2.0)
        }
    ]
    
    for i, scenario in enumerate(scenarios, 1):
        print(f"\n🔄 场景 {i}: {scenario['name']}")
        print("-" * 40)
        
        logger = Logger(name=f"TestScenario{i}", dump_mode="w", dump_folder=f"logs/test_scenario_{i}")
        
        monitor = create_monitor(
            logger=logger,
            window_minutes=1,
            print_interval=scenario['print_interval']
        )
        
        monitor.start_monitoring()
        
        start_time = time.time()
        request_count = 0
        
        while time.time() - start_time < scenario['duration']:
            duration, response, success, error_type = simulate_api_call(
                duration_range=(0.2, 1.0),
                token_range=(20, 200),
                success_rate=0.95
            )
            
            monitor.record_request(
                duration=duration,
                response=response,
                model="gpt-4o-mini",
                success=success,
                error_type=error_type
            )
            
            request_count += 1
            time.sleep(random.uniform(*scenario['call_interval']))
        
        monitor.stop_monitoring()
        print(f"✅ 场景 {i} 完成，共发送 {request_count} 个请求")


if __name__ == "__main__":
    print("🧪 PIKE-RAG 实时RPM监控功能测试")
    print("=" * 60)
    
    choice = input("选择测试模式:\n1. 基础测试（推荐）\n2. 多场景测试\n请输入 1 或 2: ").strip()
    
    if choice == "1":
        test_realtime_monitoring()
    elif choice == "2":
        test_different_scenarios()
    else:
        print("❌ 无效选择，运行基础测试")
        test_realtime_monitoring()
    
    print("\n🎉 所有测试完成！")
    print("💡 你可以查看生成的日志文件来了解实时监控的详细输出") 