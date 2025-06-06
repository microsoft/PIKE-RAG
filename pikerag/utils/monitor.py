# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import json
import time
from collections import defaultdict
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
import threading

from pikerag.utils.logger import Logger


@dataclass
class RequestStats:
    """单个请求的统计信息"""
    timestamp: float
    duration: float  # 秒
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    model: str
    success: bool
    error_type: Optional[str] = None


@dataclass
class OverallStats:
    """整体运行统计信息"""
    start_time: float
    end_time: float
    total_duration: float
    total_requests: int
    successful_requests: int
    failed_requests: int
    total_prompt_tokens: int
    total_completion_tokens: int
    total_tokens: int
    avg_tokens_per_request: float
    requests_per_minute: float
    errors_by_type: Dict[str, int]
    models_used: Dict[str, int]


class APIMonitor:
    """API请求监控器，提供RPM监控、Token统计和性能分析"""
    
    def __init__(self, logger: Optional[Logger] = None, window_minutes: int = 1):
        """
        Args:
            logger: 日志记录器
            window_minutes: RPM统计窗口时间（分钟）
        """
        self.logger = logger
        self.window_minutes = window_minutes
        self.requests: List[RequestStats] = []
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None
        self._lock = threading.Lock()
        
        # 实时RPM监控
        self._rpm_window: List[float] = []
        
    def start_monitoring(self):
        """开始监控"""
        self.start_time = time.time()
        if self.logger:
            self.logger.info("🔄 开始API监控", tag="APIMonitor")
            
    def stop_monitoring(self):
        """停止监控"""
        self.end_time = time.time()
        if self.logger:
            self.logger.info("⏹️ 停止API监控", tag="APIMonitor")
    
    def extract_token_usage(self, response: Any) -> Tuple[int, int, int]:
        """从OpenAI响应中提取token使用情况
        
        Returns:
            (prompt_tokens, completion_tokens, total_tokens)
        """
        try:
            if hasattr(response, 'usage') and response.usage:
                usage = response.usage
                prompt_tokens = getattr(usage, 'prompt_tokens', 0)
                completion_tokens = getattr(usage, 'completion_tokens', 0)
                total_tokens = getattr(usage, 'total_tokens', 0)
                return prompt_tokens, completion_tokens, total_tokens
            return 0, 0, 0
        except Exception as e:
            if self.logger:
                self.logger.warning(f"解析token使用情况失败: {e}", tag="APIMonitor")
            return 0, 0, 0
    
    def record_request(
        self, 
        duration: float, 
        response: Any = None, 
        model: str = "unknown",
        success: bool = True,
        error_type: Optional[str] = None
    ):
        """记录一个API请求
        
        Args:
            duration: 请求耗时（秒）
            response: OpenAI响应对象
            model: 使用的模型名称
            success: 请求是否成功
            error_type: 错误类型（如果失败）
        """
        timestamp = time.time()
        
        # 提取token使用情况
        prompt_tokens, completion_tokens, total_tokens = self.extract_token_usage(response)
        
        # 创建请求统计
        stats = RequestStats(
            timestamp=timestamp,
            duration=duration,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            model=model,
            success=success,
            error_type=error_type
        )
        
        with self._lock:
            self.requests.append(stats)
            self._rpm_window.append(timestamp)
            
            # 清理RPM窗口，只保留最近的记录
            cutoff_time = timestamp - (self.window_minutes * 60)
            self._rpm_window = [t for t in self._rpm_window if t > cutoff_time]
            
        # 日志记录详细信息
        if self.logger:
            status = "✅" if success else "❌"
            self.logger.debug(
                f"{status} 请求完成 | 耗时: {duration:.2f}s | "
                f"Token: {total_tokens} ({prompt_tokens}+{completion_tokens}) | "
                f"模型: {model}",
                tag="APIMonitor"
            )
    
    def get_current_rpm(self) -> float:
        """获取当前RPM（每分钟请求数）"""
        with self._lock:
            current_time = time.time()
            cutoff_time = current_time - (self.window_minutes * 60)
            recent_requests = [t for t in self._rpm_window if t > cutoff_time]
            return len(recent_requests) / self.window_minutes
    
    def check_rpm_limit(self, max_rpm: int = 60) -> bool:
        """检查是否超过RPM限制
        
        Args:
            max_rpm: 最大RPM限制
            
        Returns:
            True if within limit, False if exceeded
        """
        current_rpm = self.get_current_rpm()
        if current_rpm > max_rpm:
            if self.logger:
                self.logger.warning(
                    f"⚠️ RPM超限: {current_rpm:.1f}/{max_rpm}",
                    tag="APIMonitor"
                )
            return False
        return True
    
    def get_stats(self) -> OverallStats:
        """获取整体统计信息"""
        with self._lock:
            if not self.requests:
                return OverallStats(
                    start_time=self.start_time or 0,
                    end_time=self.end_time or 0,
                    total_duration=0,
                    total_requests=0,
                    successful_requests=0,
                    failed_requests=0,
                    total_prompt_tokens=0,
                    total_completion_tokens=0,
                    total_tokens=0,
                    avg_tokens_per_request=0,
                    requests_per_minute=0,
                    errors_by_type={},
                    models_used={}
                )
            
            # 基本统计
            total_requests = len(self.requests)
            successful_requests = sum(1 for r in self.requests if r.success)
            failed_requests = total_requests - successful_requests
            
            # Token统计
            total_prompt_tokens = sum(r.prompt_tokens for r in self.requests)
            total_completion_tokens = sum(r.completion_tokens for r in self.requests)
            total_tokens = sum(r.total_tokens for r in self.requests)
            avg_tokens_per_request = total_tokens / total_requests if total_requests > 0 else 0
            
            # 时间统计
            total_duration = (self.end_time or time.time()) - (self.start_time or 0)
            requests_per_minute = (total_requests / total_duration * 60) if total_duration > 0 else 0
            
            # 错误类型统计
            errors_by_type = defaultdict(int)
            for r in self.requests:
                if not r.success and r.error_type:
                    errors_by_type[r.error_type] += 1
            
            # 模型使用统计
            models_used = defaultdict(int)
            for r in self.requests:
                models_used[r.model] += 1
            
            return OverallStats(
                start_time=self.start_time or 0,
                end_time=self.end_time or 0,
                total_duration=total_duration,
                total_requests=total_requests,
                successful_requests=successful_requests,
                failed_requests=failed_requests,
                total_prompt_tokens=total_prompt_tokens,
                total_completion_tokens=total_completion_tokens,
                total_tokens=total_tokens,
                avg_tokens_per_request=avg_tokens_per_request,
                requests_per_minute=requests_per_minute,
                errors_by_type=dict(errors_by_type),
                models_used=dict(models_used)
            )
    
    def print_summary(self):
        """打印统计摘要"""
        stats = self.get_stats()
        
        print("\n" + "="*80)
        print("🔍 API 使用统计报告")
        print("="*80)
        
        # 基本信息
        start_dt = datetime.fromtimestamp(stats.start_time)
        end_dt = datetime.fromtimestamp(stats.end_time)
        print(f"📅 执行时间: {start_dt.strftime('%Y-%m-%d %H:%M:%S')} ~ {end_dt.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"⏱️  总耗时: {stats.total_duration:.2f} 秒 ({stats.total_duration/60:.1f} 分钟)")
        
        # 请求统计
        print(f"\n📊 请求统计:")
        print(f"   总请求数: {stats.total_requests}")
        print(f"   成功请求: {stats.successful_requests} ({stats.successful_requests/stats.total_requests*100:.1f}%)")
        print(f"   失败请求: {stats.failed_requests} ({stats.failed_requests/stats.total_requests*100:.1f}%)")
        print(f"   平均RPM: {stats.requests_per_minute:.1f}")
        
        # Token统计
        print(f"\n🎯 Token 消耗统计:")
        print(f"   总 Token 数: {stats.total_tokens:,}")
        print(f"   输入 Token: {stats.total_prompt_tokens:,}")
        print(f"   输出 Token: {stats.total_completion_tokens:,}")
        print(f"   平均每请求: {stats.avg_tokens_per_request:.1f} tokens")
        
        # 估算成本（基于GPT-4o-mini定价）
        # 输入: $0.15/1M tokens, 输出: $0.6/1M tokens
        input_cost = stats.total_prompt_tokens / 1000000 * 0.15
        output_cost = stats.total_completion_tokens / 1000000 * 0.6
        total_cost = input_cost + output_cost
        print(f"\n💰 估算成本 (GPT-4o-mini):")
        print(f"   输入成本: ${input_cost:.4f}")
        print(f"   输出成本: ${output_cost:.4f}")
        print(f"   总成本: ${total_cost:.4f}")
        
        # 模型使用情况
        if stats.models_used:
            print(f"\n🤖 模型使用情况:")
            for model, count in stats.models_used.items():
                percentage = count / stats.total_requests * 100
                print(f"   {model}: {count} 次 ({percentage:.1f}%)")
        
        # 错误统计
        if stats.errors_by_type:
            print(f"\n❌ 错误统计:")
            for error_type, count in stats.errors_by_type.items():
                percentage = count / stats.total_requests * 100
                print(f"   {error_type}: {count} 次 ({percentage:.1f}%)")
        
        print("="*80)
        
        # 如果有logger，也记录到日志
        if self.logger:
            self.logger.info(f"📊 执行完成 - 总请求: {stats.total_requests}, Token: {stats.total_tokens:,}, 耗时: {stats.total_duration:.1f}s", tag="APIMonitor")
    
    def save_detailed_report(self, filepath: str):
        """保存详细报告到文件"""
        stats = self.get_stats()
        report = {
            "overall_stats": asdict(stats),
            "detailed_requests": [asdict(req) for req in self.requests]
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        
        if self.logger:
            self.logger.info(f"📋 详细报告已保存: {filepath}", tag="APIMonitor")


# 全局监控器实例
_global_monitor: Optional[APIMonitor] = None


def get_global_monitor() -> Optional[APIMonitor]:
    """获取全局监控器实例"""
    return _global_monitor


def set_global_monitor(monitor: APIMonitor):
    """设置全局监控器实例"""
    global _global_monitor
    _global_monitor = monitor


def create_monitor(logger: Optional[Logger] = None, window_minutes: int = 1) -> APIMonitor:
    """创建并设置全局监控器"""
    monitor = APIMonitor(logger=logger, window_minutes=window_minutes)
    set_global_monitor(monitor)
    return monitor 