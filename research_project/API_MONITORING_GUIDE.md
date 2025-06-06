# API 监控系统使用指南

## 📊 功能概述

PIKE-RAG 的 API 监控系统提供了全面的 LLM API 使用统计和性能分析功能，帮助你：

- 🚀 **实时 RPM 监控**：跟踪每分钟请求数，避免超出 API 限制
- 💰 **Token 消耗统计**：详细统计输入/输出 Token 和成本估算
- ⏱️ **性能分析**：监控请求耗时和执行效率
- ❌ **错误诊断**：分类统计错误类型和成功率
- 📋 **详细报告**：生成完整的使用报告和日志

## 🔧 配置方法

### 1. 基础配置

在你的 YAML 配置文件中添加 `monitoring` 部分：

```yaml
# Monitoring Setting (监控配置)
################################################################################
monitoring:
  # 是否启用API监控
  enable: true
  
  # RPM监控窗口时间（分钟）
  rpm_window_minutes: 1
  
  # RPM限制警告阈值（可选，用于运行时警告）
  rpm_warning_threshold: 50
```

### 2. 配置参数说明

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `enable` | bool | true | 是否启用监控功能 |
| `rpm_window_minutes` | int | 1 | RPM 统计的时间窗口（分钟） |
| `rpm_warning_threshold` | int | 50 | RPM 警告阈值 |

### 3. 多线程配置

配合多线程使用时，建议根据 API 限制调整并发数：

```yaml
tagger:
  # 多线程配置：根据API限制调整
  num_parallel: 4  # 建议值：2-8

monitoring:
  enable: true
  rpm_warning_threshold: 50  # 根据API限制设置
```

## 🚀 使用方法

### 1. 启动监控

监控功能会在工作流启动时自动开始。使用方式与之前完全相同：

```bash
# 运行 tagging 任务（自动启用监控）
python examples/tagging.py examples/musique/configs/tagging.yml
```

### 2. 实时监控信息

运行过程中，你会在日志中看到实时的监控信息：

```
[TaggingWorkflow] ✅ API监控器已启用
[APIMonitor] 🔄 开始API监控
[TaggingWorkflow] 🚀 开始标注任务
[APIMonitor] ✅ 请求完成 | 耗时: 2.34s | Token: 312 (189+123) | 模型: gpt-4o-mini
```

### 3. 执行完成报告

任务完成后，系统会自动显示详细的统计报告：

```
================================================================================
🔍 API 使用统计报告
================================================================================
📅 执行时间: 2024-01-15 14:30:25 ~ 2024-01-15 14:45:12
⏱️  总耗时: 887.23 秒 (14.8 分钟)

📊 请求统计:
   总请求数: 500
   成功请求: 498 (99.6%)
   失败请求: 2 (0.4%)
   平均RPM: 33.8

🎯 Token 消耗统计:
   总 Token 数: 156,847
   输入 Token: 89,234
   输出 Token: 67,613
   平均每请求: 313.7 tokens

💰 估算成本 (GPT-4o-mini):
   输入成本: $0.0134
   输出成本: $0.0406
   总成本: $0.0540

🤖 模型使用情况:
   gpt-4o-mini: 500 次 (100.0%)
================================================================================
```

## 📋 输出文件

### 1. 详细报告文件

系统会自动生成 JSON 格式的详细报告：

```
logs/{experiment_name}/api_monitoring_report.json
```

### 2. 报告内容结构

```json
{
  "overall_stats": {
    "start_time": 1705400625.123,
    "end_time": 1705401512.456,
    "total_duration": 887.33,
    "total_requests": 500,
    "successful_requests": 498,
    "failed_requests": 2,
    "total_prompt_tokens": 89234,
    "total_completion_tokens": 67613,
    "total_tokens": 156847,
    "avg_tokens_per_request": 313.7,
    "requests_per_minute": 33.8,
    "errors_by_type": {
      "RateLimitError": 2
    },
    "models_used": {
      "gpt-4o-mini": 500
    }
  },
  "detailed_requests": [
    {
      "timestamp": 1705400625.123,
      "duration": 2.34,
      "prompt_tokens": 189,
      "completion_tokens": 123,
      "total_tokens": 312,
      "model": "gpt-4o-mini",
      "success": true,
      "error_type": null
    }
  ]
}
```

## 🎯 高级特性

### 1. 成本估算

系统基于 GPT-4o-mini 的官方定价自动计算成本：

- **输入 Token**：$0.15 / 1M tokens
- **输出 Token**：$0.6 / 1M tokens

### 2. 线程安全

监控系统完全支持多线程环境，可以安全地与 `num_parallel` 配置配合使用。

### 3. 实时 RPM 控制

监控器提供实时 RPM 检查功能，可以在代码中使用：

```python
from pikerag.utils.monitor import get_global_monitor

monitor = get_global_monitor()
if monitor and not monitor.check_rpm_limit(max_rpm=60):
    # 处理 RPM 超限情况
    time.sleep(1)
```

### 4. 自定义监控

你也可以在代码中手动创建和使用监控器：

```python
from pikerag.utils.monitor import create_monitor

# 创建自定义监控器
monitor = create_monitor(logger=your_logger, window_minutes=5)
monitor.start_monitoring()

# 手动记录请求
monitor.record_request(
    duration=2.5,
    response=openai_response,
    model="gpt-4o-mini",
    success=True
)

monitor.stop_monitoring()
monitor.print_summary()
```

## ⚙️ 性能优化建议

### 1. RPM 配置建议

根据你的 API 套餐调整配置：

| API 套餐 | 建议 num_parallel | 建议 rpm_warning_threshold |
|----------|-------------------|----------------------------|
| 免费版   | 1-2               | 3                          |
| Pay-as-you-go | 2-4          | 60                         |
| 企业版   | 4-8               | 500                        |

### 2. 缓存优化

监控系统与现有缓存机制完全兼容：

- 缓存命中不会计入 API 请求统计
- 只有实际的 API 调用才会被监控
- 重复运行时会显著提高效率

### 3. 并发调优

根据监控数据调整并发配置：

```yaml
# 如果 RPM 经常超限，减少并发数
num_parallel: 2

# 如果 RPM 利用率较低，可以增加并发数
num_parallel: 6
```

## 🔧 故障排除

### 1. 监控功能未启用

**问题**：没有看到监控输出
**解决方案**：
```yaml
monitoring:
  enable: true  # 确保启用监控
```

### 2. Token 统计显示为 0

**问题**：Token 消耗显示为 0
**原因**：可能是缓存命中或 API 响应格式问题
**排查方法**：
- 检查日志中是否有 "解析token使用情况失败" 的警告
- 确认使用的是支持 token 统计的 LLM 客户端

### 3. RPM 警告频繁出现

**问题**：频繁收到 RPM 超限警告
**解决方案**：
```yaml
tagger:
  num_parallel: 2  # 减少并发数

monitoring:
  rpm_warning_threshold: 30  # 降低警告阈值
```

### 4. 报告文件未生成

**问题**：找不到 `api_monitoring_report.json` 文件
**解决方案**：
- 检查 `log_dir` 路径是否正确
- 确认程序有写入权限
- 查看日志中是否有文件保存失败的错误

## 📝 最佳实践

### 1. 监控配置模板

```yaml
# 生产环境推荐配置
monitoring:
  enable: true
  rpm_window_minutes: 1
  rpm_warning_threshold: 50

tagger:
  num_parallel: 4

llm_client:
  cache_config:
    auto_dump: true  # 启用缓存
```

### 2. 日志分析

定期查看监控报告，优化配置：

```bash
# 查看最新的监控报告
cat logs/*/api_monitoring_report.json | jq '.overall_stats'

# 分析平均耗时
jq '.detailed_requests[].duration' logs/*/api_monitoring_report.json | awk '{sum+=$1; count++} END {print "平均耗时:", sum/count, "秒"}'
```

### 3. 成本控制

根据监控数据制定成本控制策略：

- 设置每日/每月 Token 使用上限
- 监控单次任务的成本趋势
- 优化 prompt 设计以减少 Token 消耗

## 🤝 技术支持

如果你在使用监控功能时遇到问题：

1. 查看详细日志：`logs/{experiment_name}/`
2. 检查配置文件语法
3. 参考本文档的故障排除章节
4. 联系开发团队获取支持

---

**注意**：监控功能对性能影响极小（< 1% 开销），建议在所有生产环境中启用。 