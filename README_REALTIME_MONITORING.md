# 实时RPM监控功能

## 🌟 新功能亮点

PIKE-RAG 现在支持**实时RPM统计打印**功能！在处理过程中可以定期输出模型的实时使用统计，帮助您更好地监控和优化API使用。

## ⚡ 快速开始

### 1. 启用实时监控

在您的配置文件中添加以下配置：

```yaml
monitoring:
  enable: true
  print_interval: 60  # 每60秒打印一次统计
  rpm_window_minutes: 1
  rpm_warning_threshold: 50
```

### 2. 运行任务

```bash
# 运行QA任务
python examples/qa.py examples/musique/configs/atomic_decompose.yml

# 运行Tagging任务  
python examples/tagging.py examples/musique/configs/tagging.yml
```

### 3. 观察实时输出

运行中您会看到类似的实时统计：

```
[14:30:15] 📊 [实时RPM统计] 运行时间: 2.1分钟 | 当前RPM: 45.2 | 平均RPM: 38.7 | 总请求: 82 | 成功率: 98.8% | 总Token: 25,847 || 最近60秒: 请求数: 48 | Token: 15,234 | 平均耗时: 1.85s | 成功率: 100.0%
```

## 📊 监控指标说明

| 指标 | 含义 |
|------|------|
| **当前RPM** | 最近1分钟的每分钟请求数 |
| **平均RPM** | 从开始到现在的平均每分钟请求数 |
| **总请求** | 累计发送的请求数量 |
| **成功率** | 请求成功的百分比 |
| **总Token** | 累计消耗的Token数量 |
| **最近X秒** | 指定时间窗口内的详细统计 |

## ⚙️ 配置选项

```yaml
monitoring:
  enable: true              # 是否启用监控
  print_interval: 60        # 实时打印间隔（秒），0=禁用
  rpm_window_minutes: 1     # RPM计算窗口（分钟）
  rpm_warning_threshold: 50 # RPM警告阈值
```

### 推荐配置

| 场景 | print_interval | 说明 |
|------|---------------|------|
| 快速测试 | 30 | 频繁输出，快速了解状态 |
| 生产环境 | 120 | 减少日志量，保持监控 |
| 调试场景 | 15 | 密切监控，快速发现问题 |
| 禁用实时打印 | 0 | 只保留最终统计报告 |

## 🎯 实际应用

### 性能调优
- 根据RPM使用情况调整`num_parallel`并发数
- 监控平均耗时识别性能瓶颈
- 实时调整请求频率避免限流

### 成本控制
- 实时监控Token消耗和成本
- 根据预算调整任务策略
- 预估完成时间和总成本

### 错误诊断
- 及时发现成功率下降
- 快速定位性能异常
- 分析RPM波动原因

## 🧪 测试功能

运行测试脚本体验实时监控：

```bash
python examples/test_realtime_monitoring.py
```

该脚本会模拟API调用并展示实时监控效果。

## 📚 详细文档

- [实时RPM监控详细指南](research_project/REALTIME_RPM_MONITORING.md)
- [API监控系统使用指南](research_project/API_MONITORING_GUIDE.md)

## 🔧 兼容性

- ✅ 支持所有现有工作流（QA、Tagging、Chunking等）
- ✅ 兼容并行执行配置
- ✅ 支持所有LLM客户端（OpenAI、Azure等）
- ✅ 线程安全，支持多线程环境
- ✅ 性能开销 < 0.5%

---

**注意**：实时监控功能默认启用，您可以通过设置`print_interval: 0`来禁用实时打印，但仍保留最终统计报告。 