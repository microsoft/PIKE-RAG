# 实时RPM监控功能使用指南

## 📊 功能概述

PIKE-RAG 新增了**实时RPM统计打印机制**，可以在处理过程中定期输出模型的实时使用统计，包括：

- 🔄 **实时RPM监控**：当前每分钟请求数和平均RPM
- ⏱️ **执行时间统计**：已运行时间和预估剩余时间
- 📈 **Token消耗跟踪**：实时Token使用量和成本估算
- ✅ **成功率监控**：请求成功率和错误统计
- 📊 **窗口统计**：最近时间窗口内的详细性能指标

## 🔧 配置方法

### 1. 基础配置

在你的YAML配置文件中添加`print_interval`参数：

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

  # 实时统计打印间隔（秒），0表示禁用实时打印
  print_interval: 60
```

### 2. 配置参数说明

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `print_interval` | int | 60 | 实时统计打印间隔（秒） |
| `enable` | bool | true | 是否启用监控功能 |
| `rpm_window_minutes` | int | 1 | RPM统计的时间窗口（分钟） |
| `rpm_warning_threshold` | int | 50 | RPM警告阈值 |

### 3. 不同场景的推荐配置

#### 快速测试场景
```yaml
monitoring:
  enable: true
  print_interval: 30  # 30秒打印一次，快速了解状态
  rpm_window_minutes: 1
  rpm_warning_threshold: 10
```

#### 生产环境场景
```yaml
monitoring:
  enable: true
  print_interval: 120  # 2分钟打印一次，减少日志量
  rpm_window_minutes: 1
  rpm_warning_threshold: 50
```

#### 调试场景
```yaml
monitoring:
  enable: true
  print_interval: 15  # 15秒打印一次，密切监控
  rpm_window_minutes: 1
  rpm_warning_threshold: 20
```

#### 禁用实时打印
```yaml
monitoring:
  enable: true
  print_interval: 0  # 设置为0禁用实时打印
  rpm_window_minutes: 1
  rpm_warning_threshold: 50
```

## 🚀 使用示例

### 1. 运行QA任务

```bash
# 使用带有实时监控的配置文件
python examples/qa.py examples/musique/configs/atomic_decompose.yml
```

### 2. 运行Tagging任务

```bash
# 使用带有实时监控的配置文件
python examples/tagging.py examples/musique/configs/tagging.yml
```

### 3. 实时输出示例

运行过程中，你会看到如下的实时统计输出：

```
[14:30:15] 📊 [实时RPM统计] 运行时间: 2.1分钟 | 当前RPM: 45.2 | 平均RPM: 38.7 | 总请求: 82 | 成功率: 98.8% | 总Token: 25,847 || 最近60秒: 请求数: 48 | Token: 15,234 | 平均耗时: 1.85s | 成功率: 100.0%

[14:31:15] 📊 [实时RPM统计] 运行时间: 3.1分钟 | 当前RPM: 42.8 | 平均RPM: 40.1 | 总请求: 125 | 成功率: 99.2% | 总Token: 39,156 || 最近60秒: 请求数: 43 | Token: 13,309 | 平均耗时: 1.92s | 成功率: 100.0%

[14:32:15] 📊 [实时RPM统计] 运行时间: 4.1分钟 | 当前RPM: 39.6 | 平均RPM: 39.8 | 总请求: 164 | 成功率: 98.8% | 总Token: 51,203 || 最近60秒: 请求数: 39 | Token: 12,047 | 平均耗时: 2.01s | 成功率: 97.4%
```

## 📊 输出信息解读

### 总体统计（左侧部分）
- **运行时间**：任务开始到当前的总执行时间
- **当前RPM**：最近1分钟内的每分钟请求数
- **平均RPM**：从任务开始到现在的平均每分钟请求数
- **总请求**：从任务开始到现在的总请求数
- **成功率**：总体请求成功率
- **总Token**：累计消耗的Token数量

### 窗口统计（右侧部分）
- **最近X秒**：在配置的时间窗口内的统计
- **请求数**：窗口内的请求数量
- **Token**：窗口内消耗的Token数量
- **平均耗时**：窗口内每个请求的平均耗时
- **成功率**：窗口内的请求成功率

## 🎯 实际应用场景

### 1. 性能调优

通过实时监控，你可以：
- 调整`num_parallel`参数优化并发度
- 根据RPM使用情况调整请求频率
- 监控平均耗时识别性能瓶颈

```yaml
# 根据实时监控调整并发配置
workflow:
  args:
    num_parallel: 4  # 如果当前RPM较低，可以增加并发数

monitoring:
  print_interval: 30  # 更频繁的监控以便及时调整
  rpm_warning_threshold: 80
```

### 2. 成本控制

实时监控Token消耗：
- 预估总成本和剩余成本
- 控制单位时间内的Token消耗
- 根据成本预算调整任务策略

### 3. 错误诊断

快速发现问题：
- 成功率下降时及时发现
- 平均耗时异常时快速定位
- RPM波动时分析原因

### 4. 进度跟踪

长时间任务的进度监控：
- 估算剩余完成时间
- 监控处理效率变化
- 及时调整策略

## ⚙️ 高级功能

### 1. 自定义输出格式

你可以通过继承`APIMonitor`类来自定义输出格式：

```python
from pikerag.utils.monitor import APIMonitor

class CustomAPIMonitor(APIMonitor):
    def _print_realtime_stats(self):
        # 自定义输出格式
        stats = self.get_stats()
        custom_message = f"自定义统计: RPM={self.get_current_rpm():.1f}"
        if self.logger:
            self.logger.info(custom_message, tag="CustomMonitor")
```

### 2. 集成外部监控系统

```python
def _print_realtime_stats(self):
    # 调用父类方法进行标准输出
    super()._print_realtime_stats()
    
    # 发送到外部监控系统
    stats = self.get_stats()
    send_to_monitoring_system({
        "rpm": self.get_current_rpm(),
        "total_requests": len(self.requests),
        "success_rate": stats.successful_requests / len(self.requests) * 100
    })
```

### 3. 条件性输出

根据特定条件调整输出频率：

```python
def _print_realtime_stats(self):
    current_rpm = self.get_current_rpm()
    
    # RPM异常时增加输出频率
    if current_rpm > 80 or current_rpm < 5:
        # 输出详细信息
        super()._print_realtime_stats()
    else:
        # 输出简化信息
        if self.logger:
            self.logger.info(f"📊 RPM: {current_rpm:.1f}", tag="APIMonitor")
```

## 🔧 故障排除

### 1. 实时统计未显示

**问题**：运行任务时没有看到实时统计输出

**排查步骤**：
1. 检查`monitoring.enable`是否为`true`
2. 检查`monitoring.print_interval`是否大于0
3. 确认任务有实际的API调用
4. 查看日志级别设置

**解决方案**：
```yaml
monitoring:
  enable: true
  print_interval: 60  # 确保大于0
```

### 2. 输出过于频繁

**问题**：实时统计输出太频繁，影响日志可读性

**解决方案**：
```yaml
monitoring:
  print_interval: 120  # 增加间隔到2分钟
```

### 3. 统计数据不准确

**问题**：显示的统计数据与预期不符

**排查**：
- 检查是否启用了缓存，缓存命中不会计入统计
- 确认API客户端正确集成了监控功能
- 查看详细日志中的单个请求记录

### 4. 内存使用过高

**问题**：长时间运行后内存使用过高

**解决方案**：
- 实时统计会保存所有请求记录，超长任务可能消耗较多内存
- 可以考虑定期清理历史记录或使用采样统计

## 📝 最佳实践

### 1. 选择合适的打印间隔
- **短任务（< 10分钟）**：30-60秒
- **中等任务（10分钟-2小时）**：60-120秒
- **长任务（> 2小时）**：120-300秒

### 2. 结合日志级别使用
```yaml
# 生产环境配置
monitoring:
  print_interval: 180  # 3分钟间隔
  
# 调试环境配置  
monitoring:
  print_interval: 30   # 30秒间隔
```

### 3. 监控关键指标
重点关注这些指标的变化：
- **RPM突然下降**：可能遇到速率限制
- **成功率下降**：可能遇到API错误
- **平均耗时增加**：可能网络或服务延迟
- **Token消耗异常**：可能prompt设计问题

### 4. 成本控制策略
```yaml
monitoring:
  print_interval: 60
  rpm_warning_threshold: 30  # 保守的RPM限制

workflow:
  args:
    num_parallel: 2  # 适中的并发数
```

---

**注意**：实时RPM监控功能会增加少量系统开销（< 0.5%），建议在生产环境中启用以便监控系统状态。 