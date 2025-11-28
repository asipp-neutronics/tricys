# 并发运行

对于包含大量仿真任务的参数扫描，逐个顺序执行可能会非常耗时。`tricys` 支持并发运行（并行计算），可以充分利用您计算机的多个 CPU 核心，显著缩短总仿真时间。

## 1. 如何启用并发

启用并发非常简单，只需在配置文件的 `simulation` 部分将 `concurrent` 标志设置为 `true`。

```json
{
    "paths": {
        ...
    },
    "simulation": {
        "model_name": "example_model.Cycle",
        ...
        "concurrent": true,
        "max_workers": 4
    },
    "simulation_parameters": {
        "blanket.TBR": "linspace:1:1.5:10"
    }
}
```

## 2. 配置项详解

### 2.1. `simulation.concurrent`

- **描述**: 是否启用并发运行。
- **类型**: 布尔值 (`true` 或 `false`)。
- **默认值**: `false`。
- **工作原理**: 当设置为 `true` 时，`tricys` 会启动一个进程池，将仿真任务（例如参数扫描中的每一次运行）分配给不同的进程并行执行。

### 2.2. `simulation.max_workers`

- **描述**: 控制用于并发执行的最大进程数（或称“工作进程”数量）。
- **类型**: 整数 (选填)。
- **默认值**: 如果不指定该参数，`tricys` 将默认使用您计算机上的 **所有可用 CPU 核心数**。
- **建议**:
    - 对于计算密集型任务，建议将此值设置为不超过您计算机的物理核心数，以获得最佳性能。
    - 如果在仿真过程中遇到内存不足的问题，可以适当调低 `max_workers` 的值，因为每个进程都会独立加载模型并消耗一定的内存。

在上面的示例中，`"concurrent": true` 和 `"max_workers": 4` 意味着 `tricys` 会创建一个包含 4 个工作进程的进程池，来并行处理由参数扫描生成的 10 个仿真任务。

!!! success "适用场景"
    并发运行不仅适用于标准的参数扫描，也同样适用于[协同仿真](co_simulation_module.md)和[自动分析](../tricys_analysis/index.md)流程，能够全面提升 `tricys` 的执行效率。
