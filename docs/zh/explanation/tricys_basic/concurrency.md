
# TRICYS 并发执行

TRICYS 旨在充分利用现代多核处理器的计算能力，通过并发执行来显著缩短参数扫描和批量仿真的总耗时。根据仿真类型的不同，TRICYS 会智能地采用两种不同的并发策略：**多线程**（用于标准仿真）和**多进程**（用于协同仿真）。

您可以在 `config.json` 中通过以下参数来控制并发行为：

```json
"simulation": {
  "concurrent": true,  // 设置为 true 来开启并发模式
  "max_workers": 8     // 设置最大并发工作单元数，默认为 CPU 核心数
}
```

## 1. 标准仿真：多线程并发 

对于不涉及协同仿真的标准参数扫描任务，TRICYS 默认使用**多线程模型**。

### 1.1. 工作原理

TRICYS 使用 Python 的 `concurrent.futures.ThreadPoolExecutor` 来管理一个工作线程池。每个仿真任务（即一组特定的参数）会被提交到线程池中，由一个空闲的线程来执行。

```python
# 简化版原理代码
from concurrent.futures import ThreadPoolExecutor

with ThreadPoolExecutor(max_workers=8) as executor:
    # _run_single_job 是每个线程要执行的任务
    futures = [executor.submit(_run_single_job, config, params, i) for i, params in enumerate(jobs)]
    # ... 等待并收集结果 ...
```

### 1.2. 为何使用多线程？

虽然 Python 存在全局解释器锁（GIL），使得在单个进程中无法实现真正的 CPU 密集型代码并行，但对于标准 Modelica 仿真场景，多线程仍然是高效的选择。原因如下：

*   **I/O 密集型**：调用 `OMPython` 运行仿真本质上是启动一个外部的 `simulate.exe` 可执行文件。Python 线程在此期间主要处于**等待状态**（I/O-bound），等待外部进程完成计算并写回结果文件。GIL 在此期间会被释放，允许其他线程运行。
*   **低开销**：线程比进程更轻量，创建和销毁的速度更快，占用的系统资源也更少。对于数量众多但每次仿真耗时不是特别长的任务，使用线程可以减少任务调度的额外开销。

## 2. 协同仿真：多进程并发 

对于复杂的协同仿真任务，TRICYS 则切换到更健壮的**多进程模型**。

### 2.1. 工作原理

TRICYS 使用 `concurrent.futures.ProcessPoolExecutor` 来创建一个进程池。每个协同仿真任务（`_run_co_simulation`）都在一个完全独立的子进程中运行。

```python
# 简化版原理代码
from concurrent.futures import ProcessPoolExecutor

with ProcessPoolExecutor(max_workers=8) as executor:
    # _run_co_simulation 在一个独立的进程中执行
    futures = [executor.submit(_run_co_simulation, config, params, i) for i, params in enumerate(jobs)]
    # ... 等待并收集结果 ...
```

### 2.2. 为何必须使用多进程？

协同仿真的工作流程涉及到对文件系统的**大量读写和修改**，这是其必须使用多进程的核心原因：

*   **环境隔离**：每个协同仿真任务都需要一个纯净、独立的运行环境。它会将原始 Modelica 模型包**复制**到一个临时目录，然后对其进行修改（例如，生成拦截器或直接替换模型）。如果使用多线程，多个任务会同时修改共享的模型文件，导致文件损坏和结果错乱。
*   **无状态冲突**：进程拥有独立的内存空间和文件句柄。这确保了一个任务对模型代码的修改、外部 Python 处理器（handler）的运行以及生成的所有中间文件，都**不会对其他并行任务产生任何干扰**。
*   **规避 GIL**：协同仿真中的 Python 处理器（handler）本身可能是计算密集型的。在多进程模型下，每个进程都有自己的 Python 解释器和 GIL，可以实现真正的并行计算，充分利用所有 CPU 核心。

## 3. 性能与实践建议

*   **合理设置 `max_workers`**：
    *   对于 I/O 密集型的标准仿真，可以设置比 CPU 核心数稍多的 `max_workers`（例如，核心数 * 1.5）。
    *   对于 CPU 密集型的协同仿真，`max_workers` 通常设置为等于或略小于您的 CPU 核心数，以避免过度的上下文切换开销。
    *   始终要考虑**内存限制**。每个进程都会消耗相当数量的内存，如果 `max_workers` 设置得太高，可能会导致内存耗尽。

*   **任务粒度**：
    *   并发执行本身有开销。如果单次仿真耗时非常短（例如，少于 1 秒），并发带来的性能提升可能还不足以抵消任务调度的开销。在这种情况下，串行执行 (`"concurrent": false`) 可能更快。

*   **监控系统资源**：
    *   在运行大规模并发任务时，建议使用系统监控工具（如任务管理器或 `htop`）来观察 CPU 和内存的使用情况，以便调整 `max_workers` 到最佳值。


---