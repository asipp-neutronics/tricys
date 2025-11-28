# TRICYS Concurrent Execution

TRICYS is designed to fully leverage the computational power of modern multi-core processors by using concurrent execution to significantly reduce the total time required for parameter sweeps and batch simulations. Depending on the simulation type, TRICYS intelligently adopts two different concurrency strategies: **multi-threading** (for standard simulations) and **multi-processing** (for co-simulations).

You can control the concurrency behavior in `config.json` with the following parameters:

```json
"simulation": {
  "concurrent": true,  // Set to true to enable concurrent mode
  "max_workers": 8     // Set the maximum number of concurrent workers, defaults to the number of CPU cores
}
```

## 1. Standard Simulation: Multi-threaded Concurrency

For standard parameter sweep tasks that do not involve co-simulation, TRICYS defaults to a **multi-threading model**.

### 1.1. How It Works

TRICYS uses Python's `concurrent.futures.ThreadPoolExecutor` to manage a pool of worker threads. Each simulation task (i.e., a specific set of parameters) is submitted to the thread pool and executed by an available thread.

```python
# Simplified conceptual code
from concurrent.futures import ThreadPoolExecutor

with ThreadPoolExecutor(max_workers=8) as executor:
    # _run_single_job is the task to be executed by each thread
    futures = [executor.submit(_run_single_job, config, params, i) for i, params in enumerate(jobs)]
    # ... wait for and collect results ...
```

### 1.2. Why Use Multi-threading?

Although Python has a Global Interpreter Lock (GIL) that prevents true parallelism for CPU-bound code within a single process, multi-threading is still an efficient choice for standard Modelica simulation scenarios. Here's why:

*   **I/O-Bound**: Calling `OMPython` to run a simulation is essentially launching an external `simulate.exe` executable. During this time, the Python thread is mostly in a **waiting state** (I/O-bound), waiting for the external process to complete its calculations and write back the result file. The GIL is released during this period, allowing other threads to run.
*   **Low Overhead**: Threads are more lightweight than processes. They are created and destroyed faster and consume fewer system resources. For a large number of tasks where each simulation is not particularly time-consuming, using threads can reduce the overhead of task scheduling.

## 2. Co-simulation: Multi-process Concurrency

For complex co-simulation tasks, TRICYS switches to a more robust **multi-processing model**.

### 2.1. How It Works

TRICYS uses `concurrent.futures.ProcessPoolExecutor` to create a process pool. Each co-simulation task (`_run_co_simulation`) runs in a completely separate child process.

```python
# Simplified conceptual code
from concurrent.futures import ProcessPoolExecutor

with ProcessPoolExecutor(max_workers=8) as executor:
    # _run_co_simulation is executed in a separate process
    futures = [executor.submit(_run_co_simulation, config, params, i) for i, params in enumerate(jobs)]
    # ... wait for and collect results ...
```

### 2.2. Why Must Multi-processing Be Used?

The co-simulation workflow involves **extensive reading, writing, and modification of the file system**, which is the core reason why it must use multi-processing:

*   **Environment Isolation**: Each co-simulation task requires a clean, independent execution environment. It **copies** the original Modelica model package to a temporary directory and then modifies it (e.g., by generating an interceptor or replacing a model directly). If multi-threading were used, multiple tasks would simultaneously modify the shared model files, leading to file corruption and incorrect results.
*   **State Conflict Avoidance**: Processes have independent memory spaces and file handles. This ensures that modifications to model code, the execution of external Python handlers, and all generated intermediate files by one task **will not interfere with other parallel tasks**.
*   **Bypassing the GIL**: The Python handlers in co-simulation can themselves be computationally intensive. In a multi-processing model, each process has its own Python interpreter and GIL, enabling true parallel computation and full utilization of all CPU cores.

## 3. Performance and Practical Recommendations

*   **Set `max_workers` Reasonably**:
    *   For I/O-bound standard simulations, you can set `max_workers` to be slightly higher than the number of CPU cores (e.g., `cores * 1.5`).
    *   For CPU-bound co-simulations, `max_workers` is typically set to be equal to or slightly less than your number of CPU cores to avoid excessive context-switching overhead.
    *   Always consider **memory limits**. Each process consumes a significant amount of memory. If `max_workers` is set too high, it may lead to memory exhaustion.

*   **Task Granularity**:
    *   Concurrency itself has overhead. If a single simulation is very short (e.g., less than a second), the performance gain from concurrency may not be enough to offset the scheduling overhead. In this case, serial execution (`"concurrent": false`) might be faster.

*   **Monitor System Resources**:
    *   When running large-scale concurrent tasks, it is advisable to use system monitoring tools (like Task Manager or `htop`) to observe CPU and memory usage, which can help in tuning `max_workers` to its optimal value.